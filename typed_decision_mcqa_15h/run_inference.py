"""Resumable real inference worker. Never reads evaluation gold."""
import argparse,os,time,math,threading,subprocess
from pathlib import Path
from utils import ROOT,read,save,digest,event,now,file_hash
from schema import validate_public

class GPUSampler:
    def __init__(self,out): self.out=out;self.done=threading.Event();self.samples=[]
    def loop(self):
        while not self.done.is_set():
            try:
                r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                                 capture_output=True,text=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                vals=r.stdout.strip().split(',');self.samples.append({'time':now(),'device_used_mib':int(vals[0]),'utilization_pct':int(vals[1])})
            except Exception as e: self.samples.append({'time':now(),'error':str(e)})
            self.done.wait(1)
    def __enter__(self): self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.done.set();self.thread.join(12);save(self.out,{'label':'sampled total GPU device memory; includes other processes',
                 'interval_seconds':1,'samples':self.samples,'sampled_max_device_used_mib':max([s['device_used_mib'] for s in self.samples if 'device_used_mib' in s],default=None)})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--limit',type=int);a=ap.parse_args()
    if not a.pilot and not (ROOT/'protocol_freeze.json').exists(): raise RuntimeError('Formal inference requires frozen protocol')
    import torch
    from adapters import load_adapter
    settings=read(ROOT/'configs/protocol.json') if (ROOT/'configs/protocol.json').exists() else None
    modelpath=ROOT/'models'/('laya' if a.model=='laya_base' else a.model)
    if a.model=='laya_finetuned': modelpath=ROOT/'checkpoints/laya_final'
    phase='pilot' if a.pilot else 'formal';run_id=a.model+'_'+phase+'_'+now().replace(':','').replace('.','')
    out=ROOT/('pilot_raw' if a.pilot else 'raw')/a.model
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for d in ['arc','csqa','obqa']:
        if a.pilot:
            # Immediately strip all train/pilot metadata including the answer before the adapter call.
            rows += [{'item_id':r['item_id'],'dataset':r['dataset'],'split':r['split'],'public':r['public']} for r in read(ROOT/'data/pilot'/(d+'.json'))]
        else: rows += read(ROOT/'data/public_eval'/(d+'.json'))
    if a.limit is not None:
        if not a.pilot: raise ValueError('Formal item subset is exclusively fixed by frozen manifest')
        rows=rows[:a.limit]
    protocol_hash=digest(settings)
    with GPUSampler(ROOT/'resource_logs'/(run_id+'.json')):
        t=time.perf_counter();adapter=load_adapter(a.model,modelpath,settings);torch.cuda.synchronize()
        load_seconds=time.perf_counter()-t
        checkpoint_hash=file_hash(modelpath/'model.safetensors') if a.model.startswith('laya') else None
        save(out/('run_metadata_'+run_id+'.json'),{'run_id':run_id,'phase':phase,'started_at':now(),'model_path':str(modelpath),'checkpoint_sha256':checkpoint_hash,
                'load_seconds':load_seconds,'protocol_hash':protocol_hash,'adapter':adapter.metadata,'model_revisions':read(ROOT/'sources/revisions.json')})
        # Warm-up uses train-only pilot, never formal cases or gold.
        warm=read(ROOT/'data/pilot/arc.json')[0]['public']
        warmup_results=[adapter.predict(warm) for _ in range(3)]
        save(out/'warmups'/(run_id+'.json'),{'phase':'warmup','train_only':True,'public':warm,'responses':warmup_results,'count':3,'run_id':run_id})
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        completed=0
        for row in rows:
            validate_public(row['public']);request_hash=digest(row['public'])
            dest=out/'items'/(digest(row['item_id'])+'.json')
            if dest.exists():
                prev=read(dest)
                if prev['request_hash']!=request_hash or prev['protocol_hash']!=protocol_hash: raise RuntimeError('Cannot resume mismatched existing output')
                completed+=1;continue
            torch.cuda.synchronize();t=time.perf_counter()
            try:
                result=adapter.predict(row['public']);torch.cuda.synchronize();latency=time.perf_counter()-t
                p=result['probabilities'];k=len(row['public']['questions']['answer']['criteria'])
                if len(p)!=k or any(not math.isfinite(v) or not 0<=v<=1 for v in p) or abs(sum(p)-1)>1e-5:
                    raise ValueError('Invalid model probabilities; batch stopped without score substitution')
            except Exception as e:
                event('inference_error',run_id=run_id,item_id=row['item_id'],error=repr(e));raise
            save(dest,{**row,'model_name':a.model,'run_id':run_id,'phase':phase,'time':now(),
                       'request_hash':request_hash,'protocol_hash':protocol_hash,'response':result,'latency_seconds':latency})
            completed+=1
            save(out/'checkpoint.json',{'run_id':run_id,'completed':completed,'total':len(rows),'last_item_id':row['item_id'],'status':'running','time':now()})
            if completed%20==0: print(a.model,phase,completed,'/',len(rows),flush=True)
        save(out/'checkpoint.json',{'run_id':run_id,'completed':completed,'total':len(rows),'status':'complete','time':now(),
                  'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        event('inference_complete',model=a.model,phase=phase,n=completed,run_id=run_id)
        print('complete',a.model,phase,completed,flush=True)

if __name__=='__main__': main()
