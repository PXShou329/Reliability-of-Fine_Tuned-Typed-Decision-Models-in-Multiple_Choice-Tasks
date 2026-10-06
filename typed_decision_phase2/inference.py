"""Public-only inference producer; no labels/evaluator imports. Atomic outputs resume by exact request hashes."""
import argparse,time,traceback,torch,psutil,subprocess
from utils import *
from adapters import LayaAdapter,ClefAdapter
from schema import validate_public
def rotate(p,r):
    vals=list(p['questions']['answer']['criteria'].values());k=len(vals);mapping=[(j+r)%k for j in range(k)]
    return {'state':p['state'],'questions':{'answer':{'type':'choice','instructions':p['questions']['answer']['instructions'],
              'criteria':{chr(65+j):vals[mapping[j]] for j in range(k)}}}},mapping
def load(name):
    if name=='base':return LayaAdapter(V1/'models/laya')
    if name=='v1_ft_history':
        adapter=LayaAdapter(V1/'checkpoints/laya_final');adapter.metadata['reused_background']=True;adapter.metadata['controlled_B1_recipe']=False
        return adapter
    if name=='clef':return ClefAdapter(V1/'models/clef_flash')
    if name=='kev':
        from kev_adapter import KevAdapter
        return KevAdapter()
    if name=='generative':
        from generative_adapter import GenerativeAdapter
        return GenerativeAdapter()
    if name=='multilingual':return LayaAdapter(ROOT/'models/laya_multilingual')
    assert (ROOT/'checkpoints'/name/'complete.json').exists(),name
    return LayaAdapter(ROOT/'checkpoints'/name/'final')
def timed(adapter,p):
    torch.cuda.synchronize();start=time.perf_counter();v=adapter.predict(p);torch.cuda.synchronize()
    return v,(time.perf_counter()-start)*1000
def requests(model,stage):
    rc=read(ROOT/'configs/training.json')['runs'].get(model,{})
    phases=['C_T','E_old'] if stage=='pre' else ['E_new']
    if stage=='pre' and model in ['B1_grpo_ce_20261006','clef']:phases.insert(1,'C_gate')
    for purpose in phases:
        rows=read(ROOT/'data_public'/(purpose+'.json'))
        if purpose=='C_T' and model=='generative':continue
        if purpose=='C_T' and 'C_T_ids' in rc:rows=[r for r in rows if r['item_id'] in rc['C_T_ids']]
        if purpose.startswith('E_') and rc.get('target') in ['arc','csqa','obqa']:rows=[r for r in rows if r['dataset']==rc['target']]
        for r in rows:yield purpose,r,0,r['public'],list(range(len(r['public']['questions']['answer']['criteria'])))
    if model in ['base','B1_grpo_ce_20261006','B1_ce_20261006','kev','clef']:
        purpose='E_old' if stage=='pre' else 'E_new';ids=read(ROOT/'manifests/permutation_manifest.json')['old' if stage=='pre' else 'new']
        for r in read(ROOT/'data_public'/(purpose+'.json')):
            if r['item_id'] not in ids:continue
            k=len(r['public']['questions']['answer']['criteria'])
            for rot in range(k):
                p,m=rotate(r['public'],rot);yield 'B2_'+purpose,r,rot,p,m
def raw_path(model,purpose,item_id,rotation):return ROOT/'raw'/model/purpose/(digest([item_id,rotation])+'.json')
def main():
    a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--stage',choices=['pre','new'],required=True);args=a.parse_args()
    if args.stage=='new':
        from protocol_guard import verify_lock
        lock=verify_lock(args.model)
    assert read(ROOT/'audit/data_isolation.json')['status']=='COMPLETED'
    req=list(requests(args.model,args.stage));pending=[]
    for purpose,r,rot,p,m in req:
        dest=raw_path(args.model,purpose,r['item_id'],rot)
        if dest.exists():
            old=read(dest);assert old['public_hash']==digest(p) and old['model']==args.model
        else:pending.append((purpose,r,rot,p,m,dest))
    if not pending:
        summary=ROOT/'results'/('inference_'+args.model+'_'+args.stage+'.json')
        if not summary.exists():
            failures=sum(read(raw_path(args.model,purpose,r['item_id'],rot))['status']!='COMPLETED' for purpose,r,rot,p,m in req)
            save(summary,{'time':now(),'status':'COMPLETED' if failures==0 else 'FAILED','planned':len(req),'new_calls':0,'previously_saved':len(req),'failed':failures,
                'total_wall_seconds':None,'peak_allocated_bytes':None,'peak_reserved_bytes':None,'summary_recovered_from_saved_raw':True,'note':'request outputs preserved; original process timing unavailable'})
        print(args.model,args.stage,'already complete');return
    start=time.perf_counter();save(ROOT/'gpu_active.json',{'status':'RUNNING','pid':os.getpid(),'model':args.model,'stage':args.stage,'command':sys.argv,'time':now()})
    meta={};done=0;failed=0
    try:
        t=time.perf_counter();adapter=load(args.model);torch.cuda.synchronize();load_seconds=time.perf_counter()-t
        for r in read(ROOT/'data_public/P.json')[:6]:timed(adapter,r['public'])
        torch.cuda.reset_peak_memory_stats();meta={'time':now(),'model':args.model,'metadata':adapter.metadata,'load_seconds':load_seconds,
             'warmup_n':6,'timing_boundary':'tokenization + transfer + forward + decoding + CUDA synchronization; excludes disk and initial load',
             'versions':{'torch':torch.__version__},'code_sha256':file_hash(ROOT/'inference.py')}
        canonical=ROOT/'provenance'/('model_'+args.model+'.json')
        if canonical.exists():assert read(canonical)['metadata']==meta['metadata'],'loaded model metadata changed'
        else:save(canonical,meta)
        save(ROOT/'provenance'/('model_'+args.model+'_'+args.stage+'.json'),meta)
        for purpose,r,rot,p,m,dest in pending:
            assert budget_ok(), 'report reserve reached';validate_public(p)
            row={'time':now(),'model':args.model,'purpose':purpose,'item_id':r['item_id'],'dataset':r['dataset'],'rotation':rot,'mapping_to_original':m,
                  'public':p,'public_hash':digest(p),'pid':os.getpid(),'status':'RUNNING','model_metadata_sha256':file_hash(canonical),'inference_source_sha256':meta['code_sha256']}
            try:
                out,ms=timed(adapter,p)
                if 'logits' in out:
                    z=out['logits'];assert len(z)==len(m) and all(__import__('math').isfinite(x) for x in z)
                    assert abs(sum(out['probabilities'])-1)<1e-5
                row.update(status='COMPLETED',actual_output=out,latency_ms=ms);done+=1
            except Exception as e:row.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());failed+=1
            save(dest,row)
            if (done+failed)%100==0:print(args.model,purpose,done,'done',failed,'failed',flush=True)
        all_failed=sum(read(raw_path(args.model,purpose,r['item_id'],rot))['status']!='COMPLETED' for purpose,r,rot,p,m in req)
        save(ROOT/'results'/('inference_'+args.model+'_'+args.stage+'.json'),{'time':now(),'status':'COMPLETED' if all_failed==0 else 'FAILED','planned':len(req),
            'new_calls':done+failed,'previously_saved':len(req)-len(pending),'failed':all_failed,'total_wall_seconds':time.perf_counter()-start,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    except Exception as e:
        save(ROOT/'logs'/('inference_failure_'+args.model+'_'+args.stage+'_'+str(time.time_ns())+'.json'),{'command':sys.argv,'pid':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'done':done,'failed':failed});raise
    finally:save(ROOT/'gpu_active.json',{'status':'NOT_STARTED','last_model':args.model,'ended':now()})
if __name__=='__main__':main()
