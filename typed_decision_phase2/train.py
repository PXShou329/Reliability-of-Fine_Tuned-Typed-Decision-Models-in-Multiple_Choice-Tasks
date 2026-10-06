"""V2 fixed Laya recipes. Only G labels; never imports evaluator or reads E_* / gate labels."""
import argparse,math,time,random,os,threading,subprocess
import numpy as np
import torch,psutil
from safetensors.torch import save_file
from laya.common import collate_items,proper_reward
from adapters import LayaAdapter
from utils import *

def observe(stop,rows):
    process=psutil.Process()
    while not stop.is_set():
        try:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True,creationflags=subprocess.CREATE_NO_WINDOW).strip()
            vals=text.split(',');rows.append({'time':now(),'device_memory_mib':int(vals[0]),'gpu_utilization':int(vals[1]),'gpu_temperature_C':int(vals[2]),'CPU_RSS_bytes':process.memory_info().rss})
        except Exception as e:rows.append({'time':now(),'observer_error':repr(e)})
        stop.wait(1)
def export(adapter,path):
    path.mkdir(parents=True,exist_ok=True);model=adapter.model
    tmp=path/'model.safetensors.tmp';save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(tmp));os.replace(tmp,path/'model.safetensors')
    model.encoder.config.save_pretrained(path/'encoder');adapter.tok.save_pretrained(path/'tokenizer')
    cfg=adapter.cfg.copy();cfg.update(fine_tuned=True,gradient_checkpointing=True);save(path/'rl_agent_config.json',cfg)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    config=read(ROOT/'configs/training.json');runs=config['runs'];rc=runs[args.run];tc=config['common'];out=ROOT/'checkpoints'/args.run
    if (out/'complete.json').exists():print('preserved completed',args.run,flush=True);return
    if not budget_ok(rc.get('estimated_seconds',600)):set_run(args.run,'SKIPPED_BUDGET',reason='report reserve protected');return
    set_run(args.run,'RUNNING');save(ROOT/'gpu_active.json',{'pid':os.getpid(),'run':args.run,'command':sys.argv,'time':now(),'status':'RUNNING'})
    observer=[];stop=threading.Event();thread=threading.Thread(target=observe,args=(stop,observer),daemon=True);thread.start();start=time.perf_counter()
    try:
        seed=rc['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)
        noise_rng=torch.Generator(device='cuda');noise_rng.manual_seed(seed+1000003)
        adapter=LayaAdapter(V1/'models/laya',tc);model=adapter.model;tok=adapter.tok
        all_public={r['item_id']:r for r in read(ROOT/'data_public/G.json')};labels=read(ROOT/'evaluator_private/G.json')
        ids=rc['G_ids'];items=[]
        for i in ids:
            r=all_public[i];it=adapter.encode(r['public']);k=len(it['markers']);target=[rc.get('smoothing',0)/k]*k
            target[labels[i]['gold_index']]+=1-rc.get('smoothing',0)
            items.append({**it,'target':target,'item_id':i})
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.head_checkpointing=True;model.train()
        opt=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if n.startswith('encoder.')],'lr':tc['lr_encoder']},
                              {'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.')],'lr':tc['lr_head']}],weight_decay=.01)
        micro=8;accum=8;epochs=4;chunks_n=math.ceil(len(items)/micro);total=math.ceil(chunks_n/accum)*epochs
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total,eta_min=1e-6);updates=0;start_epoch=0;resume=out/'resume.pt'
        if resume.exists():
            state=torch.load(resume,map_location='cuda',weights_only=False);assert state['recipe_hash']==digest(rc)
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start_epoch=state['next_epoch'];updates=state['updates']
            torch.set_rng_state(state['rng_cpu'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['rng_cuda']]);noise_rng.set_state(state['noise_rng'].cpu());del state
        out.mkdir(parents=True,exist_ok=True);attempt=now().replace(':','').replace('.','');save(out/('run_start_'+attempt+'.json'),{'time':now(),'run':args.run,'pid':os.getpid(),'config':rc,'common':tc,'G_hash':digest(ids),'start_weight_sha256':file_hash(V1/'models/laya/model.safetensors'),
            'noise_seed':seed+1000003,'dropout_seed':seed,'shuffle_rule':'random.Random(seed+100003*epoch)','total_updates':total,'precision':'FP32 params / BF16 autocast',
            'observer':'full load-through-completion; sampled total GPU and own CPU RSS'})
        training_start=time.perf_counter()
        for epoch in range(start_epoch,epochs):
            epoch_log=out/('epoch_'+str(epoch+1)+'_attempt_'+attempt+'.jsonl')
            order=list(range(len(items)));random.Random(seed+100003*epoch).shuffle(order);chunks=[order[i:i+micro] for i in range(0,len(order),micro)];opt.zero_grad(set_to_none=True)
            sigma=.4+(.1-.4)*epoch/3
            for ci,chunk in enumerate(chunks):
                batch=collate_items([[items[j] for j in chunk]],tok.pad_token_id)
                tensors={k:v.to('cuda') for k,v in batch.items() if k in ['input_ids','attention_mask','marker_pos','marker_mask','qtype']}
                target=batch['target'].to('cuda');mask=batch['marker_mask'].to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):logits,act=model(**tensors)
                logits=logits.float();ce=-(target*logits.masked_fill(~mask,-1e4).log_softmax(-1)).sum(-1).mean();rl=logits.new_zeros(())
                if rc['recipe']=='grpo_ce':
                    k=mask.sum(-1,keepdim=True).float();eps=torch.randn((4,)+logits.shape,device='cuda',generator=noise_rng)*sigma*mask
                    eps=(eps-eps.sum(-1,keepdim=True)/k)*mask;z=logits.detach().unsqueeze(0)+eps;q=z.masked_fill(~mask,-1e4).softmax(-1)
                    with torch.no_grad():reward=proper_reward(q,target.unsqueeze(0),tensors['qtype'],mask,w_sph=.75,w_rps=1.)
                    adv=reward-reward.mean(0,keepdim=True);adv=adv/(adv.std()+1e-6);logp=-(((z-logits.unsqueeze(0))**2)*mask).sum(-1)/(2*sigma**2);rl=-(adv*logp).mean()
                loss=(ce+rl)/accum+0.*act.sum();assert torch.isfinite(loss);loss.backward()
                row={'time':now(),'epoch':epoch+1,'microbatch':ci+1,'n':len(chunk),'loss_ce':float(ce.detach()),'loss_rl':float(rl.detach()),'ids':[items[j]['item_id'] for j in chunk]}
                with epoch_log.open('a',encoding='utf-8') as f:f.write(canon(row)+'\n');f.flush()
                if (ci+1)%accum==0 or ci+1==len(chunks):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();scheduler.step();opt.zero_grad(set_to_none=True);updates+=1
            state={'recipe_hash':digest(rc),'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'next_epoch':epoch+1,'updates':updates,
                'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'noise_rng':noise_rng.get_state()}
            torch.save(state,resume.with_suffix('.tmp'));os.replace(resume.with_suffix('.tmp'),resume)
            save(out/('epoch_'+str(epoch+1)+'_committed.json'),{'epoch':epoch+1,'log':epoch_log.name,'updates':updates,'log_sha256':file_hash(epoch_log)})
            save(out/'checkpoint.json',{'time':now(),'next_epoch':epoch+1,'updates':updates,'total_updates':total,'G_n':len(items),'status':'RUNNING'})
            print(args.run,'epoch',epoch+1,'updates',updates,'/',total,flush=True)
        torch.cuda.synchronize();training_seconds=time.perf_counter()-training_start;export(adapter,out/'final')
        result={'time':now(),'run':args.run,'status':'COMPLETED','seed':seed,'recipe':rc['recipe'],'smoothing':rc.get('smoothing',0),'G_n':len(items),
          'item_exposures':len(items)*4,'updates':updates,'epochs':4,'training_seconds':training_seconds,'total_wall_seconds':time.perf_counter()-start,
          'final_sha256':file_hash(out/'final/model.safetensors'),'config_hash':digest(rc),'code_sha256':file_hash(ROOT/'train.py'),
          'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'reused':False}
        save(out/'complete.json',result);set_run(args.run,'COMPLETED',str((out/'complete.json').relative_to(ROOT)));print(result,flush=True)
    except Exception as e:
        import traceback;save(out/'failure.json',{'time':now(),'run':args.run,'error':repr(e),'traceback':traceback.format_exc(),'config':rc,'pid':os.getpid()});set_run(args.run,'FAILED',str((out/'failure.json').relative_to(ROOT)),repr(e));raise
    finally:
        stop.set();thread.join(timeout=5);save(out/'resource_log.json',{'scope':'full process from load to completion','samples':observer})
        save(ROOT/'gpu_active.json',{'status':'NOT_STARTED','last_run':args.run,'ended':now()})
if __name__=='__main__':main()
