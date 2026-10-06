"""Single-GPU adaptation of the pinned official Laya GRPO+CE notebook.

No validation data, evaluation gold, metrics module, or formal predictions are loaded.
"""
import argparse,os,math,time,random,shutil
import numpy as np
import torch
from safetensors.torch import save_file
from laya.common import collate_items,proper_reward,TEMP_MIN,TEMP_MAX
from adapters import LayaAdapter
from utils import ROOT,read,save,event,now,digest

def model_batch(items,pad):
    b=collate_items([items],pad)
    tensors={k:v.to('cuda') for k,v in b.items() if k in ['input_ids','attention_mask','marker_pos','marker_mask','qtype']}
    return tensors,b['target'].to('cuda'),b['marker_mask'].to('cuda')

def make_items(adapter,purpose):
    out=[]
    for d in ['arc','csqa','obqa']:
        for r in read(ROOT/'data'/purpose/(d+'.json')):
            it=adapter.encode(r['public']);k=len(it['markers']);target=[0.]*k;target[r['gold_index']]=1.
            out.append({**it,'target':target,'label':r['gold_index'],'item_id':r['item_id'],'dataset':d})
    return out

def export(model,tok,cfg,path):
    path.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(path/'model.safetensors.tmp'))
    os.replace(path/'model.safetensors.tmp',path/'model.safetensors')
    model.encoder.config.save_pretrained(path/'encoder');tok.save_pretrained(path/'tokenizer');save(path/'rl_agent_config.json',cfg)

def fit_temperature(preds):
    kmax=max(len(r['logits']) for r in preds)
    z=torch.full((len(preds),kmax),-1e4);targets=torch.zeros((len(preds),kmax))
    for i,r in enumerate(preds):
        z[i,:len(r['logits'])]=torch.tensor(r['logits']);targets[i,r['gold_index']]=1
    log_t=torch.zeros(1,requires_grad=True);opt=torch.optim.LBFGS([log_t],lr=.1,max_iter=100)
    def closure():
        opt.zero_grad();loss=-(targets*torch.log_softmax(z/log_t.exp(),-1)).sum(-1).mean();loss.backward();return loss
    opt.step(closure)
    fitted=float(torch.clamp(log_t.exp(),TEMP_MIN,TEMP_MAX).item())
    if not math.isfinite(fitted): raise RuntimeError('Temperature fit failed; no silent fallback')
    return fitted

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    if not a.pilot and not (ROOT/'protocol_freeze.json').exists(): raise RuntimeError('Protocol must be frozen')
    conf=read(ROOT/'configs/protocol.json');tc=conf['training'];seed=tc['seed']
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)
    adapter=LayaAdapter(ROOT/'models/laya',conf);model=adapter.model;tok=adapter.tok;cfg=adapter.cfg.copy()
    items=make_items(adapter,'pilot' if a.pilot else 'training')
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.head_checkpointing=True;model.train();cfg['gradient_checkpointing']=True
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if 'encoder.' in n],'lr':tc['lr_encoder']},
        {'params':[p for n,p in model.named_parameters() if 'encoder.' not in n],'lr':tc['lr_head']}],weight_decay=tc['weight_decay'])
    micro=tc['micro_batch'];accum=tc['gradient_accumulation'];epochs=1 if a.pilot else tc['epochs']
    if a.pilot: items=items[:micro*2];accum=2
    updates_per_epoch=math.ceil(math.ceil(len(items)/micro)/accum)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=updates_per_epoch*epochs,eta_min=1e-6)
    rolling=ROOT/'checkpoints/training_resume.pt';start_epoch=0;start_chunk=0;updates=0
    if not a.pilot and rolling.exists():
        state=torch.load(rolling,map_location='cuda',weights_only=False)
        if state['config_hash']!=digest(conf): raise RuntimeError('Resume config mismatch')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start_epoch=state['next_epoch'];start_chunk=state['next_chunk'];updates=state['updates']
        torch.set_rng_state(state['rng_cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['rng_cuda']])
        del state
    t0=time.perf_counter();log_path=ROOT/('training_pilot.jsonl' if a.pilot else 'training_log.jsonl')
    for epoch in range(start_epoch,epochs):
        order=list(range(len(items)));random.Random(42+epoch).shuffle(order)
        sigma=tc['sigma_start']+(tc['sigma_end']-tc['sigma_start'])*epoch/max(1,epochs-1)
        chunks=[order[i:i+micro] for i in range(0,len(order),micro)];optimizer.zero_grad(set_to_none=True)
        for ci in range(start_chunk if epoch==start_epoch else 0,len(chunks)):
            tensors,target,mask=model_batch([items[i] for i in chunks[ci]],tok.pad_token_id)
            with torch.autocast('cuda',dtype=torch.bfloat16): logits,act=model(**tensors)
            logits=logits.float();k=mask.sum(-1,keepdim=True).float()
            eps=torch.randn((tc['group_size'],)+logits.shape,device='cuda')*sigma*mask
            eps=(eps-eps.sum(-1,keepdim=True)/k)*mask
            z=logits.detach().unsqueeze(0)+eps;q=z.masked_fill(~mask,-1e4).softmax(-1)
            with torch.no_grad():
                r=proper_reward(q,target.unsqueeze(0),tensors['qtype'],mask,w_sph=.75,w_rps=1.)
                adv=r-r.mean(0,keepdim=True);adv=adv/(adv.std()+1e-6)
            logp=-(((z-logits.unsqueeze(0))**2)*mask).sum(-1)/(2*sigma**2)
            loss_rl=-(adv*logp).mean();loss_ce=-(target*logits.masked_fill(~mask,-1e4).log_softmax(-1)).sum(-1).mean()
            loss=(loss_rl+loss_ce)/accum+0.*act.sum()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward()
            row={'time':now(),'epoch':epoch+1,'microbatch':ci+1,'n':len(chunks[ci]),'loss':float(loss.detach())*accum,
                 'loss_rl':float(loss_rl.detach()),'loss_ce':float(loss_ce.detach()),'reward':float(r.mean()),'sigma':sigma,
                 'seconds_since_training_start':time.perf_counter()-t0,'train_ids':[items[i]['item_id'] for i in chunks[ci]]}
            with log_path.open('a',encoding='utf-8') as f:
                import json;f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            if (ci+1)%accum==0 or ci+1==len(chunks):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);updates+=1
                next_epoch,next_chunk=(epoch+1,0) if ci+1==len(chunks) else (epoch,ci+1)
                if not a.pilot and (updates%tc['checkpoint_every_updates']==0 or next_chunk==0):
                    rolling.parent.mkdir(parents=True,exist_ok=True)
                    state={'config_hash':digest(conf),'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                           'next_epoch':next_epoch,'next_chunk':next_chunk,'updates':updates,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
                    torch.save(state,rolling.with_suffix('.tmp'));os.replace(rolling.with_suffix('.tmp'),rolling)
                    save(ROOT/'checkpoints/training_status.json',{'time':now(),'epoch':epoch+1,'next_epoch':next_epoch,'next_chunk':next_chunk,'updates':updates,'total_updates':updates_per_epoch*epochs,'status':'training'})
                if updates%10==0: print('epoch',epoch+1,'update',updates,'/',updates_per_epoch*epochs,'loss',row['loss'],flush=True)
        start_chunk=0
        if not a.pilot:
            export(model,tok,cfg,ROOT/'checkpoints'/('epoch_'+str(epoch+1)))
            event('training_epoch_complete',epoch=epoch+1,updates=updates)
    torch.cuda.synchronize()
    if a.pilot:
        save(ROOT/'training_pilot_summary.json',{'passed':True,'n':len(items),'updates':updates,'seconds':time.perf_counter()-t0,
                       'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'configuration':tc,
                       'pilot_overrides':{'epochs':epochs,'micro_batch':micro,'gradient_accumulation':accum,'effective_batch_size':micro*accum}});return
    training_seconds=time.perf_counter()-t0
    del optimizer,scheduler;model.eval();torch.cuda.empty_cache()
    calibration=[]
    with torch.inference_mode():
        for it in make_items(adapter,'calibration'):
            tensors,_,_=model_batch([it],tok.pad_token_id)
            with torch.autocast('cuda',dtype=torch.bfloat16): z,_=model(**tensors)
            calibration.append({'item_id':it['item_id'],'dataset':it['dataset'],'gold_index':it['label'],'logits':z[0,:len(it['markers'])].float().cpu().tolist()})
    fitted=fit_temperature(calibration)
    save(ROOT/'training_calibration_outputs.json',{'train_only':True,'n':len(calibration),'temperature':fitted,'rows':calibration})
    cfg['temperature']=[fitted,cfg.get('temperature',[1,1,1])[1],cfg.get('temperature',[1,1,1])[2]];cfg.pop('temperature_by_options',None)
    cfg['fine_tuned']=True;cfg['model_name']='laya-mcqa-local-20261006';cfg['training']=tc
    export(model,tok,cfg,ROOT/'checkpoints/laya_final')
    save(ROOT/'checkpoints/training_status.json',{'status':'complete','time':now(),'epochs':epochs,'updates':updates,
                'gradient_training_n':len(items),'calibration_n':len(calibration),'calibration_temperature':fitted,'training_seconds':training_seconds,
                'final_checkpoint_rule':'fixed final epoch; no validation-based selection'})
    event('training_complete',epochs=epochs,updates=updates,temperature=fitted)
    print('training complete',training_seconds,'seconds; temperature',fitted,flush=True)

if __name__=='__main__': main()
