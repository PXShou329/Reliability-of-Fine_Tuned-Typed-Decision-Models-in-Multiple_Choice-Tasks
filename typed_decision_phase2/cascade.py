"""Real conditional local calls; no gold/cache or evaluator imports."""
import argparse,torch,time,traceback
import numpy as np
from utils import *
from inference import load,timed
from scipy.special import logsumexp
def main():
    a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');args=a.parse_args();policy=read(ROOT/'configs/cascade_policy.json')
    if not args.pilot:
        from protocol_guard import verify_lock
        verify_lock(policy['small']);verify_lock(policy['expert'])
    save(ROOT/'gpu_active.json',{'status':'RUNNING','pid':os.getpid(),'model':'live_cascade','stage':'pilot' if args.pilot else 'new','command':sys.argv,'time':now()})
    small=load(policy['small']);expert=load(policy['expert']);torch.cuda.synchronize()
    for r in read(ROOT/'data_public/P.json')[:6]:timed(small,r['public']);timed(expert,r['public'])
    save(ROOT/'audit/co_residency.json',{'time':now(),'status':'COMPLETED','small':small.metadata,'expert':expert.metadata,'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()})
    torch.cuda.reset_peak_memory_stats();public_p=read(ROOT/'data_public/P.json')
    rows=[r for ds in ['arc','csqa','obqa'] for r in [q for q in public_p if q['dataset']==ds][:4]] if args.pilot else read(ROOT/'data_public/E_new.json')
    block_n=3 if args.pilot else 1;conditions=['always_small','always_expert','confidence','random'];start=time.perf_counter();done=0
    for block in range(block_n):
        for r in rows:
            offset=(int(digest(r['item_id']),16)+block)%4;order=conditions[offset:]+conditions[:offset]
            for condition in order:
                assert budget_ok(), 'experiment stop reached; preserve incomplete matrix'
                dest=ROOT/'raw/cascade'/('pilot' if args.pilot else 'E_new')/(digest([r['item_id'],condition,block])+'.json')
                if dest.exists():continue
                record={'time':now(),'condition':condition,'item_id':r['item_id'],'dataset':r['dataset'],'block':block,'execution_order':order,
                    'small_call':False,'expert_call':False,'small_ms':0.,'expert_ms':0.,'policy_sha256':file_hash(ROOT/'configs/cascade_policy.json'),'public':r['public']}
                torch.cuda.synchronize();begin=time.perf_counter()
                try:
                    small_out=None;expert_out=None
                    if condition=='always_small' or (condition=='confidence' and policy['policy']!='bypass'):
                        small_out,record['small_ms']=timed(small,r['public']);record['small_call']=True
                    if condition=='random':
                        accept=int(digest(r['item_id']),16)/2**256<policy['random_small_coverage']
                        if accept:small_out,record['small_ms']=timed(small,r['public']);record['small_call']=True
                        else:expert_out,record['expert_ms']=timed(expert,r['public']);record['expert_call']=True
                    elif condition=='always_expert' or (condition=='confidence' and policy['policy']=='bypass'):
                        expert_out,record['expert_ms']=timed(expert,r['public']);record['expert_call']=True
                    elif condition=='confidence':
                        z=np.asarray(small_out['logits'])/policy['small_T'];p=np.exp(z-logsumexp(z));record['small_confidence']=float(p.max())
                        if p.max()<policy['tau']:expert_out,record['expert_ms']=timed(expert,r['public']);record['expert_call']=True
                    out=expert_out if record['expert_call'] else small_out
                    assert out is not None;record.update(status='COMPLETED',actual_small=small_out,actual_expert=expert_out,prediction_index=int(np.argmax(out['logits'])))
                except Exception as e:record.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
                torch.cuda.synchronize();record['end_to_end_ms']=(time.perf_counter()-begin)*1000;save(dest,record);done+=1
            if done%100==0:print('cascade',done,flush=True)
    folder=ROOT/'raw/cascade'/('pilot' if args.pilot else 'E_new')
    records=[read(p) for p in folder.glob('*.json')];failed=sum(r['status']!='COMPLETED' for r in records)
    save(ROOT/'results'/('cascade_pilot.json' if args.pilot else 'cascade_execution.json'),{'time':now(),'status':'COMPLETED' if failed==0 and len(records)==len(rows)*4*block_n else 'FAILED','planned_records':len(rows)*4*block_n,'new_records':done,'saved_records':len(records),'failed_records':failed,
         'wall_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'live':True})
    save(ROOT/'gpu_active.json',{'status':'NOT_STARTED','last_model':'live_cascade','ended':now()})
    print('cascade live finished',done,flush=True)
if __name__=='__main__':main()
