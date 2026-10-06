"""Non-evaluation timing subset, 3 rotated model blocks, no competing GPU jobs."""
import argparse,time,torch,psutil,subprocess
from utils import *
from inference import load,timed
def main():
    a=argparse.ArgumentParser();a.add_argument('--models',nargs='+',required=True);args=a.parse_args();allrows=read(ROOT/'data_public/P.json')
    rows=[r for d in ['arc','csqa','obqa'] for r in [x for x in allrows if x['dataset']==d][:4]]
    results=[]
    for block in range(3):
        order=args.models[block:]+args.models[:block]
        for model in order:
            t=time.perf_counter();adapter=load(model);torch.cuda.synchronize();load_s=time.perf_counter()-t
            for r in rows[:6]:timed(adapter,r['public'])
            torch.cuda.reset_peak_memory_stats()
            for r in rows:
                out,ms=timed(adapter,r['public']);results.append({'block':block,'model':model,'order':canon(order),'item_id':r['item_id'],'dataset':r['dataset'],'latency_ms':ms,
                       'input_tokens':out['input_tokens'],'load_seconds':load_s,'time':now(),'own_RSS_bytes':psutil.Process().memory_info().rss})
            save(ROOT/'raw/timing'/(str(block)+'_'+model+'.json'),{'rows':results[-len(rows):],'metadata':adapter.metadata,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            del adapter;torch.cuda.empty_cache()
            print('timing',block,model,'done',flush=True)
    csv_write(ROOT/'results/timing_blocks.csv',results)
    save(ROOT/'provenance/timing_interference.json',{'time':now(),'single_heavy_GPU_job':True,'GPU_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks.current.sm','--format=csv'],text=True),
       'process_snapshot':[{'pid':p.pid,'name':p.info['name']} for p in psutil.process_iter(['name'])],'WDDM_process_memory':'N/A if driver does not expose','notes':'model order rotates across3 blocks; fixed12 train-only P records,6 warmups each load; disk/load excluded from per-call boundary'})
if __name__=='__main__':main()
