"""Actual read-only diagnostic replays. No formal record or method is replaced."""
import csv
from utils import *
def main():
    save(ROOT/'gpu_active.json',{'time':now(),'status':'RUNNING','pid':os.getpid(),'stage':'diagnostic_replay','command':sys.argv})
    from protocol_guard import verify_lock
    from inference import load,timed,raw_path
    selection=read(ROOT/'configs/diagnostic_replay.lock.json');results=[]
    for model,paths in selection['MCQA'].items():
        verify_lock(model);adapter=load(model)
        for r in read(ROOT/'data_public/P.json')[:6]:timed(adapter,r['public'])
        for path in paths:
            original=read(ROOT/path);out,ms=timed(adapter,original['public']);z=out['logits'];expected=original['actual_output']['logits'];d=max(abs(a-b) for a,b in zip(z,expected))
            results.append({'model':model,'kind':'MCQA','original':path,'original_sha256':file_hash(ROOT/path),'new_output':out,'latency_ms':ms,'max_raw_logit_difference':d,'prediction_identical':max(range(len(z)),key=lambda j:z[j])==max(range(len(expected)),key=lambda j:expected[j])})
        del adapter
        import torch,gc
        gc.collect();torch.cuda.empty_cache()
    sys.path.insert(0,str(ROOT/'macrolab'));from model_adapter import MacroModel
    for model,path in selection['MacroLab'].items():
        original=read(ROOT/path);adapter=MacroModel(model)
        for trace in original['traces'][:3]:
            q=trace['request'];out=adapter.decide(q['state'],list(q['questions']['answer']['criteria'].values()));d=max(abs(a-b) for a,b in zip(out['raw_logits'],trace['raw_logits']))
            results.append({'model':model,'kind':'MacroLab','original':path,'call':trace['call'],'original_sha256':file_hash(ROOT/path),'new_output':out,'max_raw_logit_difference':d,'prediction_identical':out['selected_index']==trace['selected_index']})
        del adapter;gc.collect();torch.cuda.empty_cache()
    save(ROOT/'audit/diagnostic_replays.json',{'time':now(),'status':'COMPLETED' if all(r['prediction_identical'] for r in results) else 'FAILED','replay_n':len(results),'pid':os.getpid(),'code_sha256':file_hash(ROOT/'diagnostic_replay.py'),'selection':selection,'results':results,
        'formal_outputs_untouched':True,'no_method_change_or_score_selection':True,'exact_logits_all':all(r['max_raw_logit_difference']==0 for r in results)})
    print('actual diagnostic replay completed',len(results),all(r['max_raw_logit_difference']==0 for r in results),flush=True)
if __name__=='__main__':
    try:main()
    finally:save(ROOT/'gpu_active.json',{'time':now(),'status':'NOT_STARTED','last_stage':'diagnostic_replay'})
