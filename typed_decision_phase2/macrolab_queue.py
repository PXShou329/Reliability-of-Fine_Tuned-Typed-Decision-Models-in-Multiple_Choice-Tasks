"""Separate generated-task queue after the main study is deliverable."""
import subprocess,traceback
from utils import *
def step(tag,args):
    command=[sys.executable,'-u',str(ROOT/'macrolab/execute.py'),*args]
    with (ROOT/'logs'/('macro_'+tag+'.log')).open('ab') as f:
        p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        save(ROOT/'macrolab/queue.json',{'time':now(),'status':'RUNNING','pid':os.getpid(),'child_pid':p.pid,'stage':tag,'command':command})
        code=p.wait()
    event('macrolab_process_exit',stage=tag,command=command,returncode=code)
    print(tag,'exit',code,flush=True);return code
def main():
    assert read(ROOT/'results/analysis_complete.json')['status']=='COMPLETED'
    assert read(ROOT/'audit/v2_independent_recompute.json')['status']=='COMPLETED'
    assert read(ROOT/'results/cascade_execution.json')['status']=='COMPLETED'
    assert read(ROOT/'macrolab/safety_tests.json')['status']=='COMPLETED'
    if not budget_ok(10800):
        set_run('C2_macrolab','SKIPPED_BUDGET',reason='new generated-task module requires >=3h experiment budget');return
    models=['B1_grpo_ce_20261006']
    if (ROOT/'results/inference_clef_new.json').exists() and read(ROOT/'results/inference_clef_new.json')['status']=='COMPLETED':models.append('clef')
    save(ROOT/'macrolab/resource_gate.json',{'time':now(),'status':'COMPLETED','formal_models':models,'reason':'fixed optional expert chosen by actual runtime availability and >=3h budget only; no MCQA scores used',
      'remaining_experiment_hours':(datetime.fromisoformat(read(ROOT/'deadline.json')['experiment_stop_utc'])-datetime.now(timezone.utc)).total_seconds()/3600,'conditions':['M0','M1','M2','random','enumeration']})
    set_run('C2_macrolab','RUNNING','macrolab/resource_gate.json','actual typed development followed by independent frozen generated-task exploration')
    failures=[]
    if step('development',['--stage','development'])!=0:
        set_run('C2_macrolab','BLOCKED','logs/macro_development.log','actual development/safety/task gate failed; main MCQA unaffected');return
    for model in models:
        if step('formal_'+model,['--stage','formal','--model',model])!=0:failures.append(model)
    if step('search',['--stage','search'])!=0:failures.append('search')
    command=[sys.executable,'-u',str(ROOT/'macrolab/evaluate.py')]
    with (ROOT/'logs/macro_evaluate.log').open('ab') as f:code=subprocess.call(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    assert code==0,'hidden evaluator failed; retain existing traces'
    result=read(ROOT/'macrolab/evaluation_complete.json')
    set_run('C2_macrolab','COMPLETED' if not failures and result['status']=='COMPLETED' else 'FAILED','results/macrolab_metrics.csv','missing/failed condition episodes retained in planned denominator' if failures else '')
    save(ROOT/'macrolab/queue.json',{'time':now(),'status':'COMPLETED' if not failures else 'FAILED','pid':os.getpid(),'failures':failures,'evaluation_status':result['status']})
    print('MacroLab queue done',result,flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(ROOT/'logs'/('macro_queue_failure_'+str(time.time_ns())+'.json'),{'time':now(),'pid':os.getpid(),'error':repr(e),'traceback':traceback.format_exc()});raise
