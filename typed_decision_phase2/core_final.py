"""Sequential dependency chain; each child has its own resumable artifacts."""
import subprocess,traceback,argparse
from utils import *
resume_at=None

def step(name,script,args=()):
    global resume_at
    if resume_at:
        if name!=resume_at:print('preserve prior dependency',name,flush=True);return
        resume_at=None
    command=[sys.executable,'-u',str(ROOT/script),*args]
    with (ROOT/'logs'/('final_'+name+'.log')).open('ab') as f:
        p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        save(ROOT/'manifests/core_final_queue.json',{'time':now(),'status':'RUNNING','pid':os.getpid(),'child_pid':p.pid,'step':name,'command':command})
        result=p.wait()
    event('core_final_step_exit',step=name,returncode=result,command=command)
    print(name,'exit',result,flush=True)
    assert result==0,('dependency gate failed',name,result)

def main():
    global resume_at
    parser=argparse.ArgumentParser();parser.add_argument('--from-step',choices=['extras','training_audit','calibration','cascade_pilot','timing','freeze','formal_mcqa','formal_cascade','mcqa_analysis','cascade_analysis','independent_recompute','progress']);args=parser.parse_args();resume_at=args.from_step
    assert read(ROOT/'manifests/pre_core_queue.json')['status']=='COMPLETED'
    step('extras','extras_pre.py')
    step('training_audit','train_audit.py')
    step('calibration','calibrate.py',['--policy'])
    set_run('C1_live_cascade','RUNNING',reason='actual co-resident P pilot, formal routing follows frozen policy')
    step('cascade_pilot','cascade.py',['--pilot'])
    models=['base','B1_grpo_ce_20261006','B1_ce_20261006','kev','clef']
    for name in ['generative','multilingual']:
        p=ROOT/'results'/('inference_'+name+'_pre.json')
        if p.exists() and read(p)['status']=='COMPLETED':models.append(name)
    step('timing','timing.py',['--models',*models])
    step('freeze','freeze_protocol.py')
    set_run('B2_order','RUNNING',reason='fixed cyclic requests, E_new methods frozen')
    step('formal_mcqa','inference_queue.py',['--stage','new'])
    step('formal_cascade','cascade.py')
    step('mcqa_analysis','analyze_mcqa.py')
    step('cascade_analysis','analyze_cascade.py')
    step('independent_recompute','audit_v2_results.py')
    step('progress','progress.py')
    save(ROOT/'manifests/core_final_queue.json',{'time':now(),'status':'COMPLETED','pid':os.getpid(),'step':'all main formal outputs and analysis saved'})

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(ROOT/'logs'/('core_final_failure_'+str(time.time_ns())+'.json'),{'time':now(),'pid':os.getpid(),'error':repr(e),'traceback':traceback.format_exc()})
        raise
