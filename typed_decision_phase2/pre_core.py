"""Core gates and historical/calibration inference, while optional downloads complete."""
import subprocess
from utils import *
def run(script,args):
    command=[sys.executable,'-u',str(ROOT/script),*args];name=script.replace('.py','')+'_'+digest(args)[:16]
    with (ROOT/'logs'/(name+'.log')).open('ab') as f:
        p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        save(ROOT/'manifests/pre_core_queue.json',{'time':now(),'status':'RUNNING','pid':os.getpid(),'child_pid':p.pid,'command':command});code=p.wait()
    event('pre_core_job_exit',command=command,returncode=code);print(script,args,code,flush=True);return code
def main():
    assert read(ROOT/'manifests/queue_core.json')['status']=='COMPLETED'
    assert read(ROOT/'audit/kev_pilot.json')['status']=='COMPLETED'
    for model in ['base','v1_ft_history','B1_grpo_ce_20261006','B1_ce_20261006','clef']:
        gate=ROOT/'audit'/('adapter_'+model+'.json')
        if gate.exists() and read(gate)['status']=='COMPLETED':continue
        assert run('adapter_gate.py',['--model',model])==0,('shared adapter gate failed',model)
    models=['base','v1_ft_history']+[r for r in read(ROOT/'configs/training.json')['runs'] if r.startswith(('B1_','B3_'))]+['kev','clef']
    run('inference_queue.py',['--stage','pre','--models',*models])
    save(ROOT/'manifests/pre_core_queue.json',{'time':now(),'status':'COMPLETED','pid':os.getpid(),'models':models,'per_condition_results_separate':True})
if __name__=='__main__':main()
