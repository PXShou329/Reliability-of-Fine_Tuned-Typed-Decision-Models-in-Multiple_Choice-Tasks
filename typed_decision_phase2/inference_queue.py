"""Minimal sequential inference queue; one GPU child and resumable raw files."""
import argparse,subprocess
from utils import *
def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['pre','new'],required=True);a.add_argument('--models',nargs='+');args=a.parse_args()
    models=args.models or read(ROOT/'manifests/protocol.lock.json')['models'];job='inference_'+args.stage;done=[]
    for model in models:
        if not budget_ok(900):event('inference_skipped_budget',model=model,stage=args.stage);continue
        command=[sys.executable,'-u',str(ROOT/'inference.py'),'--model',model,'--stage',args.stage]
        log=ROOT/'logs'/(job+'_'+model+'.log')
        with log.open('ab') as f:
            p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            save(ROOT/'manifests'/(job+'_queue.json'),{'status':'RUNNING','time':now(),'pid':os.getpid(),'child_pid':p.pid,'model':model,'command':command,'completed':done})
            code=p.wait()
        event('inference_process_exit',model=model,stage=args.stage,returncode=code,log=str(log.relative_to(ROOT)));done.append({'model':model,'returncode':code})
        print(job,model,code,flush=True)
    save(ROOT/'manifests'/(job+'_queue.json'),{'time':now(),'status':'COMPLETED','pid':os.getpid(),'results':done,'per_condition_success_separate':True})
if __name__=='__main__':main()
