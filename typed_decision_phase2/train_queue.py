"""Minimal sequential GPU queue. Preserves failures and advances independent runs."""
import argparse,subprocess,os,time
from utils import *
def main():
    a=argparse.ArgumentParser();a.add_argument('--group',choices=['core','D2','D3','D4'],default='core');args=a.parse_args()
    conf=read(ROOT/'configs/training.json');runs=[r for r in conf['runs'] if (r.startswith(('B1_','B3_')) if args.group=='core' else r.startswith(args.group+'_'))]
    queue_path=ROOT/'manifests'/('queue_'+args.group+'.json')
    save(queue_path,{'time':now(),'pid':os.getpid(),'command':sys.argv,'runs':runs,'status':'RUNNING'})
    for run in runs:
        if (ROOT/'checkpoints'/run/'complete.json').exists():continue
        if not budget_ok(600):set_run(run,'SKIPPED_BUDGET',reason='report reserve protected');continue
        log=ROOT/'logs'/(run+'.log')
        with log.open('ab') as f:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'train.py'),'--run',run],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            save(queue_path,{'time':now(),'pid':os.getpid(),'child_pid':p.pid,'command':[sys.executable,'train.py','--run',run],'current':run,'status':'RUNNING'})
            code=p.wait()
        event('training_process_exit',run_id=run,returncode=code,log=str(log.relative_to(ROOT)))
        print(run,'exit',code,flush=True)
        # Infrastructure failure is preserved; this queue never changes a recipe or retries answers.
    save(queue_path,{'time':now(),'pid':os.getpid(),'runs':runs,'status':'COMPLETED','note':'Per-run states distinguish success/failure/skipped'})
if __name__=='__main__':main()
