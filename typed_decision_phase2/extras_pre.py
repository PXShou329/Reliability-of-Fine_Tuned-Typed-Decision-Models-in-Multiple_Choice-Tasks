"""D branches start in fixed resource-only order, one GPU child; no score reading."""
import subprocess,traceback
from utils import *
def run(script,args,tag):
    cmd=[sys.executable,'-u',str(ROOT/script),*args]
    with (ROOT/'logs'/(tag+'.log')).open('ab') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        save(ROOT/'manifests/extras_queue.json',{'time':now(),'status':'RUNNING','pid':os.getpid(),'child_pid':p.pid,'tag':tag,'command':cmd});code=p.wait()
    event('D_job_exit',tag=tag,command=cmd,returncode=code);print(tag,code,flush=True);return code
def main():
    assert read(ROOT/'manifests/pre_core_queue.json')['status']=='COMPLETED'
    # Availability deadlines are resolved before launch; downloads retain exact pinned files.
    if (ROOT/'models/generative/download_manifest.json').exists():
        set_run('D1_generative','RUNNING');code=run('adapter_gate.py',['--model','generative'],'D1_pilot')
        if code==0:
            code=run('inference.py',['--model','generative','--stage','pre'],'D1_pre')
            if code!=0:set_run('D1_generative','FAILED','logs/D1_pre.log','actual pre-evaluation process failed; preserve trace')
        else:set_run('D1_generative','FAILED','audit/adapter_generative.json','actual pilot failed; retain traceback')
    else:set_run('D1_generative','BLOCKED','manifests/download_status.json','pinned download not completed at D1 availability gate')
    for group,module in [('D2','D2_data_curve'),('D3','D3_smoothing'),('D4','D4_lodo_seeds')]:
        if not budget_ok(14400):
            set_run(module,'SKIPPED_BUDGET',reason='preserve >=3h C2 and90min report; resource-only gate')
            for name in read(ROOT/'configs/training.json')['runs']:
                if name.startswith(group+'_'):set_run(name,'SKIPPED_BUDGET',reason='resource-only module gate')
            continue
        set_run(module,'RUNNING');run('train_queue.py',['--group',group],group+'_training')
        models=[name for name in read(ROOT/'configs/training.json')['runs'] if name.startswith(group+'_') and (ROOT/'checkpoints'/name/'complete.json').exists()]
        if models:run('inference_queue.py',['--stage','pre','--models',*models],group+'_pre')
    if (ROOT/'models/laya_multilingual/download_manifest.json').exists():
        set_run('D5_laya_variant','RUNNING');code=run('adapter_gate.py',['--model','multilingual'],'D5_pilot')
        if code==0:
            code=run('inference.py',['--model','multilingual','--stage','pre'],'D5_pre')
            if code!=0:set_run('D5_laya_variant','FAILED','logs/D5_pre.log','actual pre-evaluation process failed')
        else:set_run('D5_laya_variant','FAILED','audit/adapter_multilingual.json','actual pilot failed')
    else:set_run('D5_laya_variant','BLOCKED','manifests/download_status.json','pinned multilingual checkpoint not complete')
    save(ROOT/'manifests/extras_queue.json',{'time':now(),'status':'COMPLETED','pid':os.getpid(),'order':'D1,D2,D3,D4,D5','branch_success_separate':True})
if __name__=='__main__':main()
