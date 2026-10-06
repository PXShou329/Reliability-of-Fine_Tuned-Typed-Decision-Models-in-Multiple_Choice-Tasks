import argparse,time,psutil
from run_inference import GPUSampler
from utils import ROOT,save,read,now

ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True);a=ap.parse_args()
original=psutil.Process(a.pid).create_time()
save(ROOT/'resource_logs/training_observer_scope.json',{'start':now(),'pid':a.pid,'training_status_at_attachment':read(ROOT/'checkpoints/training_status.json'),
                                                     'coverage':'observed training tail only, not full-run peak'})
with GPUSampler(ROOT/'resource_logs/laya_training_observed_tail.json'):
    while psutil.pid_exists(a.pid):
        try:
            if psutil.Process(a.pid).create_time()!=original: break
        except psutil.NoSuchProcess: break
        time.sleep(.5)
