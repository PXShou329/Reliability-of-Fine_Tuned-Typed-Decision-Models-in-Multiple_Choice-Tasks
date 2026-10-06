"""Non-blocking progress snapshot; can be called while one GPU runner is active."""
from utils import *
def main():
    current=now();stamp=current.replace(':','').replace('.','');runs=list(__import__('csv').DictReader((ROOT/'manifests/run_matrix.csv').open(encoding='utf-8-sig')))
    calls=sum(1 for p in (ROOT/'raw').glob('*/*/*.json') if read(p).get('status') in ['COMPLETED','FAILED'])
    completed=[r for r in runs if r['status']=='COMPLETED'];deadline=read(ROOT/'deadline.json')
    state={'time':current,'elapsed_hours_from_original_T0':(datetime.fromisoformat(current)-datetime.fromisoformat(deadline['original_t0'])).total_seconds()/3600,
       'remaining_experiment_hours':(datetime.fromisoformat(deadline['experiment_stop_utc'])-datetime.fromisoformat(current)).total_seconds()/3600,
       'runs_completed':len(completed),'actual_saved_model_requests':calls,'completed_run_ids':[r['run_id'] for r in completed],
       'gpu_active':read(ROOT/'gpu_active.json') if (ROOT/'gpu_active.json').exists() else None,
       'download':read(ROOT/'manifests/download_status.json') if (ROOT/'manifests/download_status.json').exists() else None,
       'E_new_state':read(ROOT/'status.json')['E_new'],'decisions':'decisions.md','blockers':'blockers.md'}
    save(ROOT/'logs/progress'/('progress_'+stamp+'.json'),state);print({k:v for k,v in state.items() if k not in ['completed_run_ids','gpu_active','download']},flush=True)
if __name__=='__main__':main()
