"""Final availability/method lock before first E_new forward. No E_new labels or scores loaded."""
from utils import *
import csv
def main():
    dest=ROOT/'manifests/protocol.lock.json'
    if dest.exists():assert read(dest)['state']=='FROZEN';print('preserve frozen protocol');return
    assert read(ROOT/'audit/audit_v1.json')['status']=='COMPLETED'
    assert read(ROOT/'audit/data_isolation.json')['status']=='COMPLETED'
    assert read(ROOT/'audit/scoring_tests.json')['status']=='COMPLETED'
    config=read(ROOT/'configs/training.json');matrix=list(csv.DictReader((ROOT/'manifests/run_matrix.csv').open(encoding='utf-8-sig')));byid={r['run_id']:r for r in matrix}
    models=['base','v1_ft_history','clef'];exceptions=[]
    for model in ['kev','generative','multilingual']:
        path=ROOT/'results'/('inference_'+model+'_pre.json')
        if path.exists() and read(path)['status']=='COMPLETED':models.append(model)
        else:
            module={'kev':'A1_kev','generative':'D1_generative','multilingual':'D5_laya_variant'}[model]
            assert byid[module]['status'] in ['FAILED','BLOCKED','SKIPPED_BUDGET'],('availability unresolved',module)
            exceptions.append({'model':model,'status':byid[module]['status'],'reason':byid[module]['reason']})
    for model in config['runs']:
        if (ROOT/'checkpoints'/model/'complete.json').exists():models.append(model)
        else:
            assert byid[model]['status'] in ['FAILED','BLOCKED','SKIPPED_BUDGET'],('training unresolved',model)
            exceptions.append({'model':model,'status':byid[model]['status'],'reason':byid[model]['reason']})
    calibrators=read(ROOT/'calibration/calibrators.json');identities={}
    for model in models:
        assert read(ROOT/'results'/('inference_'+model+'_pre.json'))['status']=='COMPLETED',model
        if model!='generative':assert model in calibrators,model
        meta=read(ROOT/'provenance'/('model_'+model+'.json'));identity={'metadata_sha256':file_hash(ROOT/'provenance'/('model_'+model+'.json')),'metadata':meta['metadata']}
        if model in config['runs']:
            path=ROOT/'checkpoints'/model/'final/model.safetensors';assert file_hash(path)==read(ROOT/'checkpoints'/model/'complete.json')['final_sha256'];identity['weight_sha256']=file_hash(path)
        if model in config['runs']:directory=ROOT/'checkpoints'/model/'final'
        else:directory={'base':V1/'models/laya','v1_ft_history':V1/'checkpoints/laya_final','clef':V1/'models/clef_flash','kev':ROOT/'models/kev','generative':ROOT/'models/generative','multilingual':ROOT/'models/laya_multilingual'}[model]
        weight_paths=list(directory.glob('*.safetensors'))
        if model=='kev':weight_paths+=list((ROOT/'models/kev_base').glob('*.safetensors'))
        assert weight_paths,model
        identity['weight_files']={str(p):file_hash(p) for p in sorted(weight_paths)}
        directories=[directory]+([ROOT/'models/kev_base'] if model=='kev' else [])
        identity['model_files']={str(p):file_hash(p) for folder in directories for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in ['.json','.txt','.model','.jinja','.py','.tiktoken'] and '.cache' not in p.parts}
        identities[model]=identity
    assert read(ROOT/'results/cascade_pilot.json')['status']=='COMPLETED'
    files=['utils.py','adapters.py','schema.py','inference.py','protocol_guard.py','kev_adapter.py','generative_adapter.py','train.py','calibrate.py','scoring.py','bootstrap_stats.py','analyze_mcqa.py','audit_v2_results.py','cascade.py','analyze_cascade.py','timing.py']
    (ROOT/'manifests/run_matrix_at_freeze.csv').write_bytes((ROOT/'manifests/run_matrix.csv').read_bytes())
    artifacts=['configs/training.json','configs/calibration_views.json','calibration/calibrators.json','configs/cascade_policy.json','manifests/split_manifest.json',
       'manifests/permutation_manifest.json','manifests/lodo_manifest.json','manifests/data_budget_nested.json','manifests/run_matrix_at_freeze.csv','audit/data_isolation.json',
       'data_public/C_T.json','data_public/C_gate.json','data_public/P.json','data_public/E_old.json','provenance/environment_versions.json']
    dependency_paths=list((V1/'.venv/Lib/site-packages/laya').rglob('*.py'))+list((ROOT/'sources/kev_runtime').rglob('*.py'))
    lock={'time':now(),'state':'FROZEN','models':models,'identities':identities,'unavailable_before_scores':exceptions,'seeds':config['seeds'],'recipes':config,
          'code_sha256':{f:file_hash(ROOT/f) for f in files},'artifact_sha256':{f:file_hash(ROOT/f) for f in artifacts},
          'dependency_source_sha256':{str(p):file_hash(p) for p in sorted(dependency_paths)},
          'E_new_public_sha256':file_hash(ROOT/'data_public/E_new.json'),'scoring':'planned denominator; macro equal dataset; multiclass summed Brier; stable NLL; ECE15 primary; paired10000 stratified stem clusters',
          'C1':read(ROOT/'configs/cascade_policy.json'),'B2':'complete cyclic rotations on identical public-content safety subset; inverse map before metrics',
          'D_order':'D1,D2,D3,D4,D5; resource availability only; locked before new model forward','C2':'independent generated task protocol with separate pre-formal library lock; no MCQA E_new-driven method changes'}
    save(dest,lock);save(ROOT/'manifests/models_v2.json',identities)
    s=read(ROOT/'status.json');s['E_new']='FROZEN methods; authorized producer may evaluate fixed240 items';s['time']=now();save(ROOT/'status.json',s)
    decision('G2: All available conditions/checkpoints, own C_T calibrators, C_gate policy, cyclic mapping, scoring and D matrix locked at '+lock['time']+'. E_new remains unscored at lock. Exceptions retained before new score exposure; no result-dependent branch selection.')
    print('FROZEN',len(models),'models;',len(exceptions),'unavailable',flush=True)
if __name__=='__main__':main()
