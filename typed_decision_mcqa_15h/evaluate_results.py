"""Evaluator reads saved public responses + private gold, with no model invocation."""
import csv,json,math
import numpy as np
from prepare_data import question_fingerprint
from utils import ROOT,read,save,digest,event,now
from metrics import score,aggregate,self_test

MODELS=['laya_base','laya_finetuned','clef_flash']
DATASETS=['arc','csqa','obqa']

def write_csv(path,rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def bootstrap_pair(a,b,metric,reps=10000):
    rng=np.random.default_rng(20261006);diffs=[];observed=[]
    for d in DATASETS:
        aa={r['item_id']:r for r in a if r['dataset']==d};bb={r['item_id']:r for r in b if r['dataset']==d}
        if set(aa)!=set(bb) or not aa: return None
        ids=sorted(aa);delta=np.array([bb[i][metric]-aa[i][metric] for i in ids]);observed.append(float(delta.mean()))
        clusters={}
        for j,i in enumerate(ids): clusters.setdefault(aa[i]['question_cluster'],[]).append(j)
        totals=np.array([delta[js].sum() for js in clusters.values()]);sizes=np.array([len(js) for js in clusters.values()])
        samples=rng.integers(0,len(clusters),size=(reps,len(clusters)))
        diffs.append(totals[samples].sum(1)/sizes[samples].sum(1))
    vals=np.mean(diffs,axis=0)
    return {'delta_b_minus_a':float(np.mean(observed)),'bootstrap_95_ci':np.percentile(vals,[2.5,97.5]).tolist(),
            'repetitions':reps,'seed':20261006,'method':'paired bootstrap stratified by dataset; repeated normalized stems clustered within dataset; no seed/checkpoint search'}

def main():
    self_test();conf=read(ROOT/'configs/protocol.json');ph=digest(conf)
    public={r['item_id']:r for d in DATASETS for r in read(ROOT/'data/public_eval'/(d+'.json'))}
    gold={i:g for d in DATASETS for i,g in read(ROOT/'evaluator_private/gold'/(d+'.json')).items()}
    revisions=read(ROOT/'sources/revisions.json')
    all_rows=[];groups={};coverage={}
    for model in MODELS:
        rows=[];rawfiles=list((ROOT/'raw'/model/'items').glob('*.json'))
        coverage[model]={'completed':len(rawfiles),'expected':len(public),'by_dataset':{d:0 for d in DATASETS}}
        for path in rawfiles:
            r=read(path);i=r['item_id']
            if r['phase']!='formal' or i not in public or r['protocol_hash']!=ph or r['request_hash']!=digest(public[i]['public']): raise RuntimeError('Raw record mismatch')
            g=gold[i]['gold_index'];p=r['response']['probabilities'];s=score(p,g)
            q=public[i]['public']['questions']['answer'];keys=list(q['criteria'])
            row={'model':model,'model_id':'Cloudflare/clef-flash' if model=='clef_flash' else ('local/laya-mcqa-local-20261006' if model=='laya_finetuned' else 'convaiinnovations/laya'),
                 'source_revision':revisions['clef_flash' if model=='clef_flash' else 'laya'],
                 'dataset':r['dataset'],'split':r['split'],'item_id':i,'run_id':r['run_id'],'question_cluster':question_fingerprint(public[i]['public']['state']),
                 'predicted_option':keys[s['pred_index']],'gold_option':keys[g],'original_gold_label':gold[i]['original_answer_label'],
                 'probabilities':json.dumps(dict(zip(keys,p))),'logits':json.dumps(r['response']['logits']),
                 'temperature':r['response']['temperature'],'input_tokens':r['response']['input_tokens'],
                 'latency_seconds':r['latency_seconds'],**s}
            rows.append(row);coverage[model]['by_dataset'][r['dataset']]+=1
        if len({r['item_id'] for r in rows})!=len(rows): raise RuntimeError('Duplicate raw predictions')
        groups[model]=rows;all_rows+=rows
    tables=[];calibration=[];metrics={}
    for model,rows in groups.items():
        metrics[model]={}
        for d in DATASETS:
            selected=[r for r in rows if r['dataset']==d];m=aggregate(selected);metrics[model][d]=m
            tables.append({'model':model,'dataset':d,**{k:v for k,v in m.items() if k!='bins'}})
            for b in m.get('bins',[]): calibration.append({'model':model,'dataset':d,**b})
        if all(metrics[model][d]['n']==len([i for i in public if public[i]['dataset']==d]) for d in DATASETS):
            metrics[model]['macro']={k:float(np.mean([metrics[model][d][k] for d in DATASETS])) for k in ['accuracy','brier','nll','ece']}
    pairs={}
    for a,b in [('laya_base','laya_finetuned'),('laya_base','clef_flash'),('laya_finetuned','clef_flash')]:
        if len(groups[a])==len(public) and len(groups[b])==len(public):
            pairs[a+'__'+b]={m:bootstrap_pair(groups[a],groups[b],m) for m in ['correct','brier','nll']}
            pairs[a+'__'+b]['by_dataset']={d:{'a_wrong_b_right':sum(not aa['correct'] and bb['correct'] for aa,bb in _paired(groups[a],groups[b],d)),
                                            'a_right_b_wrong':sum(aa['correct'] and not bb['correct'] for aa,bb in _paired(groups[a],groups[b],d))} for d in DATASETS}
    uncalibrated=[]
    for model,rows in groups.items():
        if not model.startswith('laya'): continue
        for r in rows:
            z=np.array(json.loads(r['logits']),dtype=np.float64);p=np.exp(z-z.max());p/=p.sum()
            g=gold[r['item_id']]['gold_index'];uncalibrated.append({**r,'analysis_model':model+'_T1',**score(p,g)})
    uncal_metrics={model:{d:aggregate([r for r in uncalibrated if r['analysis_model']==model and r['dataset']==d]) for d in DATASETS}
                   for model in ['laya_base_T1','laya_finetuned_T1']}
    random_expected={d:{'n':sum(r['dataset']==d for r in public.values()),
                       'expected_accuracy':float(np.mean([1/len(r['public']['questions']['answer']['criteria']) for r in public.values() if r['dataset']==d])),
                       'uniform_brier':float(np.mean([1-1/len(r['public']['questions']['answer']['criteria']) for r in public.values() if r['dataset']==d])),
                       'uniform_nll':float(np.mean([math.log(len(r['public']['questions']['answer']['criteria'])) for r in public.values() if r['dataset']==d]))} for d in DATASETS}
    save(ROOT/'results/metrics.json',{'generated_at':now(),'coverage':coverage,'metrics':metrics,'paired_comparisons':pairs,
                'uncalibrated_secondary':uncal_metrics,'random_uniform_analytical':random_expected})
    write_csv(ROOT/'results/predictions.csv',sorted(all_rows,key=lambda r:(r['model'],r['dataset'],r['item_id'])))
    # n=0 metrics rows have different keys; omit until actual inference exists, never invent zeros.
    write_csv(ROOT/'results/metrics_by_dataset.csv',[r for r in tables if r['n']])
    write_csv(ROOT/'results/calibration.csv',calibration)
    write_csv(ROOT/'results/model_comparison.csv',[{'model':m,**v['macro']} for m,v in metrics.items() if 'macro' in v])
    write_csv(ROOT/'results/error_cases.csv',[r for r in all_rows if not r['correct']])
    event('evaluation_saved',coverage=coverage)
    print({'coverage':coverage,'macro':{m:v.get('macro') for m,v in metrics.items()}},flush=True)

def _paired(a,b,d):
    aa={r['item_id']:r for r in a if r['dataset']==d};bb={r['item_id']:r for r in b if r['dataset']==d}
    return [(aa[i],bb[i]) for i in sorted(aa.keys()&bb.keys())]

if __name__=='__main__': main()
