"""Evaluator only. All formal production completes before E_new score visibility."""
import argparse,collections,math
import numpy as np
from utils import *
from scoring import individual,aggregate,distribution,macro_aggregate
from bootstrap_stats import paired
def main():
    a=argparse.ArgumentParser();a.add_argument('--old-only',action='store_true');args=a.parse_args();models=read(ROOT/'manifests/protocol.lock.json')['models']
    if not args.old_only:
        for model in models:assert (ROOT/'results'/('inference_'+model+'_new.json')).exists(),('not all formal producers completed',model)
    purposes=['E_old'] if args.old_only else ['E_old','E_new'];tables=[];cases=[];groups={};perms=[];contrasts=[];training=read(ROOT/'configs/training.json')
    calibrators=read(ROOT/'calibration/calibrators.json')
    for purpose in purposes:
        public={r['item_id']:r for r in read(ROOT/'data_public'/(purpose+'.json'))};gold=read(ROOT/'evaluator_private'/(purpose+'.json'))
        for model in models:
            target=training['runs'].get(model,{}).get('target');ids=[i for i,r in public.items() if target not in ['arc','csqa','obqa'] or r['dataset']==target]
            raws={r['item_id']:r for p in (ROOT/'raw'/model/purpose).glob('*.json') for r in [read(p)]}
            for view in ['Raw','Native_Historical','Shared_fit'] if model!='generative' else ['Generated']:
                rows=[]
                for i in ids:
                    raw=raws.get(i,{'item_id':i,'dataset':public[i]['dataset'],'model':model,'purpose':purpose,'rotation':0,'status':'MISSING'})
                    T=1
                    if view=='Shared_fit':T=calibrators[model]['T']
                    if view=='Native_Historical':T=raw.get('actual_output',{}).get('temperature',1.)
                    row=individual(raw,gold[i],T);row.update(view=view,source_raw=str((ROOT/'raw'/model/purpose/(digest([i,0])+'.json')).relative_to(ROOT)));rows.append(row);cases.append(row)
                groups[purpose,model,view]=rows
                dsrows=[]
                for d in sorted({r['dataset'] for r in rows}):
                    q=[r for r in rows if r['dataset']==d];v=aggregate(q);dsrows.append(v);tables.append({'purpose':purpose,'model':model,'view':view,'dataset':d,**v,'pooled_ece15':None})
                macro=macro_aggregate(dsrows)
                pooled=aggregate(rows);tables.append({'purpose':purpose,'model':model,'view':view,'dataset':'macro',**macro,'pooled_ece15':pooled['ece15']})
            if model in ['base','B1_grpo_ce_20261006','B1_ce_20261006','kev','clef']:
                pr=[read(p) for p in (ROOT/'raw'/model/('B2_'+purpose)).glob('*.json')];byid=collections.defaultdict(list)
                for r in pr:byid[r['item_id']].append(r)
                for i,views in byid.items():
                    k=len(public[i]['public']['questions']['answer']['criteria']);assert len(views)==k
                    pvec=[];choices=[];correct=[]
                    for r in sorted(views,key=lambda r:r['rotation']):
                        assert r['status']=='COMPLETED';p,_=distribution(r['actual_output']['logits'],calibrators[model]['T']);aligned=np.zeros(k)
                        aligned[r['mapping_to_original']]=p;pvec.append(aligned);j=int(np.argmax(aligned));choices.append(j);correct.append(int(j==gold[i]['gold_index']))
                    ensemble=np.mean(pvec,axis=0);drift=np.mean([np.abs(p-pvec[0]).sum()/2 for p in pvec[1:]])
                    perms.append({'purpose':purpose,'model':model,'item_id':i,'dataset':public[i]['dataset'],'k':k,'rotation_correct_n':sum(correct),'permutation_accuracy':sum(correct)/k,
                        'original_correct':correct[0],'all_rotation_correct':int(all(correct)),'any_answer_flip':int(len(set(choices))>1),'ensemble_correct':int(np.argmax(ensemble)==gold[i]['gold_index']),
                        'mean_TV_to_original':float(drift),'max_probability_drift':float(np.max(np.abs(np.array(pvec)-pvec[0]))),'original_choices':canon(choices),
                        'sum_rotation_latency_ms':sum(r['latency_ms'] for r in views),'original_rotation_latency_ms':next(r['latency_ms'] for r in views if r['rotation']==0)})
        def contrast(x,y,label,view='Shared_fit',dataset=None):
            if (purpose,x,view) not in groups or (purpose,y,view) not in groups:return
            A=groups[purpose,x,view];B=groups[purpose,y,view]
            if dataset:A=[r for r in A if r['dataset']==dataset];B=[r for r in B if r['dataset']==dataset]
            for row in paired(A,B,public):contrasts.append({'purpose':purpose,'comparison':label,'A':x,'B':y,'view':view,'dataset':dataset or 'macro',**row})
        for model in models:
            if (purpose,model,'Shared_fit') in groups:
                for row in paired(groups[purpose,model,'Shared_fit'],groups[purpose,model,'Raw'],public,metrics=('nll','brier')):
                    contrasts.append({'purpose':purpose,'comparison':'Shared_minus_Raw_'+model,'A':model,'B':model,'view':'Shared_fit_vs_Raw','dataset':training['runs'].get(model,{}).get('target','macro'),**row})
        for seed in training['seeds']:
            R0='B1_grpo_ce_'+str(seed);R1='B1_ce_'+str(seed)
            for view in ['Raw','Shared_fit']:
                contrast(R0,R1,'R0_minus_R1_seed'+str(seed),view);contrast(R0,'base','R0_minus_Base_seed'+str(seed),view);contrast(R1,'base','R1_minus_Base_seed'+str(seed),view)
                contrast('D3_smooth_'+str(seed),R1,'Smoothing_minus_CE_seed'+str(seed),view)
            for target in ['arc','csqa','obqa']:
                fold=('B3_' if seed==training['s0'] else 'D4_')+target+'_'+str(seed);matched=('B3_' if seed==training['s0'] else 'D4_')+'matched_all_'+str(seed)
                contrast(fold,matched,'LODO_minus_equal_data_all_'+target+'_seed'+str(seed),dataset=target)
        for x in ['kev','clef','multilingual']:contrast(x,'base',x+'_minus_Base_exploratory')
        if (purpose,'generative','Generated') in groups:
            for row in paired(groups[purpose,'generative','Generated'],groups[purpose,'base','Raw'],public,metrics=('correct',)):
                contrasts.append({'purpose':purpose,'comparison':'generative_minus_Base_exploratory','A':'generative','B':'base','view':'Generated_vs_Raw','dataset':'macro',**row})
    suffix='_old' if args.old_only else ''
    csv_write(ROOT/'results'/('mcqa_metrics'+suffix+'.csv'),tables);csv_write(ROOT/'results'/('case_level'+suffix+'.csv'),cases);csv_write(ROOT/'results'/('paired_comparisons'+suffix+'.csv'),contrasts)
    csv_write(ROOT/'results'/('order_robustness_cases'+suffix+'.csv'),perms)
    order=[]
    for purpose,model in sorted({(r['purpose'],r['model']) for r in perms}):
        for d in ['arc','csqa','obqa','macro']:
            q=[r for r in perms if r['purpose']==purpose and r['model']==model and (d=='macro' or r['dataset']==d)]
            if q:
                means={k:float(np.mean([np.mean([r[k] for r in q if r['dataset']==ds]) for ds in sorted({x['dataset'] for x in q})])) for k in ['permutation_accuracy','original_correct','all_rotation_correct','any_answer_flip','ensemble_correct','mean_TV_to_original','max_probability_drift','sum_rotation_latency_ms','original_rotation_latency_ms']}
                order.append({'purpose':purpose,'model':model,'dataset':d,'n':len(q),**means})
    csv_write(ROOT/'results'/('order_robustness'+suffix+'.csv'),order)
    order_ci=[]
    for purpose,model in sorted({(r['purpose'],r['model']) for r in perms}):
        public={r['item_id']:r for r in read(ROOT/'data_public'/(purpose+'.json'))};q=[r for r in perms if r['purpose']==purpose and r['model']==model]
        for a,b,label in [('permutation_accuracy','original_correct','permutation_minus_original'),('ensemble_correct','original_correct','ensemble_minus_original'),('any_answer_flip',None,'answer_flip_rate'),('mean_TV_to_original',None,'mean_TV_drift')]:
            aa=[{'item_id':r['item_id'],'dataset':r['dataset'],'correct':r[a]} for r in q];bb=[{'item_id':r['item_id'],'dataset':r['dataset'],'correct':r[b] if b else 0.} for r in q]
            for c in paired(aa,bb,public,metrics=('correct',)):
                order_ci.append({'purpose':purpose,'model':model,'estimand':label,**c})
    csv_write(ROOT/'results'/('order_robustness_paired'+suffix+'.csv'),order_ci)
    ablation=[];recipe_summaries=[]
    for r in tables:
        if r['dataset']=='macro' and r['model'].startswith(('B1_','D3_')):
            rc=training['runs'][r['model']];ablation.append({'seed':rc['seed'],'recipe':rc['recipe'],'smoothing':rc.get('smoothing',0),**r})
    for purpose,view,recipe,smoothing in sorted({(r['purpose'],r['view'],r['recipe'],r['smoothing']) for r in ablation}):
        q=[r for r in ablation if (r['purpose'],r['view'],r['recipe'],r['smoothing'])==(purpose,view,recipe,smoothing)]
        row={'purpose':purpose,'view':view,'recipe':recipe,'smoothing':smoothing,'seed_n':len(q),'seeds':canon([r['seed'] for r in q])}
        for m in ['accuracy','nll','brier','ece15']:
            values=[r[m] for r in q];row[m+'_mean']=float(np.mean(values)) if all(v is not None for v in values) else None
            row[m+'_sample_sd']=float(np.std(values,ddof=1)) if len(q)>1 and all(v is not None for v in values) else None
        recipe_summaries.append(row)
    csv_write(ROOT/'results'/('ablation_by_seed'+suffix+'.csv'),ablation);csv_write(ROOT/'results'/('recipe_seed_summary'+suffix+'.csv'),recipe_summaries)
    save(ROOT/'results'/('analysis_complete'+suffix+'.json'),{'time':now(),'metrics_rows':len(tables),'case_view_rows':len(cases),'contrasts':len(contrasts),'permutation_original_items':len(perms),'status':'COMPLETED','analyst':'AI; not human adjudication'})
    print('analyzed',len(cases),'case-view rows;',len(perms),'order item-model pairs',flush=True)
if __name__=='__main__':main()
