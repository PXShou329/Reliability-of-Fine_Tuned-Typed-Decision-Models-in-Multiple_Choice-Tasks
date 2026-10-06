"""Prescribed position tables, evaluated after all frozen formal production."""
import collections,csv
from utils import *
from bootstrap_stats import paired
def main():
    assert read(ROOT/'results/analysis_complete.json')['status']=='COMPLETED'
    hist=[];strata=[];comparisons=[]
    models=['base','B1_grpo_ce_20261006','B1_ce_20261006','kev','clef']
    cases=list(csv.DictReader((ROOT/'results/order_robustness_cases.csv').open(encoding='utf-8-sig')))
    for purpose in ['E_old','E_new']:
        gold=read(ROOT/'evaluator_private'/(purpose+'.json'));public={r['item_id']:r for r in read(ROOT/'data_public'/(purpose+'.json'))}
        for model in models:
            records=[read(p) for p in (ROOT/'raw'/model/('B2_'+purpose)).glob('*.json')]
            for subset in ['all_cyclic','identity_rotation0']:
                by=collections.defaultdict(list)
                for r in records:
                    if subset=='identity_rotation0' and r['rotation']!=0:continue
                    by[r['dataset'],len(r['mapping_to_original'])].append(r)
                for (ds,k),rs in sorted(by.items()):
                    choices=[]
                    for r in rs:
                        j=max(range(k),key=lambda q:r['actual_output']['logits'][q]) if r['status']=='COMPLETED' else None
                        g=r['mapping_to_original'].index(gold[r['item_id']]['gold_index']);choices.append((j,g,r['status']))
                    for position in range(k):
                        q=[x for x in choices if x[1]==position]
                        hist.append({'purpose':purpose,'model':model,'dataset':ds,'k':k,'subset':subset,'display_position':chr(65+position),'raw_views_n':len(rs),
                            'original_questions_n':len({r['item_id'] for r in rs}),'predicted_n':sum(j==position for j,g,s in choices),'predicted_fraction':sum(j==position for j,g,s in choices)/len(rs),
                            'gold_n':len(q),'gold_fraction':len(q)/len(rs),'failed_n':sum(s!='COMPLETED' for j,g,s in choices),'descriptive_correlated_views':True})
                        strata.append({'purpose':purpose,'model':model,'dataset':ds,'k':k,'subset':subset,'gold_display_position':chr(65+position),'n':len(q),
                            'correct':sum(j==g for j,g,s in q),'accuracy':sum(j==g for j,g,s in q)/len(q) if q else None,'unit':'correlated cyclic observations; no independent-view CI'})
        for A,B in [('B1_grpo_ce_20261006','base'),('B1_ce_20261006','base'),('B1_grpo_ce_20261006','B1_ce_20261006')]:
            aa=[r for r in cases if r['purpose']==purpose and r['model']==A];bb=[r for r in cases if r['purpose']==purpose and r['model']==B]
            for metric in ['any_answer_flip','permutation_accuracy','all_rotation_correct','mean_TV_to_original']:
                x=[{'item_id':r['item_id'],'dataset':r['dataset'],'correct':float(r[metric])} for r in aa];y=[{'item_id':r['item_id'],'dataset':r['dataset'],'correct':float(r[metric])} for r in bb]
                for c in paired(x,y,public,metrics=('correct',)):comparisons.append({'purpose':purpose,'A':A,'B':B,'estimand':metric,**c})
    csv_write(ROOT/'results/order_position_histogram.csv',hist);csv_write(ROOT/'results/order_gold_position.csv',strata);csv_write(ROOT/'results/order_model_paired.csv',comparisons)
    save(ROOT/'results/order_positions_complete.json',{'time':now(),'status':'COMPLETED','source_sha256':file_hash(ROOT/'order_positions.py'),'histogram_rows':len(hist),'strata_rows':len(strata),'paired_rows':len(comparisons),'model_methods_unchanged':True})
    print('prescribed position tables saved',len(hist),'rows')
if __name__=='__main__':main()
