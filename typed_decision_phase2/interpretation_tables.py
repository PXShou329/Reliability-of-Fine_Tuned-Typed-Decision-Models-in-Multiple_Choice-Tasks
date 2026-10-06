"""Post-evaluation descriptive summaries; no method selection or exclusion."""
import csv,collections,statistics
from utils import *
def rows(name):return list(csv.DictReader((ROOT/'results'/name).open(encoding='utf-8-sig')))
def main():
    cs=rows('case_level.csv');metrics=rows('mcqa_metrics.csv');changes=[];confidence=[]
    for purpose in ['E_old','E_new']:
        baseline={r['item_id']:r for r in cs if r['model']=='base' and r['purpose']==purpose and r['view']=='Raw'}
        for model in ['v1_ft_history','B1_grpo_ce_20261006','B1_ce_20261006','B1_grpo_ce_1729','B1_ce_1729','B1_grpo_ce_2718','B1_ce_2718']:
            rs=[r for r in cs if r['model']==model and r['purpose']==purpose and r['view']=='Raw']
            changes.append({'purpose':purpose,'model':model,'n':len(rs),'gains':sum(r['correct']=='1' and baseline[r['item_id']]['correct']=='0' for r in rs),
                 'regressions':sum(r['correct']=='0' and baseline[r['item_id']]['correct']=='1' for r in rs),'unchanged_predictions':sum(r['prediction_index']==baseline[r['item_id']]['prediction_index'] for r in rs)})
        for model in ['base','B1_grpo_ce_20261006','B1_ce_20261006','kev','clef']:
            for view in ['Raw','Native_Historical','Shared_fit']:
                for ds in ['arc','csqa','obqa','pooled']:
                    rs=[r for r in cs if r['model']==model and r['purpose']==purpose and r['view']==view and (ds=='pooled' or r['dataset']==ds)]
                    for threshold in [.7,.9,.95]:
                        accepted=[r for r in rs if r['confidence'] and float(r['confidence'])>=threshold];wrong=sum(r['correct']=='0' for r in accepted)
                        confidence.append({'purpose':purpose,'model':model,'view':view,'dataset':ds,'threshold':threshold,'planned_n':len(rs),'accepted':len(accepted),'wrong':wrong,
                            'coverage':len(accepted)/len(rs),'wrong_over_accepted':wrong/len(accepted) if accepted else None,'descriptive_post_evaluation':True})
    csv_write(ROOT/'results/gain_regression.csv',changes);csv_write(ROOT/'results/high_confidence_errors.csv',confidence)
    timing=rows('timing_blocks.csv');summary=[]
    for model in sorted({r['model'] for r in timing}):
        blocks=[]
        for b in sorted({r['block'] for r in timing if r['model']==model}):blocks.append(statistics.mean(float(r['latency_ms']) for r in timing if r['model']==model and r['block']==b))
        summary.append({'model':model,'blocks':len(blocks),'request_n':sum(r['model']==model for r in timing),'block_mean_ms':canon(blocks),'mean_ms':statistics.mean(blocks),'block_sample_sd_ms':statistics.stdev(blocks)})
    csv_write(ROOT/'results/timing_summary.csv',summary)
    out={'time':now(),'status':'COMPLETED','baseline_macro':[{k:r[k] for k in ['purpose','model','view','planned_n','correct','accuracy','nll','brier','ece15','mean_latency_ms','invalid_or_missing_n']} for r in metrics if r['dataset']=='macro' and r['view'] in ['Raw','Shared_fit','Generated'] and r['model'] in ['base','v1_ft_history','kev','clef','generative','multilingual']],
       'gain_regression':changes,'timing_summary':summary,'note':'descriptive summaries after formal freeze; not new methods, no changed scoring or selected data'}
    save(ROOT/'reports/interpretation_data.json',out);print('interpretation tables saved')
if __name__=='__main__':main()
