"""Evaluator of live records only, separate from routing and policy selection."""
import numpy as np
from utils import *
from bootstrap_stats import paired
def main():
    gold=read(ROOT/'evaluator_private/E_new.json');public={r['item_id']:r for r in read(ROOT/'data_public/E_new.json')};records=[read(p) for p in (ROOT/'raw/cascade/E_new').glob('*.json')];groups={};rows=[];metrics=[]
    for condition in ['always_small','always_expert','confidence','random']:
        rs={r['item_id']:r for r in records if r['condition']==condition};assert set(rs)==set(gold),(condition,len(rs))
        group=[]
        for i,r in rs.items():
            good=r['status']=='COMPLETED';row={'item_id':i,'dataset':r['dataset'],'condition':condition,'status':r['status'],'prediction_index':r.get('prediction_index'),
               'gold_index':gold[i]['gold_index'],'correct':int(good and r.get('prediction_index')==gold[i]['gold_index']),'small_call':r['small_call'],'expert_call':r['expert_call'],
               'small_ms':r['small_ms'],'expert_ms':r['expert_ms'],'end_to_end_ms':r['end_to_end_ms'],'execution_order':canon(r['execution_order'])}
            group.append(row);rows.append(row)
        groups[condition]=group
        for d in ['arc','csqa','obqa','macro']:
            q=[r for r in group if d=='macro' or r['dataset']==d];lat=[r['end_to_end_ms'] for r in q];small=[r for r in q if not r['expert_call'] and r['small_call']]
            metrics.append({'condition':condition,'dataset':d,'n':len(q),'valid_n':sum(r['status']=='COMPLETED' for r in q),'correct':sum(r['correct'] for r in q),
                'accuracy':sum(r['correct'] for r in q)/len(q),'small_calls':sum(r['small_call'] for r in q),'expert_calls':sum(r['expert_call'] for r in q),
                'expert_call_rate':sum(r['expert_call'] for r in q)/len(q),'small_coverage':len(small)/len(q),'small_conditional_error':sum(1-r['correct'] for r in small)/len(small) if small else None,
                'mean_latency_ms':float(np.mean(lat)),'median_latency_ms':float(np.median(lat)),'p95_latency_ms':float(np.quantile(lat,.95)),'summed_request_seconds':sum(lat)/1000,'live':True})
    contrasts=[]
    for condition in ['always_small','confidence','random']:
        for metric in ['correct','end_to_end_ms']:
            for v in paired(groups[condition],groups['always_expert'],public,metrics=(metric,)):
                contrasts.append({'comparison':condition+'_minus_always_expert',**v})
    csv_write(ROOT/'results/cascade_live.csv',rows);csv_write(ROOT/'results/cascade_metrics.csv',metrics);csv_write(ROOT/'results/cascade_paired.csv',contrasts)
    macro={r['condition']:r for r in metrics if r['dataset']=='macro'};C=macro['confidence'];E=macro['always_expert'];ci=[r for r in contrasts if r['comparison']=='confidence_minus_always_expert' and r['metric']=='correct'][0]
    result={'time':now(),'point_accuracy_loss_pp':100*(E['accuracy']-C['accuracy']),'mean_latency_reduction_fraction':1-C['mean_latency_ms']/E['mean_latency_ms'],
        'point_accuracy_target_met':C['accuracy']-E['accuracy']>=-.02-1e-12,'mean_latency_target_met':C['mean_latency_ms']/E['mean_latency_ms']<=.8,
        'accuracy_noninferiority_interval_supported':ci['ci95_low']>-.02,'CI_note':'conditional on frozen policy/checkpoint; item bootstrap not guarantee for deployment distributions',
        'policy':read(ROOT/'configs/cascade_policy.json')['policy'],'status':'COMPLETED','live':True}
    save(ROOT/'results/cascade_conclusion.json',result);print(result)
if __name__=='__main__':main()
