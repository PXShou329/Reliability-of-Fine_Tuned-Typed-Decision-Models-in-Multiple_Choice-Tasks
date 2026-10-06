"""Delivery projections of already-scored tables; no model/method selection."""
import csv
from utils import *
def rows(name):return list(csv.DictReader((ROOT/'results'/name).open(encoding='utf-8-sig')))
def main():
    assert read(ROOT/'results/analysis_complete.json')['status']=='COMPLETED'
    metrics=rows('mcqa_metrics.csv');training=read(ROOT/'configs/training.json');trained={r['run']:r for r in rows('training_matrix.csv')}
    baseline=[r for r in metrics if r['model'] in ['base','v1_ft_history','kev','clef','generative','multilingual']]
    csv_write(ROOT/'results/baseline_completion.csv',baseline)
    csv_write(ROOT/'results/calibration_comparison.csv',metrics)
    transfer=[];budget=[]
    for r in metrics:
        model=r['model'];rc=training['runs'].get(model,{})
        if rc.get('target') in ['arc','csqa','obqa'] and r['dataset']==rc['target']:
            matched=('B3_' if rc['seed']==training['s0'] else 'D4_')+'matched_all_'+str(rc['seed'])
            ref=next(q for q in metrics if q['purpose']==r['purpose'] and q['model']==matched and q['view']==r['view'] and q['dataset']==r['dataset'])
            transfer.append({**r,'seed':rc['seed'],'matched_model':matched,'matched_accuracy':ref['accuracy'],'accuracy_minus_matched':float(r['accuracy'])-float(ref['accuracy']),
                'matched_nll':ref['nll'],'nll_minus_matched':float(r['nll'])-float(ref['nll']),'G_n':len(rc['G_ids']),'C_T_n':len(rc['C_T_ids'])})
        if model.startswith('D2_') or model=='B1_ce_20261006':
            if r['dataset']=='macro':budget.append({**r,'samples_per_dataset':len(rc['G_ids'])//3,'unique_G_n':len(rc['G_ids']),
                'item_exposures':trained[model]['item_exposures'],'optimizer_steps':trained[model]['updates'],'training_seconds':trained[model]['training_seconds'],'B1_900_point_reused':model=='B1_ce_20261006'})
    csv_write(ROOT/'results/transfer_results.csv',transfer);csv_write(ROOT/'results/data_budget.csv',budget)
    modules=[('A0','A0_audit',['audit/audit_v1.json']),('A1','A1_kev',['results/inference_kev_pre.json','results/inference_kev_new.json']),
        ('B1','B1',['results/ablation_by_seed.csv','audit/training_audit.json','audit/v2_independent_recompute.json']),
        ('B2','B2_order',['results/order_robustness.csv']),('B3','B3',['results/transfer_results.csv']),('C1','C1_live_cascade',['results/cascade_execution.json','results/cascade_conclusion.json']),
        ('C2','C2_macrolab',['macrolab/evaluation_complete.json']),('D1','D1_generative',['results/inference_generative_new.json']),('D2','D2_data_curve',['results/data_budget.csv']),
        ('D3','D3_smoothing',['results/recipe_seed_summary.csv']),('D4','D4_lodo_seeds',['results/transfer_results.csv']),('D5','D5_laya_variant',['results/inference_multilingual_new.json'])]
    completion=[]
    expected_models={'A1':['kev'],'B1':[m for m in training['runs'] if m.startswith('B1_')],'B3':[m for m in training['runs'] if m.startswith('B3_')],
       'D1':['generative'],'D2':[m for m in training['runs'] if m.startswith('D2_')]+['B1_ce_20261006'],'D3':[m for m in training['runs'] if m.startswith('D3_')],
       'D4':[m for m in training['runs'] if m.startswith('D4_')],'D5':['multilingual']}
    for module,run,evidence in modules:
        available=all((ROOT/p).exists() for p in evidence);status='COMPLETED' if available else read(ROOT/'status.json')['runs'].get(run,'NOT_STARTED')
        expected=expected_models.get(module,[]);complete_models=[]
        for model in expected:
            paths=[ROOT/'results'/('inference_'+model+'_'+stage+'.json') for stage in ['pre','new']]
            if all(p.exists() and read(p)['status']=='COMPLETED' and read(p)['failed']==0 for p in paths):complete_models.append(model)
        if expected and len(complete_models)!=len(expected):status=read(ROOT/'status.json')['runs'].get(run,'FAILED');status='FAILED' if status in ['RUNNING','PLANNED','COMPLETED'] else status
        if module=='B2':
            expected=['base','B1_grpo_ce_20261006','B1_ce_20261006','kev','clef'];complete_models=[]
            if (ROOT/'results/order_robustness_cases.csv').exists():
                order=rows('order_robustness_cases.csv')
                for model in expected:
                    if all(len([r for r in order if r['model']==model and r['purpose']==purpose])==n for purpose,n in [('E_old',118),('E_new',235)]):complete_models.append(model)
            status='COMPLETED' if len(complete_models)==5 else 'FAILED'
        if module=='C2' and available:status=read(ROOT/'macrolab/evaluation_complete.json')['status']
        if module=='C1' and available:status=read(ROOT/'results/cascade_execution.json')['status']
        planned_conditions=len(expected);completed_conditions=len(complete_models);unit='model/checkpoint old-new evaluation';planned_outputs=sum(280 if training['runs'].get(m,{}).get('target') in ['arc','csqa','obqa'] else 840 for m in expected)
        completed_outputs=sum(1 for m in expected for purpose in ['E_old','E_new'] for p in (ROOT/'raw'/m/purpose).glob('*.json'))
        if module=='A0':planned_conditions=1;completed_conditions=int(status=='COMPLETED');unit='evidence audit';planned_outputs=1;completed_outputs=int(available)
        if module=='B2':unit='model cyclic-view evaluation';planned_outputs=7650;completed_outputs=sum(1 for m in expected for purpose in ['B2_E_old','B2_E_new'] for p in (ROOT/'raw'/m/purpose).glob('*.json'))
        if module=='C1':planned_conditions=4;completed_conditions=4 if status=='COMPLETED' else 0;unit='live routing policy';planned_outputs=960;completed_outputs=sum(1 for p in (ROOT/'raw/cascade/E_new').glob('*.json'))
        if module=='C2':planned_conditions=8;completed_conditions=8 if status=='COMPLETED' else 0;unit='formal model/control condition';planned_outputs=320;completed_outputs=read(ROOT/'macrolab/evaluation_complete.json')['completed_condition_episodes'] if available else 0
        completion.append({'module':module,'status':status,'planned_model_conditions':len(expected),'completed_model_conditions':len(complete_models),'planned_conditions':planned_conditions,'completed_conditions':completed_conditions,'condition_unit':unit,'planned_outputs':planned_outputs,'completed_outputs':completed_outputs,'evidence':';'.join(evidence),'actual_output_exists':available,'note':'AI evidence review; human reviewed0' if module=='A0' else '900 point reused from B1 CE/s0; no duplicated training' if module=='D2' else 'optional second/third LODO seeds; all folds retained' if module=='D4' else 'outputs overlap other modules; do not sum matrix counts'})
        if run in read(ROOT/'status.json')['runs'] and available:set_run(run,status,';'.join(evidence))
    if all((ROOT/'results'/('inference_'+m+'_new.json')).exists() for m in ['v1_ft_history']):set_run('v1_ft_history','COMPLETED','results/baseline_completion.csv','reused V1 historical background; not controlled B1')
    csv_write(ROOT/'results/completion_matrix.csv',completion)
    outputs=[]
    for p in (ROOT/'results').glob('inference_*.json'):
        if p.stem.endswith(('_pre','_new')):
            r=read(p);outputs.append({'run':p.stem,'status':r['status'],'planned':r['planned'],'saved':r['new_calls']+r['previously_saved'],'failed':r['failed'],'evidence':str(p.relative_to(ROOT))})
    csv_write(ROOT/'results/request_completion.csv',outputs)
    save(ROOT/'results/delivery_counts.json',{'time':now(),'MCQA_saved_requests':sum(r['saved'] for r in outputs),'MCQA_failed_requests':sum(r['failed'] for r in outputs),
       'fresh_training_runs':len(trained),'E_old_unique_n':600,'E_new_unique_n':240,'original_new_condition_outputs':sum(1 for p in (ROOT/'raw').glob('*/E_new/*.json') if p.parent.parent.name!='cascade'),
       'original_old_condition_outputs':sum(1 for p in (ROOT/'raw').glob('*/E_old/*.json')),'rotated_old_outputs':sum(1 for p in (ROOT/'raw').glob('*/B2_E_old/*.json')),
       'rotated_new_outputs':sum(1 for p in (ROOT/'raw').glob('*/B2_E_new/*.json')),'counts_exclude_warmups_training_forwards_and_macro':True})
    print('delivery tables saved',len(metrics),'metric rows; modules',len(completion))
if __name__=='__main__':main()
