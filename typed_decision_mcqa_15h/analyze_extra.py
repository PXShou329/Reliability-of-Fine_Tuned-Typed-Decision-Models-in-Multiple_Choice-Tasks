"""Cached-output analysis only: no model, prompt, checkpoint, or main score changes."""
import math
import numpy as np
import torch
from utils import ROOT,read,save
from prepare_data import question_fingerprint
from metrics import score,aggregate
from evaluate_results import bootstrap_pair,write_csv

def main():
    torch.set_num_threads(4)
    gold={i:g for d in ['arc','csqa','obqa'] for i,g in read(ROOT/'evaluator_private/gold'/(d+'.json')).items()}
    groups={};raw_groups={};globals={};selective=[];confidence=[]
    for m in ['laya_base','laya_finetuned','clef_flash']:
        rows=[];uncal=[]
        for path in (ROOT/'raw'/m/'items').glob('*.json'):
            r=read(path);i=r['item_id'];g=gold[i]['gold_index']
            shared={'dataset':r['dataset'],'item_id':i,'latency_seconds':r['latency_seconds'],'question_cluster':question_fingerprint(r['public']['state'])}
            rows.append({**shared,**score(r['response']['probabilities'],g)})
            z=np.array(r['response']['logits'],dtype=np.float64);p=np.exp(z-z.max());p/=p.sum()
            uncal.append({**shared,**score(p,g)})
        groups[m]=rows;raw_groups[m]=uncal
        globals[m]=aggregate(rows)
        globals[m]['mean_confidence']=float(np.mean([r['confidence'] for r in rows]))
        raw_records=[read(p) for p in (ROOT/'raw'/m/'items').glob('*.json')]
        tokens=[r['response']['input_tokens'] for r in raw_records]
        globals[m]['input_tokens_mean']=float(np.mean(tokens));globals[m]['input_tokens_max']=max(tokens)
        runmeta=[read(p) for p in (ROOT/'raw'/m).glob('run_metadata_*.json')]
        globals[m]['load_seconds']=[r['load_seconds'] for r in runmeta]
        device=[read(ROOT/'resource_logs'/(r['run_id']+'.json'))['sampled_max_device_used_mib'] for r in runmeta]
        globals[m]['sampled_max_total_device_memory_mib']=max(device)
        globals[m]['torch_inference_peak']=read(ROOT/'raw'/m/'checkpoint.json')
        for d in ['arc','csqa','obqa','pooled']:
            selected=rows if d=='pooled' else [r for r in rows if r['dataset']==d]
            hi=[r for r in selected if r['confidence']>=.9]
            confidence.append({'model':m,'dataset':d,'n':len(selected),'high_confidence_n':len(hi),
                'coverage':len(hi)/len(selected),'wrong_n':sum(not r['correct'] for r in hi),
                'accuracy_when_confidence_ge_point9':sum(r['correct'] for r in hi)/len(hi) if hi else None,
                'mean_confidence':float(np.mean([r['confidence'] for r in selected])),'accuracy':float(np.mean([r['correct'] for r in selected]))})
            ordered=sorted(selected,key=lambda r:(-r['confidence'],r['item_id']))
            for coverage in [.5,.7,.9]:
                k=round(len(selected)*coverage);top=ordered[:k]
                selective.append({'model':m,'dataset':d,'retained_fraction':coverage,'retained_n':k,
                                  'accuracy':float(np.mean([r['correct'] for r in top])),
                                  'min_retained_confidence':top[-1]['confidence']})
    raw_metrics={m:aggregate(rs) for m,rs in raw_groups.items() if m.startswith('laya')}
    raw_delta={metric:bootstrap_pair(raw_groups['laya_base'],raw_groups['laya_finetuned'],metric) for metric in ['correct','brier','nll']}
    # Repeat official LBFGS on saved TRAIN-only calibration logits, preserve unbounded optimum for diagnosis.
    cal=read(ROOT/'training_calibration_outputs.json');preds=cal['rows'];kmax=max(len(r['logits']) for r in preds)
    z=torch.full((len(preds),kmax),-1e4);targets=torch.zeros_like(z)
    for i,r in enumerate(preds): z[i,:len(r['logits'])]=torch.tensor(r['logits']);targets[i,r['gold_index']]=1
    log_t=torch.zeros(1,requires_grad=True);opt=torch.optim.LBFGS([log_t],lr=.1,max_iter=100)
    def closure():
        opt.zero_grad();loss=-(targets*torch.log_softmax(z/log_t.exp(),-1)).sum(-1).mean();loss.backward();return loss
    opt.step(closure);unbounded=float(log_t.exp().item())
    probes=[{'temperature':t,'TRAIN_calibration_nll':float(-(targets*torch.log_softmax(z/t,-1)).sum(-1).mean())} for t in [1.,2.,3.,5.,unbounded]]
    boundary={'analysis':'post-hoc investigation of active official temperature bound, on train-only calibration data',
              'n':len(preds),'unbounded_LBFGS_optimum':unbounded,'official_bound':[.5,5.],'primary_temperature_unchanged':5.,'probes':probes,
              'formal_eval_at_new_temperature':'NOT performed; no tuning on evaluation outcomes'}
    save(ROOT/'results/calibration_boundary_diagnostic.json',boundary)
    save(ROOT/'results/extra_analysis.json',{'pooled_metrics':globals,'T1_secondary_pooled':raw_metrics,'T1_paired_deltas':raw_delta,
                                           'confidence_summary':confidence,'selective_accuracy':selective,'temperature_bound_diagnostic':boundary})
    write_csv(ROOT/'results/high_confidence.csv',confidence);write_csv(ROOT/'results/selective_accuracy.csv',selective)
    print({'pooled':{m:{k:v[k] for k in ['accuracy','mean_confidence','latency_median_seconds','sampled_max_total_device_memory_mib']} for m,v in globals.items()},'T1_pooled':{m:{k:v[k] for k in ['accuracy','brier','nll','ece','nll_clipped']} for m,v in raw_metrics.items()},'calibration_boundary':boundary},flush=True)

if __name__=='__main__': main()
