"""Independent scalar-math recomputation; no import of primary scorer."""
import csv,math,collections
from utils import *

def metric(rs):
    n=len(rs);valid=sum(r['valid'] for r in rs);out={'planned_n':n,'valid_n':valid,'invalid_or_missing_n':n-valid,'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/n}
    for key in ['nll','brier']:
        out[key]=math.fsum(r[key] for r in rs)/n if all(r[key] is not None for r in rs) else None
    for bins in [15,10]:
        e=None
        if all(r['confidence'] is not None for r in rs):
            cells=collections.defaultdict(list)
            for r in rs:cells[min(bins-1,int(r['confidence']*bins))].append(r)
            e=math.fsum(abs(math.fsum(r['correct']-r['confidence'] for r in q))/n for q in cells.values())
        out['ece'+str(bins)]=e
    return out

def main():
    lock=read(ROOT/'manifests/protocol.lock.json');cal=read(ROOT/'calibration/calibrators.json');config=read(ROOT/'configs/training.json')
    actual={ (r['purpose'],r['model'],r['view'],r['dataset']):r for r in csv.DictReader((ROOT/'results/mcqa_metrics.csv').open(encoding='utf-8-sig'))};checks=[];maxdiff=0.;raw_count=0
    for purpose in ['E_old','E_new']:
        public=read(ROOT/'data_public'/(purpose+'.json'));gold=read(ROOT/'evaluator_private'/(purpose+'.json'))
        for model in lock['models']:
            target=config['runs'].get(model,{}).get('target');items=[r for r in public if target not in ['arc','csqa','obqa'] or r['dataset']==target]
            for view in (['Generated'] if model=='generative' else ['Raw','Native_Historical','Shared_fit']):
                rows=[]
                for r in items:
                    p=ROOT/'raw'/model/purpose/(digest([r['item_id'],0])+'.json');record=read(p) if p.exists() else {'status':'MISSING'}
                    out=record.get('actual_output',{});g=gold[r['item_id']]['gold_index'];v={'dataset':r['dataset'],'valid':False,'correct':0,'nll':None,'brier':None,'confidence':None}
                    if record['status']=='COMPLETED':
                        assert record['public_hash']==digest(r['public']);assert record['public']==r['public']
                        if model=='generative':v.update(valid=out['valid_answer'],correct=int(out['valid_answer'] and out['prediction_index']==g))
                        else:
                            T=1. if view=='Raw' else cal[model]['T'] if view=='Shared_fit' else out.get('temperature',1.)
                            z=[float(x)/T for x in out['logits']];m=max(z);normalizer=math.fsum(math.exp(x-m) for x in z);prob=[math.exp(x-m)/normalizer for x in z];j=max(range(len(z)),key=lambda k:z[k])
                            v.update(valid=True,correct=int(j==g),nll=m+math.log(normalizer)-z[g],brier=math.fsum((q-int(k==g))**2 for k,q in enumerate(prob)),confidence=prob[j])
                        if view in ['Raw','Generated']:raw_count+=1
                    rows.append(v)
                ds={d:metric([r for r in rows if r['dataset']==d]) for d in sorted({r['dataset'] for r in rows})}
                macro={k:(sum(a[k] for a in ds.values()) if k in ['planned_n','valid_n','invalid_or_missing_n','correct'] else math.fsum(a[k] for a in ds.values())/len(ds) if all(a[k] is not None for a in ds.values()) else None) for k in next(iter(ds.values()))}
                for d,independent in {**ds,'macro':macro}.items():
                    mainrow=actual[purpose,model,view,d]
                    for key,value in independent.items():
                        reported=mainrow[key]
                        if value is None:assert reported=='',(purpose,model,view,d,key)
                        else:
                            diff=abs(float(reported)-value);maxdiff=max(maxdiff,diff);assert diff<1e-9,(purpose,model,view,d,key,diff)
                    checks.append({'purpose':purpose,'model':model,'view':view,'dataset':d,'planned_n':independent['planned_n'],'passed':True})
    save(ROOT/'audit/v2_independent_recompute.json',{'time':now(),'status':'COMPLETED','auditor':'separate scalar math; no primary scoring import','raw_original_records':raw_count,'metric_groups':len(checks),'max_absolute_difference':maxdiff,'checks':checks,'human_verified':False})
    print('independent V2 recomputation passed',raw_count,'raw records;',maxdiff,'maximum difference',flush=True)

if __name__=='__main__':main()
