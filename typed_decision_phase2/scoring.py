"""Evaluator only: explicit gold ownership, never called by model adapters."""
import numpy as np,math
from scipy.special import logsumexp
from utils import *
def distribution(z,T=1):
    z=np.asarray(z,dtype=np.float64)/T;lp=z-logsumexp(z);return np.exp(lp),lp
def individual(raw,gold,T):
    base={k:raw[k] for k in ['item_id','dataset','model','purpose','rotation']};base.update(gold_index=gold['gold_index'],valid=False,correct=0,
       prediction_index=None,confidence=None,nll=None,brier=None,entropy=None,margin=None,logit_sd=None,latency_ms=raw.get('latency_ms'))
    if raw['status']!='COMPLETED':return base
    out=raw['actual_output']
    if 'logits' not in out:
        valid=out['valid_answer'];j=out['prediction_index'];base.update(valid=valid,prediction_index=j,correct=int(valid and j==gold['gold_index']))
        return base
    z=out['logits'];p,lp=distribution(z,T);k=len(z);g=gold['gold_index'];j=int(np.argmax(p));target=np.eye(k)[g];ordered=np.sort(z)
    base.update(valid=True,correct=int(j==g),prediction_index=j,confidence=float(p[j]),nll=float(-lp[g]),brier=float(((p-target)**2).sum()),
       entropy=float(-(p*lp).sum()),margin=float(ordered[-1]-ordered[-2]),logit_sd=float(np.std(z)),T=T,k=k)
    return base
def ece(rows,bins=15):
    n=len(rows);ans=0.
    for b in range(bins):
        q=[r for r in rows if int(min(r['confidence']*bins,bins-1))==b]
        if q:ans+=len(q)/n*abs(np.mean([r['correct'] for r in q])-np.mean([r['confidence'] for r in q]))
    return float(ans)
def aggregate(rows):
    n=len(rows);valid=sum(r['valid'] for r in rows);prob=all(r.get('nll') is not None for r in rows)
    lat=[r['latency_ms'] for r in rows if r['latency_ms'] is not None]
    result={'planned_n':n,'valid_n':valid,'invalid_or_missing_n':n-valid,'correct':sum(r['correct'] for r in rows),'accuracy':sum(r['correct'] for r in rows)/n,
      'nll':float(np.mean([r['nll'] for r in rows])) if prob else None,'brier':float(np.mean([r['brier'] for r in rows])) if prob else None,
      'ece15':ece(rows) if prob else None,'ece10':ece(rows,10) if prob else None,'mean_latency_ms':float(np.mean(lat)) if lat else None,
      'median_latency_ms':float(np.median(lat)) if lat else None,'p95_latency_ms':float(np.quantile(lat,.95)) if lat else None,
      'entropy':float(np.mean([r['entropy'] for r in rows])) if prob else None,'margin':float(np.mean([r['margin'] for r in rows])) if prob else None,
      'logit_sd':float(np.mean([r['logit_sd'] for r in rows])) if prob else None}
    if prob:
        for threshold in [.7,.9,.95]:
            q=[r for r in rows if r['confidence']>=threshold];key='conf'+str(threshold)
            result[key+'_accepted']=len(q);result[key+'_coverage']=len(q)/n;result[key+'_wrong']=sum(1-r['correct'] for r in q);result[key+'_error_rate']=sum(1-r['correct'] for r in q)/len(q) if q else None
        order=sorted(rows,key=lambda r:(-r['confidence'],digest(r['item_id'])))
        for rate in [.5,.7,.9]:
            q=order[:math.ceil(n*rate)];result['selective_'+str(rate)]=sum(r['correct'] for r in q)/len(q)
    else:
        for threshold in [.7,.9,.95]:
            for suffix in ['_accepted','_coverage','_wrong','_error_rate']:result['conf'+str(threshold)+suffix]=None
        for rate in [.5,.7,.9]:result['selective_'+str(rate)]=None
    return result

def macro_aggregate(dataset_metrics):
    result={};counts=['planned_n','valid_n','invalid_or_missing_n','correct']
    for key in dataset_metrics[0]:
        values=[r[key] for r in dataset_metrics]
        if any(v is None for v in values):result[key]=None
        elif key in counts or key.endswith('_accepted') or key.endswith('_wrong'):result[key]=sum(values)
        else:result[key]=float(np.mean(values))
    return result
def self_test():
    rows=[]
    for k in [2,3,4,5]:
        raw={'item_id':str(k),'dataset':'synthetic','model':'test','purpose':'test','rotation':0,'status':'COMPLETED','actual_output':{'logits':[0]*k}}
        r=individual(raw,{'gold_index':0},1);assert abs(r['nll']-math.log(k))<1e-12 and abs(r['brier']-(1-1/k))<1e-12
        z=np.array([3.,1.,-2.]);p,lp=distribution(z,.6);perm=[2,0,1];q,lq=distribution(z[perm],.6);assert np.allclose(p[perm],q)
        assert np.argmax(distribution(z,100)[0])==np.argmax(z)
        rows.append(r)
    assert ece([{'confidence':1.,'correct':1}],15)==0 and ece([{'confidence':1.,'correct':0}],15)==1
    p,lp=distribution([10000,-10000]);assert np.isfinite(-lp[1]) and -lp[1]==20000
    generated=[aggregate([{'valid':True,'correct':1,'nll':None,'brier':None,'latency_ms':1.},{'valid':False,'correct':0,'nll':None,'brier':None,'latency_ms':1.}]) for _ in range(3)]
    macro=macro_aggregate(generated);assert macro['planned_n']==6 and macro['valid_n']==3 and macro['accuracy']==.5 and macro['nll'] is None and macro['conf0.9_accepted'] is None
    save(ROOT/'audit/scoring_tests.json',{'time':now(),'status':'COMPLETED','cases':['uniform k2..5 analytic','linked permutation','positive T argmax','ECE right-edge','stable large-logit NLL','generated baseline missing probabilities stay undefined; invalid denominator retained']})
if __name__=='__main__':self_test();print('scoring analytic tests passed')
