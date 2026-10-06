"""Evaluator only: never imported into the model adapter or training loop."""
import math
import numpy as np

ECE_BINS=15
NLL_FLOOR=1e-12

def validate_probs(p,k):
    a=np.asarray(p,dtype=np.float64)
    if a.shape!=(k,) or not np.isfinite(a).all() or (a<0).any() or (a>1).any():
        raise ValueError('invalid probability distribution')
    if abs(float(a.sum())-1)>1e-5: raise ValueError('probabilities do not sum to one')
    return a

def score(p,gold):
    p=validate_probs(p,len(p))
    if not 0<=gold<len(p): raise ValueError('gold index outside option range')
    pred=int(np.argmax(p));y=np.zeros(len(p));y[gold]=1
    return {'pred_index':pred,'correct':int(pred==gold),'confidence':float(p[pred]),
            'gold_probability':float(p[gold]),'brier':float(np.sum((p-y)**2)),
            'nll':-math.log(max(float(p[gold]),NLL_FLOOR)), 'nll_clipped':bool(p[gold]<NLL_FLOOR)}

def aggregate(rows):
    if not rows: return {'n':0}
    n=len(rows);bins=[]
    for i in range(ECE_BINS):
        selected=[r for r in rows if min(int(r['confidence']*ECE_BINS),ECE_BINS-1)==i]
        bins.append({'bin':i,'lower':i/ECE_BINS,'upper':(i+1)/ECE_BINS,'n':len(selected),
                     'accuracy':np.mean([r['correct'] for r in selected]).item() if selected else None,
                     'confidence':np.mean([r['confidence'] for r in selected]).item() if selected else None})
    ece=sum(b['n']/n*abs(b['accuracy']-b['confidence']) for b in bins if b['n'])
    latency=np.asarray([r['latency_seconds'] for r in rows])
    return {'n':n,'accuracy':float(np.mean([r['correct'] for r in rows])),
            'brier':float(np.mean([r['brier'] for r in rows])),
            'nll':float(np.mean([r['nll'] for r in rows])),'ece':float(ece),
            'nll_clipped':sum(r['nll_clipped'] for r in rows),
            'latency_median_seconds':float(np.median(latency)),
            'latency_p90_seconds':float(np.percentile(latency,90)),
            'latency_p95_seconds':float(np.percentile(latency,95)),
            'high_confidence_wrong_n':sum(r['confidence']>=.9 and not r['correct'] for r in rows),
            'high_confidence_n':sum(r['confidence']>=.9 for r in rows),'bins':bins}

def self_test():
    # Analytical values, variable choice counts, tie handling, boundary bins, rejection.
    s=score([.25]*4,2)
    assert s['correct']==0 and abs(s['brier']-.75)<1e-12 and abs(s['nll']-math.log(4))<1e-12
    s=score([0,1,0],1);s['latency_seconds']=0
    a=aggregate([s]);assert a['accuracy']==1 and a['ece']==0 and a['brier']==0 and a['bins'][-1]['n']==1
    assert score([0,1],0)['nll_clipped']
    for p,k in [([.2,.2],2),([float('nan'),1],2),([1.1,-.1],2),([1],2)]:
        try: validate_probs(p,k)
        except ValueError: pass
        else: raise AssertionError('invalid probabilities accepted')
    return {'analytic_cases':'passed','invalid_distribution_cases':'passed'}

if __name__=='__main__': print(self_test())
