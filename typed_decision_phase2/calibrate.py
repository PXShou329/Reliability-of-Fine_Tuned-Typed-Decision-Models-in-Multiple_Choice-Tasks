"""C_T fitting and C_gate policy selection only. Never opens E_new."""
import math,argparse
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from utils import *
def soft(z,T=1):
    a=np.asarray(z,dtype=np.float64)/T;return np.exp(a-logsumexp(a))
def fit(rows,gold):
    z=[np.asarray(r['actual_output']['logits'],dtype=np.float64) for r in rows];y=[gold[r['item_id']]['gold_index'] for r in rows]
    def obj(logT):
        T=math.exp(float(logT[0]));v=0.;g=0.
        for a,i in zip(z,y):
            q=soft(a,T);v+=logsumexp(a/T)-a[i]/T;g+=(a[i]-np.dot(q,a))/T
        return v/len(z),np.array([g/len(z)])
    attempts=[]
    for initial in [.5,1,5]:
        result=minimize(obj,np.array([math.log(initial)]),jac=True,method='L-BFGS-B',bounds=[(math.log(.05),math.log(100))],options={'maxiter':500,'ftol':1e-12,'gtol':1e-9})
        attempts.append({'initial_T':initial,'T':math.exp(float(result.x[0])),'NLL':float(result.fun),'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit)})
    best=min(attempts,key=lambda r:r['NLL']);T=best['T'];assert all(np.argmax(a)==np.argmax(soft(a,T)) for a in z)
    return {'T':T,'n':len(rows),'C_T_ids':[r['item_id'] for r in rows],'objective':'float64 NLL','solver':'L-BFGS-B logT [.05,100] starts [.5,1,5]','attempts':attempts,'boundary':T<.050001 or T>99.999,'argmax_invariant':True}
def complete_rows(model,purpose):
    return [read(p) for p in sorted((ROOT/'raw'/model/purpose).glob('*.json')) if read(p)['status']=='COMPLETED']
def main():
    a=argparse.ArgumentParser();a.add_argument('--policy',action='store_true');args=a.parse_args();cal={}
    gold=read(ROOT/'evaluator_private/C_T.json')
    for model in sorted(p.name for p in (ROOT/'raw').iterdir() if p.is_dir()):
        rs=complete_rows(model,'C_T')
        if not rs:continue
        expected=read(ROOT/'configs/training.json')['runs'].get(model,{}).get('C_T_ids',list(gold))
        assert set(r['item_id'] for r in rs)==set(expected),(model,len(rs),len(expected))
        cal[model]=fit(rs,gold);save(ROOT/'calibration'/(model+'.json'),cal[model])
    save(ROOT/'calibration/calibrators.json',cal)
    if args.policy:
        small='B1_grpo_ce_20261006';expert='clef';G=read(ROOT/'evaluator_private/C_gate.json');sr={r['item_id']:r for r in complete_rows(small,'C_gate')};er={r['item_id']:r for r in complete_rows(expert,'C_gate')}
        assert set(sr)==set(er)==set(G);thresholds=[0,.25,.35,.45,.55,.65,.75,.85,.90,.95,.99,1.]
        exp_acc=float(np.mean([np.argmax(er[i]['actual_output']['logits'])==G[i]['gold_index'] for i in G]));candidates=[]
        for tau in thresholds:
            accepts={i:bool(max(soft(sr[i]['actual_output']['logits'],cal[small]['T']))>=tau) for i in G}
            correct=sum(bool(np.argmax(sr[i]['actual_output']['logits'] if accepts[i] else er[i]['actual_output']['logits'])==G[i]['gold_index']) for i in G)
            n=sum(accepts.values());acc=correct/len(G)
            candidates.append({'tau':tau,'small_n':n,'coverage':n/len(G),'accuracy':acc,'accuracy_minus_expert':acc-exp_acc,'eligible':n>=30 and acc-exp_acc>=-.02-1e-12})
        eligible=[r for r in candidates if r['eligible']];best=max(eligible,key=lambda r:(r['coverage'],r['tau'])) if eligible else None
        policy={'time':now(),'small':small,'expert':expert,'small_T':cal[small]['T'],'policy':'confidence' if best else 'bypass','tau':best['tau'] if best else None,
          'random_small_coverage':best['coverage'] if best else 0.,'C_gate_ids':sorted(G),'n':len(G),'expert_accuracy':float(exp_acc),'candidates':candidates,
          'selection_rule':'max coverage with >=30 auto samples and calibration accuracy loss<=.02; tie conservative tau; else bypass',
          'test_target':{'accuracy_loss_pp':2,'mean_latency_reduction_fraction':.2},'weight_sha256':read(ROOT/'checkpoints'/small/'complete.json')['final_sha256'],
          'shared_calibrator_sha256':file_hash(ROOT/'calibration'/(small+'.json')),'random_rule':'int(SHA256(item_id),16)/2**256 < fixed C_gate small coverage'}
        save(ROOT/'configs/cascade_policy.json',policy);print({'policy':policy['policy'],'tau':policy['tau'],'coverage':policy['random_small_coverage']},flush=True)
    print('calibrators',len(cal),flush=True)
if __name__=='__main__':main()
