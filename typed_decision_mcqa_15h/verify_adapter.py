"""Train-only check against official inference APIs, preserving raw probabilities."""
import argparse,torch
from adapters import load_adapter
from utils import ROOT,read,save,event

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);a=ap.parse_args()
    adapter=load_adapter(a.model,ROOT/'models'/('laya' if a.model=='laya_base' else a.model),read(ROOT/'configs/protocol.json'))
    checks=[]
    for d in ['arc','csqa','obqa']:
        r=read(ROOT/'data/pilot'/(d+'.json'))[0];p=r['public'];ours=adapter.predict(p)
        if a.model=='laya_base': official=adapter.agent.predict(p['state'],p['questions'])['answers']['answer']['probabilities']
        else:
            official=adapter.mod.systemone(adapter.model,adapter.processor,{'model':'clef-flash',**p},max_length=adapter.max_length)['answers']['answer']['probabilities']
        keys=list(p['questions']['answer']['criteria']);error=max(abs(ours['probabilities'][i]-official[k]) for i,k in enumerate(keys))
        if error>0.000051: raise RuntimeError('Adapter differs from official output beyond 4-decimal rounding')
        checks.append({'item_id':r['item_id'],'public':p,'adapter_response':ours,'rounded_official_probabilities':official,
                       'max_absolute_difference_from_rounded_official':error,'passed':True})
    save(ROOT/'adapter_verification'/(a.model+'.json'),{'model':a.model,'train_only':True,'checks':checks,'adapter':adapter.metadata})
    event('adapter_verified',model=a.model,n=len(checks));print(checks,flush=True)

if __name__=='__main__': main()
