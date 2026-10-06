"""Post-freeze evaluator case selection: fixed random sample plus labelled purposeful cases."""
import csv,random,collections
from utils import *
def main():
    assert read(ROOT/'results/analysis_complete.json')['status']=='COMPLETED'
    model_names=['base','B1_grpo_ce_20261006','B1_ce_20261006','kev','clef','generative']
    public={r['item_id']:r for r in read(ROOT/'data_public/E_new.json')};gold=read(ROOT/'evaluator_private/E_new.json')
    cases=list(csv.DictReader((ROOT/'results/case_level.csv').open(encoding='utf-8-sig')))
    data={(r['model'],r['view'],r['item_id']):r for r in cases if r['purpose']=='E_new'}
    choices={};rng=random.Random(260623)
    for ds in ['arc','csqa','obqa']:
        for i in rng.sample(sorted(i for i,r in public.items() if r['dataset']==ds),8):choices.setdefault(i,[]).append('fixed_random_seed260623')
    def pick(ids,label,n=6):
        for i in sorted(ids,key=digest)[:n]:choices.setdefault(i,[]).append(label)
    ids=list(public)
    B=lambda i:data['base','Raw',i]['correct']=='1'
    F=lambda i:data['B1_grpo_ce_20261006','Raw',i]['correct']=='1'
    pick([i for i in ids if F(i) and not B(i)],'purposeful_R0_gain')
    pick([i for i in ids if B(i) and not F(i)],'purposeful_R0_regression')
    pick([i for i in ids if not F(i) and float(data['B1_grpo_ce_20261006','Raw',i]['confidence'] or 0)>=.9],'purposeful_R0_raw_high_confidence_error')
    pick([i for i in ids if data.get(('clef','Raw',i),{}).get('correct')=='0'],'purposeful_expert_error')
    rows=[]
    for i,reasons in choices.items():
        outputs={}
        for model in model_names:
            p=ROOT/'raw'/model/'E_new'/(digest([i,0])+'.json')
            if not p.exists():continue
            r=read(p);outputs[model]={'raw_path':str(p.relative_to(ROOT)),'status':r['status'],'actual_output':r.get('actual_output'),'views':{v:data.get((model,v,i)) for v in ['Raw','Native_Historical','Shared_fit','Generated'] if (model,v,i) in data}}
        rows.append({'item_id':i,'dataset':public[i]['dataset'],'selection':reasons,'public':public[i]['public'],'gold':gold[i],'outputs':outputs,'AI_reviewed':False,'human_verified':False})
    save(ROOT/'reports/case_review_inputs.json',{'time':now(),'selection_seed':260623,'random_n':24,'cases':rows,'note':'post-score analysis only; no methods altered; dataset gold is not external truth certification'})
    print('case review packet',len(rows),'cases',flush=True)
if __name__=='__main__':main()
