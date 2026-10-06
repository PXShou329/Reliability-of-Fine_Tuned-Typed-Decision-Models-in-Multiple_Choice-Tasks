"""Descriptive behavior analysis after all MacroLab actions were frozen and scored."""
import csv,collections,statistics
from utils import *
def main():
    tasks={t['episode_id']:t for t in read(ROOT/'macrolab/private/tasks.json')};rows=list(csv.DictReader((ROOT/'results/macrolab_results.csv').open(encoding='utf-8-sig')));summary={}
    for model in ['B1_grpo_ce_20261006','clef']:
        dev=[read(p) for p in (ROOT/'macrolab/raw/B1_grpo_ce_20261006/development').glob('*.json')] if model.startswith('B1') else []
        counts=collections.Counter();idempotent=0;total=0;changed=0;m1m2=0;rawidentical=0;supportpass=0;changed_vs_M0=0;first_equal=0
        for i in [t['episode_id'] for t in tasks.values() if t['split']!='development']:
            r1=read(ROOT/'macrolab/raw'/model/'M1'/(i+'.json'));r2=read(ROOT/'macrolab/raw'/model/'M2'/(i+'.json'))
            r0=read(ROOT/'macrolab/raw'/model/'M0'/(i+'.json'));changed_vs_M0+=r1['selected_program']!=r0['selected_program'];first_equal+=r1['candidate_programs'][0]==r0['candidate_programs'][0]
            m1m2 += r1['candidate_programs']==r2['candidate_programs'] and r1['selected_program']==r2['selected_program']
            rawidentical += all(a['request']==b['request'] and a['raw_logits']==b['raw_logits'] for a,b in zip(r1['traces'],r2['traces'])) and len(r1['traces'])==len(r2['traces'])
            changed += len({canon(p) for p in r1['candidate_programs']})>1
            for condition in ['M0','M1','M2']:
                r=read(ROOT/'macrolab/raw'/model/condition/(i+'.json'))
                for p in r['candidate_programs']:
                    total+=1;counts[canon(p)]+=1
                    idempotent += len(p)>1 and len(set(p))==1 and p[0] in ['ABS','FILTER_EVEN','FILTER_POSITIVE','SORT_ASC','DEDUP_STABLE']
                supportpass += any(t.get('support_result',{}).get('passed')==5 for t in r['traces'])
        summary[model]={'M1_changed_candidates_episodes':changed,'M1_vs_M0_changed_final_program_episodes':changed_vs_M0,'M1_M0_same_first_candidate_episodes':first_equal,'M1_M2_same_programs_episodes':m1m2,'M1_M2_bit_identical_requests_logits_episodes':rawidentical,
             'formal_support_pass_episodes':supportpass,'candidate_n':total,'repeated_idempotent_candidates':idempotent,'most_common_candidates':counts.most_common(6),
             'development_episodes':len(dev),'development_model_calls':sum(r['model_calls'] for r in dev),'development_support_pass_episode_n':sum(any(t.get('support_result',{}).get('passed')==5 for t in r['traces']) for r in dev)}
    enumeration=[r for r in rows if r['model']=='search' and r['condition']=='enumeration'];falsepass=[]
    for r in enumeration:
        raw=read(ROOT/r['raw_path'])
        if raw['traces'][-1]['support_result']['passed']==5 and r['task_success']=='0':falsepass.append({'episode_id':r['episode_id'],'raw_path':r['raw_path'],'hidden_correct_n':r['hidden_correct_n']})
    save(ROOT/'reports/macrolab_interpretation.json',{'time':now(),'status':'COMPLETED','models':summary,'enumeration_support_pass_hidden_failure':falsepass,'description':'post-evaluation observation; no new model condition, method or exclusion'})
    print(summary);print('support-pass hidden-failure',falsepass)
if __name__=='__main__':main()
