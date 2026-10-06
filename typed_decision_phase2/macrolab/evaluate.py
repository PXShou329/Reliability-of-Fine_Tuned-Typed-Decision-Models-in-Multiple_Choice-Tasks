"""Hidden tests are read only after frozen episodes produced; never passed to model."""
import sys,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils import *
from interpreter import run
def main():
    tasks={t['episode_id']:t for t in read(ROOT/'macrolab/private/tasks.json') if t['split']!='development'};rows=[]
    for p in (ROOT/'macrolab/raw').glob('*/*/*.json'):
        r=read(p);i=r['episode_id']
        if i not in tasks:continue
        t=tasks[i];prog=r['selected_program'];correct=0 if prog is None else sum(run(prog,x)==y for x,y in zip(t['hidden_inputs'],t['hidden_outputs']))
        rows.append({'episode_id':i,'split':t['split'],'family_id':t['family_id'],'model':r['model'],'condition':r['condition'],'status':r['status'],
            'program':canon(prog),'task_success':int(correct==10),'hidden_correct_n':correct,'hidden_n':10,'model_calls':r['model_calls'],'verifier_calls':r['verifier_calls'],
            'primitive_steps':len(prog) if prog is not None else 0,'library_action_n':r['library_action_n'],'model_latency_ms':r['latency_ms'],
            'episode_wall_ms':r.get('episode_wall_ms',r['latency_ms']),'compatible_programs_after_demos':t['compatible_programs_after_demos'],
            'raw_path':str(p.relative_to(ROOT)),'human_verified':False})
    expected=[(m,c) for m in read(ROOT/'macrolab/protocol.lock.json')['formal_models'] for c in ['M0','M1','M2']]+[('search','random'),('search','enumeration')]
    existing={(r['model'],r['condition'],r['episode_id']) for r in rows}
    for model,condition in expected:
        for i,t in tasks.items():
            if (model,condition,i) not in existing:
                rows.append({'episode_id':i,'split':t['split'],'family_id':t['family_id'],'model':model,'condition':condition,'status':'MISSING',
                    'program':'null','task_success':0,'hidden_correct_n':0,'hidden_n':10,'model_calls':0,'verifier_calls':0,'primitive_steps':0,'library_action_n':0,
                    'model_latency_ms':0.,'episode_wall_ms':0.,'compatible_programs_after_demos':t['compatible_programs_after_demos'],'raw_path':'','human_verified':False})
    metrics=[]
    for model,condition in sorted({(r['model'],r['condition']) for r in rows}):
        for split in ['reuse','novel_composition']:
            rs=[r for r in rows if r['model']==model and r['condition']==condition and r['split']==split];assert len(rs)==20,(model,condition,split,len(rs))
            valid=[r for r in rs if r['status']=='COMPLETED']
            metrics.append({'model':model,'condition':condition,'split':split,'n':len(rs),'completed_n':len(valid),'missing_or_failed_n':len(rs)-len(valid),'success_n':sum(r['task_success'] for r in rs),
                'task_success':float(np.mean([r['task_success'] for r in rs])),'per_input_accuracy':sum(r['hidden_correct_n'] for r in rs)/(10*len(rs)),
                'mean_model_calls':float(np.mean([r['model_calls'] for r in rs])),'mean_verifier_calls':float(np.mean([r['verifier_calls'] for r in rs])),
                'mean_primitive_steps':float(np.mean([r['primitive_steps'] for r in rs])),'library_episode_use_rate':float(np.mean([r['library_action_n']>0 for r in rs])),
                'mean_model_latency_ms':float(np.mean([r['model_latency_ms'] for r in valid])) if valid else None,'mean_episode_wall_ms':float(np.mean([r['episode_wall_ms'] for r in valid])) if valid else None,
                'ambiguous_demonstration_n':sum(r['compatible_programs_after_demos']>1 for r in rs)})
    contrasts=[]
    for model in sorted({r['model'] for r in rows if r['model']!='search'}):
        for split in ['reuse','novel_composition']:
            for A,B in [('M1','M0'),('M2','M1')]:
                aa={r['episode_id']:r for r in rows if r['model']==model and r['condition']==A and r['split']==split};bb={r['episode_id']:r for r in rows if r['model']==model and r['condition']==B and r['split']==split};assert set(aa)==set(bb)
                clusters=collections.defaultdict(list)
                for i in aa:clusters[aa[i]['family_id']].append(i)
                ss=np.array([sum(aa[i]['task_success']-bb[i]['task_success'] for i in ids) for ids in clusters.values()]);nn=np.array([len(ids) for ids in clusters.values()]);rng=np.random.default_rng(260622)
                w=rng.multinomial(len(ss),np.ones(len(ss))/len(ss),size=10000);bs=w@ss/(w@nn);lo,hi=np.quantile(bs,[.025,.975])
                contrasts.append({'model':model,'split':split,'A':A,'B':B,'difference':float(sum(ss)/sum(nn)),'ci95_low':float(lo),'ci95_high':float(hi),
                    'family_clusters':len(ss),'n':len(aa),'replicates':10000,'seed':260622,'exploratory':True})
    csv_write(ROOT/'results/macrolab_results.csv',rows);csv_write(ROOT/'results/macrolab_metrics.csv',metrics);csv_write(ROOT/'results/macrolab_paired.csv',contrasts)
    save(ROOT/'macrolab/evaluation_complete.json',{'time':now(),'status':'COMPLETED' if all(r['status']=='COMPLETED' for r in rows) else 'FAILED','formal_condition_episodes':len(rows),'completed_condition_episodes':sum(r['status']=='COMPLETED' for r in rows),'original_episodes':40,'model_calls':sum(r['model_calls'] for r in rows),
        'hidden_never_returned_to_model':True,'human_verified_cases':0})
    print('MacroLab evaluator scored',len(rows),'condition episodes',flush=True)
if __name__=='__main__':main()
