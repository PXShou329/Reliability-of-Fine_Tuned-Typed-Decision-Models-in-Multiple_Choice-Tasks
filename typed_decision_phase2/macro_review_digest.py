"""Fixed random and transparent purposeful trace selection after frozen evaluation."""
import csv,random
from utils import *
def main():
    rows=list(csv.DictReader((ROOT/'results/macrolab_results.csv').open(encoding='utf-8-sig')));chosen=[]
    for model in ['B1_grpo_ce_20261006','clef']:
        candidates=sorted([r for r in rows if r['model']==model and r['condition']=='M1' and r['split']=='reuse'],key=lambda r:r['episode_id'])
        chosen.append(('fixed_random_seed260624_'+model,random.Random(260624).choice(candidates)))
    for label,predicate in [('purposeful_enumeration_success',lambda r:r['model']=='search' and r['condition']=='enumeration' and r['task_success']=='1'),
                            ('purposeful_enumeration_hidden_failure',lambda r:r['model']=='search' and r['condition']=='enumeration' and r['task_success']=='0')]:
        candidates=sorted([r for r in rows if predicate(r)],key=lambda r:digest(r['episode_id']));chosen.append((label,candidates[0]))
    falsepass=[r for r in rows if r['model']=='search' and r['condition']=='enumeration' and r['task_success']=='0' and read(ROOT/r['raw_path'])['traces'][-1]['support_result']['passed']==5]
    if falsepass:chosen.append(('purposeful_support_pass_hidden_fail',sorted(falsepass,key=lambda r:digest(r['episode_id']))[0]))
    lines=[];packet=[];tasks={t['episode_id']:t for t in read(ROOT/'macrolab/private/tasks.json')}
    for label,row in chosen:
        raw=read(ROOT/row['raw_path']);t=tasks[row['episode_id']];lines += [label+' '+row['model']+'/'+row['condition']+'/'+row['episode_id'],'demos='+canon(t['demos']),'selected='+canon(raw['selected_program'])+' hidden correct='+row['hidden_correct_n']+'/10']
        for tr in raw['traces']:
            if raw['model']=='search':lines.append('candidate='+canon(tr['program'])+' support='+canon(tr['support_result']))
            else:lines.append('call'+str(tr['call'])+' before='+canon(tr['before_program'])+' action='+tr['action']+' param='+str(tr['parameter'])+' choice='+tr['typed_output']['choice']+' support='+canon(tr.get('support_result')))
        lines.append('raw='+row['raw_path']);lines.append('')
        packet.append({'selection':label,'result':row,'raw_path':row['raw_path'],'AI_reviewed':False,'human_reviewed':False})
    write(ROOT/'reports/macrolab_trace_digest.txt','\n'.join(lines)+'\n');save(ROOT/'reports/macrolab_trace_review_inputs.json',{'time':now(),'selection_seed':260624,'cases':packet});print(len(chosen),'complete traces selected')
if __name__=='__main__':main()
