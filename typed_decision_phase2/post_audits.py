"""Independent scalar B2 and private-oracle MacroLab verification of saved outputs."""
import argparse,csv,math,collections,importlib.util
from utils import *
def rows(name):return list(csv.DictReader((ROOT/'results'/name).open(encoding='utf-8-sig')))
def b2():
    table={(r['purpose'],r['model'],r['item_id']):r for r in rows('order_robustness_cases.csv')};cal=read(ROOT/'calibration/calibrators.json');checks=0;maxdiff=0.;requests=0
    for purpose in ['E_old','E_new']:
        public={r['item_id']:r['public'] for r in read(ROOT/'data_public'/(purpose+'.json'))};gold=read(ROOT/'evaluator_private'/(purpose+'.json'))
        for model in ['base','B1_grpo_ce_20261006','B1_ce_20261006','kev','clef']:
            groups=collections.defaultdict(list)
            for path in (ROOT/'raw'/model/('B2_'+purpose)).glob('*.json'):
                r=read(path);groups[r['item_id']].append(r);requests+=1
                assert set(r['public'])=={'state','questions'} and r['public_hash']==digest(r['public'])
                original=public[r['item_id']];mapping=r['mapping_to_original'];k=len(mapping)
                assert sorted(mapping)==list(range(k)) and r['public']['state']==original['state']
                assert mapping==[(j+r['rotation'])%k for j in range(k)]
                original_options=list(original['questions']['answer']['criteria'].values())
                assert list(r['public']['questions']['answer']['criteria'].values())==[original_options[j] for j in mapping]
            for i,rs in groups.items():
                rs.sort(key=lambda r:r['rotation']);k=len(rs);assert [r['rotation'] for r in rs]==list(range(k))
                vectors=[];choices=[]
                for r in rs:
                    z=[v/cal[model]['T'] for v in r['actual_output']['logits']];m=max(z);e=[math.exp(v-m) for v in z];p=[v/sum(e) for v in e];aligned=[0.]*k
                    for j,old in enumerate(r['mapping_to_original']):aligned[old]=p[j]
                    vectors.append(aligned);choices.append(max(range(k),key=lambda j:aligned[j]))
                correct=[int(j==gold[i]['gold_index']) for j in choices];avg=[sum(p[j] for p in vectors)/k for j in range(k)]
                got={'permutation_accuracy':sum(correct)/k,'original_correct':correct[0],'all_rotation_correct':int(all(correct)),
                     'any_answer_flip':int(len(set(choices))>1),'ensemble_correct':int(max(range(k),key=lambda j:avg[j])==gold[i]['gold_index']),
                     'mean_TV_to_original':sum(sum(abs(p[j]-vectors[0][j]) for j in range(k))/2 for p in vectors[1:])/(k-1),
                     'max_probability_drift':max(abs(p[j]-vectors[0][j]) for p in vectors for j in range(k))}
                expected=table[purpose,model,i]
                for key,v in got.items():
                    d=abs(v-float(expected[key]));maxdiff=max(maxdiff,d);assert d<1e-10,(purpose,model,i,key,d)
                assert choices==read_json_string(expected['original_choices']);checks+=1
    save(ROOT/'audit/b2_independent_recompute.json',{'time':now(),'status':'COMPLETED','original_question_model_pairs':checks,'actual_rotation_requests':requests,'max_absolute_difference':maxdiff,'mapping_and_option_text_checked':True,'no_primary_scorer_import':True})
def read_json_string(s):return json.loads(s)
def macro():
    assert read(ROOT/'macrolab/evaluation_complete.json')['status']=='COMPLETED'
    spec=importlib.util.spec_from_file_location('private_oracle_audit',ROOT/'macrolab/oracle.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    tasks={t['episode_id']:t for t in read(ROOT/'macrolab/private/tasks.json')};lib=read(ROOT/'macrolab/frozen_library.json');lock=read(ROOT/'macrolab/protocol.lock.json');checks=0;calls=0
    for row in rows('macrolab_results.csv'):
        raw=read(ROOT/row['raw_path']);task=tasks[row['episode_id']];program=raw['selected_program'];assert raw['hidden_access'] is False
        correct=0 if program is None else sum(oracle.target(program,x)==y for x,y in zip(task['hidden_inputs'],task['hidden_outputs']))
        assert correct==int(row['hidden_correct_n']) and int(correct==10)==int(row['task_success'])
        assert raw['verifier_calls']<=4 and raw['model_calls']<=20
        if raw['model']!='search':
            assert raw['all_actions_actual_model_outputs'] and raw['library_sha256']==lock['library_sha256']
            current=[];complete=[];feedback=[];selected=None
            for trace in raw['traces']:
                assert current==trace['before_program'] and trace['truncated'] is False
                req=trace['request'];assert set(req)=={'state','questions'};q=req['questions']['answer'];assert set(q)=={'type','instructions','criteria'}
                assert trace['selected_index']==max(range(len(trace['raw_logits'])),key=lambda j:trace['raw_logits'][j])
                assert trace['typed_output']['choice']==chr(65+trace['selected_index'])
                op=trace['action'];param=trace['parameter']
                if op=='append':current.append(param)
                elif op=='library':assert raw['condition']=='M2';current.extend(lib['programs'][param])
                elif op=='finish':
                    complete.append(current.copy());f=trace['support_result'];pred=[oracle.target(current,x) for x in task['support_inputs']];truth=[oracle.target(task['private_target'],x) for x in task['support_inputs']]
                    assert f['passed']==sum(a==b for a,b in zip(pred,truth));assert trace['support_visible']==(raw['condition']!='M0')
                    feedback.append(f);current=[]
                elif op=='keep':assert feedback[param]['passed']==5;selected=complete[param];assert trace['episode_finished_by_model']
                else:raise AssertionError(op)
                calls+=1
            assert raw['candidate_programs']==complete
            assert raw['selected_program']==(selected if selected is not None else (complete[-1] if complete else None))
        checks+=1
    save(ROOT/'audit/macrolab_independent_recompute.json',{'time':now(),'status':'COMPLETED','condition_episodes':checks,'actual_typed_calls_checked':calls,'support_and_hidden_outputs_recomputed_by_private_oracle':True,'candidate_provenance_checked':True,'frozen_library_n':len(lib['programs']),'human_reviewed':0})
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--macro',action='store_true');args=a.parse_args();macro() if args.macro else b2();print('independent post audit passed', 'macro' if args.macro else 'B2')
