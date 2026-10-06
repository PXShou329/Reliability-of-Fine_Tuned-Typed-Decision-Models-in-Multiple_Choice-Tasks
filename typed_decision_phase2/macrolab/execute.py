"""Model development/formal episodes. Hidden scoring intentionally separate."""
import sys,argparse,traceback,itertools,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils import *
from interpreter import PRIMITIVES,run
from verifier import SupportVerifier
from episode import episode
from model_adapter import MacroModel
def public(t):return {k:t[k] for k in ['episode_id','split','demos']}
def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['development','formal','search'],required=True);a.add_argument('--model',default='B1_grpo_ce_20261006');args=a.parse_args()
    assert budget_ok(3600),'MacroLab protects experiment/report reserve'
    tasks=read(ROOT/'macrolab/private/tasks.json');proto=read(ROOT/'macrolab/macrolab_protocol.json')
    if args.stage=='development':
        assert (ROOT/'results/cascade_execution.json').exists(),'A/B/C1 must be deliverable first'
        model=MacroModel(args.model);library=[];gate_start=time.time();dev=[t for t in tasks if t['split']=='development']
        for j,t in enumerate(dev):
            dest=ROOT/'macrolab/raw'/args.model/'development'/(t['episode_id']+'.json')
            if dest.exists():result=read(dest)
            else:
                v=SupportVerifier(t['private_target'],t['support_inputs']);result=episode(model,public(t),'M1',v,[]);result['status']='COMPLETED';save(dest,result)
            for action in result['traces']:
                if action['action']=='finish' and action['support_result']['passed']==5:
                    p=result['candidate_programs'][sum(x['action']=='finish' for x in result['traces'][:action['call']])-1]
                    if p not in library:library.append(p)
                    break  # Predeclared first support-passing candidate per development episode, <=12 tools.
            if j==4:
                assert time.time()-gate_start<=5400 and len(result['traces'])<=20 and result['selected_program'] is not None
                save(ROOT/'macrolab/development_gate.json',{'time':now(),'status':'COMPLETED','actual_model':args.model,'complete_development_episodes':5,
                     'no_truncation':True,'actual_typed_outputs':True,'seconds':time.time()-gate_start,'wrong_answers_do_not_block':True})
        assert len(library)<=12
        save(ROOT/'macrolab/frozen_library.json',{'time':now(),'model':args.model,'programs':library,'provenance':'first actual development FINISH candidate with5/5 support per episode; deduplicated; at most12 tools; no hidden scores used'})
        # Split custodian checks library-family separation without any formal prediction/score.
        rng=random.Random(read(ROOT/'macrolab/private/signature_seed.json')['seed']);probes=[[rng.randint(-9,9) for _ in range(rng.randint(3,8))] for _ in range(48)]+[[0,-1,2,2,-4,7],[8,4,4,0,-3]]
        actual_library_families={digest([run(p,x) for x in probes]) for p in library}
        novel_families={t['family_id'] for t in tasks if t['split']=='novel_composition'}
        collision=actual_library_families&novel_families
        save(ROOT/'macrolab/split_library_check.json',{'time':now(),'status':'COMPLETED' if not collision else 'BLOCKED','collision_families':sorted(collision),'finite_signature_only':True,'no_formal_outputs_used':True})
        assert not collision,'Novel task family overlaps actual library; data gate requires pre-formal repair'
        proto.update(library_admission='first actual support-passing candidate per development episode; deduplicate; <=12 tools; no hidden ranking',input_budget={'Laya_max_len':1024,'Laya_head_max_len':256,'Clef_max_length':16384},
             observation='two demos, current program, last support result/counterexample plus all previously passing candidates; no M0 feedback',
             reference_model='clef',reference_library='same frozen Laya-generated library (transfer reference, not independently expert-generated tools)')
        proto['enumeration_order']='increasing length then primitive Cartesian order; prune using only2 public demonstrations, first4 compatible candidates get support tests; enumeration bound4680; real preprocessing cost recorded'
        save(ROOT/'macrolab/macrolab_protocol.json',proto)
        files=['macrolab/interpreter.py','macrolab/oracle.py','macrolab/generate.py','macrolab/verifier.py','macrolab/episode.py','macrolab/model_adapter.py','macrolab/execute.py','macrolab/evaluate.py']
        resource=read(ROOT/'macrolab/resource_gate.json')
        save(ROOT/'macrolab/protocol.lock.json',{'time':now(),'state':'FROZEN','formal_models':resource['formal_models'],'conditions':['M0','M1','M2'], 'protocol_sha256':file_hash(ROOT/'macrolab/macrolab_protocol.json'),
            'main_protocol_sha256':file_hash(ROOT/'manifests/protocol.lock.json'),'resource_gate_sha256':file_hash(ROOT/'macrolab/resource_gate.json'),
            'tasks_sha256':file_hash(ROOT/'macrolab/private/tasks.json'),'library_sha256':file_hash(ROOT/'macrolab/frozen_library.json'),'code_sha256':{f:file_hash(ROOT/f) for f in files}})
        print('development done; library programs',len(library),flush=True);return
    lock=read(ROOT/'macrolab/protocol.lock.json');assert lock['state']=='FROZEN'
    for f,h in lock['code_sha256'].items():assert file_hash(ROOT/f)==h,f
    assert file_hash(ROOT/'macrolab/frozen_library.json')==lock['library_sha256']
    assert file_hash(ROOT/'macrolab/private/tasks.json')==lock['tasks_sha256']
    assert file_hash(ROOT/'macrolab/macrolab_protocol.json')==lock['protocol_sha256']
    assert file_hash(ROOT/'manifests/protocol.lock.json')==lock['main_protocol_sha256']
    assert file_hash(ROOT/'macrolab/resource_gate.json')==lock['resource_gate_sha256']
    library=read(ROOT/'macrolab/frozen_library.json')['programs'];formal=[t for t in tasks if t['split']!='development']
    if args.stage=='formal':
        model=MacroModel(args.model)
        for t in formal:
            conditions=['M0','M1','M2'];offset=int(digest(t['episode_id']),16)%3;conditions=conditions[offset:]+conditions[:offset]
            for c in conditions:
                dest=ROOT/'macrolab/raw'/args.model/c/(t['episode_id']+'.json')
                if dest.exists():continue
                v=SupportVerifier(t['private_target'],t['support_inputs']);result=episode(model,public(t),c,v,library if c=='M2' else [])
                result.update(status='COMPLETED',library_sha256=lock['library_sha256']);save(dest,result)
            print(args.model,t['episode_id'],'three paired conditions done',flush=True)
    else:
        programs=[list(p) for n in range(1,5) for p in itertools.product(PRIMITIVES,repeat=n)]
        for t in formal:
            for c in ['random','enumeration']:
                begin=time.perf_counter()
                dest=ROOT/'macrolab/raw/search'/c/(t['episode_id']+'.json')
                if dest.exists():continue
                v=SupportVerifier(t['private_target'],t['support_inputs']);rng=random.Random(int(digest(['260621',t['episode_id']]),16));traces=[]
                choices=rng.sample(programs,4) if c=='random' else [p for p in programs if all(run(p,d['input'])==d['output'] for d in t['demos'])][:4]
                assert choices,'public-demo compatible target must exist'
                for candidate in choices:
                    feedback=v.verify(candidate);traces.append({'program':candidate,'support_result':feedback})
                    if feedback['passed']==5:break
                save(dest,{'time':now(),'episode_id':t['episode_id'],'split':t['split'],'condition':c,'model':'search','selected_program':traces[-1]['program'],
                    'traces':traces,'model_calls':0,'verifier_calls':v.calls,'library_action_n':0,'latency_ms':(time.perf_counter()-begin)*1000,'status':'COMPLETED','hidden_access':False,
                    'policy':'first support-passing candidate else fourth; no hidden selection'})
        print('nonmodel paired search done',flush=True)
if __name__=='__main__':main()
