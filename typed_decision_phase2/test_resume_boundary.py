"""Fault injection unit test, never an experiment/model output."""
import tempfile
from utils import *
import inference
def main():
    real_root=inference.ROOT;oldargv=sys.argv;cases=[]
    try:
        for failed in [False,True]:
            with tempfile.TemporaryDirectory(dir=ROOT/'cache',prefix='UNIT_resume_') as tmp:
                testroot=Path(tmp);inference.ROOT=testroot
                save(testroot/'configs/training.json',{'runs':{}});save(testroot/'audit/data_isolation.json',{'status':'COMPLETED'})
                save(testroot/'manifests/permutation_manifest.json',{'old':[]})
                public={'state':'UNIT_TEST_NOT_EXPERIMENT','questions':{'answer':{'type':'choice','instructions':'unit','criteria':{'A':'unit','B':'unit2'}}}}
                row={'item_id':'UNIT','dataset':'unit','public':public}
                paths=[]
                for purpose in ['C_T','E_old']:
                    save(testroot/'data_public'/(purpose+'.json'),[row]);p=inference.raw_path('base',purpose,'UNIT',0)
                    save(p,{'public_hash':digest(public),'model':'base','status':'FAILED' if failed and purpose=='E_old' else 'COMPLETED','unit_not_model_output':True});paths.append(p)
                before={str(p):file_hash(p) for p in paths}
                inference.load=lambda name:(_ for _ in ()).throw(AssertionError('resume must not load or call model'))
                sys.argv=['inference.py','--model','base','--stage','pre'];inference.main()
                result=read(testroot/'results/inference_base_pre.json')
                assert result['status']==('FAILED' if failed else 'COMPLETED') and result['new_calls']==0 and result['failed']==int(failed)
                assert all(file_hash(p)==before[str(p)] for p in paths)
                cases.append({'saved_failure':failed,'new_model_calls':0,'raw_bytes_unchanged':True,'summary_status':result['status']})
    finally:inference.ROOT=real_root;sys.argv=oldargv
    save(ROOT/'audit/resume_boundary_tests.json',{'time':now(),'status':'COMPLETED','unit_only':True,'cases':cases})
    print('resume fault-injection checks passed')
if __name__=='__main__':main()
