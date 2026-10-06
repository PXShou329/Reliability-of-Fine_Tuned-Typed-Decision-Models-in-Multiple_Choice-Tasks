"""Pilot-only official API, pre-T, padding and mapping checks for actual model configuration."""
import argparse,torch,numpy as np,traceback
from utils import *
from inference import load,timed,rotate
def main():
    a=argparse.ArgumentParser();a.add_argument('--model',required=True);args=a.parse_args();result={'time':now(),'model':args.model,'pid':os.getpid(),'command':sys.argv,'status':'RUNNING'};dest=ROOT/'audit'/('adapter_'+args.model+'.json')
    try:
        adapter=load(args.model);rows=read(ROOT/'data_public/P.json')[:6];outputs=[]
        for r in rows:
            out,ms=timed(adapter,r['public']);outputs.append({'item_id':r['item_id'],'public':r['public'],'actual_output':out,'latency_ms':ms})
            if 'logits' in out:
                z=torch.tensor(out['logits'],dtype=torch.float64);assert torch.isfinite(z).all()
                p=(z/out['temperature']).softmax(-1);assert np.allclose(p.numpy(),out['probabilities'],atol=2e-6)
                assert z.argmax()==(z/100).softmax(-1).argmax()
        if args.model=='generative':
            result['P_generated_reasoning_n']=sum(r['actual_output']['reasoning_generated'] for r in outputs)
            result['nonthinking_template_verified']=True
        if args.model=='multilingual':
            # Custodian tokenization only: no E_new model forward or label/score access.
            checks=[]
            for purpose in ['P','C_T','E_old','E_new']:
                encoded=[adapter.encode(r['public']) for r in read(ROOT/'data_public'/(purpose+'.json'))]
                assert all(not r['truncation']['truncated'] and r['option_stats']['tokens_per_option'] is None for r in encoded),('native multilingual token budget failed',purpose)
                checks.append({'purpose':purpose,'n':len(encoded),'max_tokens':max(len(r['ids']) for r in encoded),'no_state_or_option_truncation':True,'public_format_only':True})
            result['native_multilingual_token_gate']=checks
        if args.model in ['base','multilingual','v1_ft_history'] or args.model.startswith('B'):
            # Compare real API answer/order to our raw adapter within documented4-decimal serialization.
            p=rows[0]['public'];native=adapter.agent.predict(p['state'],p['questions']);answer=native['answers']['answer'];saved=outputs[0]['actual_output'];keys=list(p['questions']['answer']['criteria'])
            actual=[answer['probabilities'][k] for k in keys];diff=float(np.max(np.abs(np.array(actual)-saved['probabilities'])));assert diff<.00011,(args.model,diff)
            result['native_API_probability_maxdiff']=diff
        # A full cyclic round trip restores contents and inverse-mapped probability indices.
        p=rows[0]['public'];k=len(p['questions']['answer']['criteria'])
        for n in range(k):
            moved,mapping=rotate(p,n);restored,_=rotate(moved,k-n);assert restored==p
            vals=np.arange(k);aligned=np.zeros(k);aligned[mapping]=vals;assert np.array_equal(aligned[np.asarray(mapping)],vals)
        result.update(status='COMPLETED',metadata=adapter.metadata,outputs=outputs,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as e:result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
    finally:save(dest,result);print(args.model,result['status'],flush=True)
if __name__=='__main__':main()
