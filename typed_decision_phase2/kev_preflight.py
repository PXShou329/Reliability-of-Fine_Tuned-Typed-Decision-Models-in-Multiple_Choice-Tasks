"""Actual bounded Kev trial; pilot only, no E_new or evaluator."""
import argparse,time,traceback
from utils import *
def main():
    a=argparse.ArgumentParser();a.add_argument('--import-only',action='store_true');args=a.parse_args();start=time.perf_counter()
    evidence={'time':now(),'pid':os.getpid(),'command':sys.argv,'attempt':'import' if args.import_only else 'bf16_pilot','status':'RUNNING'}
    dest=ROOT/'audit'/('kev_import.json' if args.import_only else 'kev_pilot.json');save(dest,evidence)
    try:
        sys.path.insert(0,str(ROOT/'sources/kev_runtime'))
        from kev.checkpoint import Checkpoint,LoadOptions
        import torch,transformers,peft
        ck=Checkpoint(str(ROOT/'models/kev'))
        evidence.update(versions={'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},
            meta={k:getattr(ck.meta,k) for k in ['base','base_revision','temperature','head_dim','weights_dtype','option_isolation']})
        if not args.import_only:
            from kev_adapter import KevAdapter
            model=KevAdapter();p=read(ROOT/'data_public/P.json')[:6];outputs=[]
            for r in p:
                z=model.predict(r['public']);assert len(z['logits'])==len(r['public']['questions']['answer']['criteria'])
                assert abs(sum(z['probabilities'])-1)<1e-5
                outputs.append({'item_id':r['item_id'],'public':r['public'],'actual_output':z})
            enc=model.encode(p[0]['public']);raw=torch.tensor(outputs[0]['actual_output']['logits'])
            model.model.head.temperature=model.native_T
            with torch.inference_mode():native=model.model(enc)[0].float().cpu()
            evidence['temperature_maxdiff']=float((native-raw/model.native_T).abs().max());assert evidence['temperature_maxdiff']<1e-5
            model.model.head.temperature=1.;evidence.update(metadata=model.metadata,outputs=outputs,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        evidence['status']='COMPLETED'
    except Exception as e:evidence.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
    finally:evidence['wall_seconds']=time.perf_counter()-start;save(dest,evidence);print(evidence['status'],flush=True)
if __name__=='__main__':main()
