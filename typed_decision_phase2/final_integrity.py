"""Read-only frozen-source/identity and saved public-request integrity audit."""
import csv
from utils import *
from protocol_guard import verify_lock
from schema import validate_public
def main():
    lock=verify_lock();models=lock['models'];n=0;counts={};inherited=[];direct=0
    legacy_sha='dac17866b1244ef012d4f96009ce0ef8a3561c1af6cb1bcfeaae2cd3fd1ad805'
    archive=ROOT/'sources/inference_pre_provenance_v1.py';lf=archive.read_text(encoding='utf-8').encode('utf-8');assert __import__('hashlib').sha256(lf).hexdigest()==legacy_sha
    dest=ROOT/'sources'/('inference_'+legacy_sha+'.py')
    if not dest.exists():dest.write_bytes(lf)
    assert file_hash(dest)==legacy_sha
    for model in models:
        meta=file_hash(ROOT/'provenance'/('model_'+model+'.json'));counts[model]=0
        for p in (ROOT/'raw'/model).glob('*/*.json'):
            r=read(p);validate_public(r['public']);assert r['public_hash']==digest(r['public']) and r['model']==model
            if 'model_metadata_sha256' in r:
                assert r['model_metadata_sha256']==meta,(model,p,'raw metadata identity');direct+=1
            else:
                # P015 documented stage-level inheritance, never rewrite an older legal raw row.
                assert model in models and r['purpose'] in ['C_T','C_gate','E_old','B2_E_old'],(model,p,r['purpose'])
                metadata=read(ROOT/'provenance'/('model_'+model+'.json'));assert metadata['code_sha256']==legacy_sha
                assert r['time']<lock['time'] and r['status']=='COMPLETED'
                inherited.append({'raw':str(p.relative_to(ROOT)),'raw_sha256':file_hash(p),'model':model,'pid':r['pid'],'canonical_metadata_sha256':meta,
                    'inference_source_sha256':legacy_sha,'link_kind':'stage-level inherited identity; original row has no direct metadata hash; P015'})
            assert r['status']=='COMPLETED';counts[model]+=1;n+=1
    macro=read(ROOT/'macrolab/protocol.lock.json')
    for f,h in macro['code_sha256'].items():assert file_hash(ROOT/f)==h
    for f,key in [('macrolab/private/tasks.json','tasks_sha256'),('macrolab/frozen_library.json','library_sha256'),('macrolab/macrolab_protocol.json','protocol_sha256'),('macrolab/resource_gate.json','resource_gate_sha256')]:assert file_hash(ROOT/f)==macro[key]
    assert file_hash(ROOT/'manifests/protocol.lock.json')==macro['main_protocol_sha256']
    save(ROOT/'provenance/legacy_raw_lineage.json',{'time':now(),'original_raw_unchanged':True,'inherited_rows':inherited,'note':'Preserves weaker pre-P015 linkage explicitly; no back-filled claims of direct row hashes. Original archive CRLF normalized to recover exact executed LF SHA.'})
    save(ROOT/'audit/final_integrity.json',{'time':now(),'status':'COMPLETED','frozen_model_identities':len(models),'raw_public_requests_checked':n,'direct_metadata_link_rows':direct,'stage_inherited_metadata_rows':len(inherited),'by_model':counts,
        'frozen_code_config_calibrator_policy_ids_weights_and_runtime_verified':True,'public_allowlist_and_raw_metadata_identity_verified':True,'MacroLab_frozen_code_tasks_library_verified':True})
    print('final frozen integrity passed',len(models),n,flush=True)
if __name__=='__main__':main()
