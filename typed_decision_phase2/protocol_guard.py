"""Integrity checks only; no answers or scores are parsed by inference."""
from utils import *

def verify_lock(model=None):
    lock=read(ROOT/'manifests/protocol.lock.json');assert lock['state']=='FROZEN'
    for section in ['code_sha256','artifact_sha256']:
        for name,h in lock[section].items():assert file_hash(ROOT/name)==h,('frozen artifact changed',name)
    assert file_hash(ROOT/'data_public/E_new.json')==lock['E_new_public_sha256']
    for p,h in lock.get('dependency_source_sha256',{}).items():assert file_hash(Path(p))==h,('pinned runtime source changed',p)
    names=lock['models'] if model is None else [model]
    for name in names:
        assert name in lock['models']
        identity=lock['identities'][name]
        assert file_hash(ROOT/'provenance'/('model_'+name+'.json'))==identity['metadata_sha256']
        for p,h in identity.get('weight_files',{}).items():assert file_hash(Path(p))==h,('checkpoint changed',name,p)
        for p,h in identity.get('model_files',{}).items():assert file_hash(Path(p))==h,('model config/tokenizer changed',name,p)
    return lock
