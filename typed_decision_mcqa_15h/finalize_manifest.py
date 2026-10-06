"""Inventory real completed artifacts and verify protocol consistency."""
from utils import ROOT,read,save,file_hash,now,digest

def main():
    freeze=read(ROOT/'protocol_freeze.json');mismatch=[]
    for name,expected in freeze['files_sha256'].items():
        p=ROOT/name
        if not p.exists() or file_hash(p)!=expected: mismatch.append(name)
    models={};revs=read(ROOT/'sources/revisions.json')
    for key,repo,rev in [('laya_base','convaiinnovations/laya',revs['laya']),('laya_finetuned','local/laya-mcqa-local-20261006',None),('clef_flash','Cloudflare/clef-flash',revs['clef_flash'])]:
        metadata=[read(p) for p in (ROOT/'raw'/key).glob('run_metadata_*.json')]
        checkpoint=read(ROOT/'raw'/key/'checkpoint.json') if (ROOT/'raw'/key/'checkpoint.json').exists() else None
        path=ROOT/'models'/('laya' if key=='laya_base' else key)
        if key=='laya_finetuned': path=ROOT/'checkpoints/laya_final'
        weightfiles=sorted(path.glob('*.safetensors'))
        models[key]={'model_id':repo,'revision':rev,'fine_tuned_from_revision':revs['laya'] if key=='laya_finetuned' else None,
                     'nominal_parameters':'421M' if key.startswith('laya') else '9B',
                     'weights_sha256':{p.name:file_hash(p) for p in weightfiles},'actual_run_metadata':metadata,'checkpoint':checkpoint,
                     'training_status':read(ROOT/'checkpoints/training_status.json') if key=='laya_finetuned' and (ROOT/'checkpoints/training_status.json').exists() else None}
    save(ROOT/'model_manifest.json',{'time':now(),'models':models})
    raw={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'raw').glob('*/items/*.json')}
    save(ROOT/'raw_integrity_manifest.json',{'time':now(),'n_formal_raw_records':len(raw),'files_sha256':raw})
    save(ROOT/'protocol_integrity_check.json',{'time':now(),'freeze_hash_mismatches':mismatch,'passed':not mismatch,
                                             'note':'Any mismatch must be explained in decisions/deviation log before claiming protocol consistency.'})
    if mismatch: raise RuntimeError('Frozen files changed: '+str(mismatch))
    print('Manifest finalized, formal raw records',len(raw),'freeze files unchanged',flush=True)

if __name__=='__main__': main()
