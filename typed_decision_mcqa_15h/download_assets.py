"""Only pinned public model/dataset downloads; no inference or eval selection."""
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download,hf_hub_download
from datasets import load_dataset
from utils import ROOT, read, save, event, now

REVS=read(ROOT/'sources/revisions.json')
MODELS={
 'laya':('convaiinnovations/laya',['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*']),
 'clef_flash':('Cloudflare/clef-flash',['*.json','*.safetensors','chat_template.jinja','joint_schema_model.py','README.md','LICENSE']),
}
DATASETS={'arc':('allenai/ai2_arc','ARC-Challenge'),'csqa':('tau/commonsense_qa',None),'obqa':('allenai/openbookqa','main')}

def model(key):
    repo,patterns=MODELS[key]
    event('download_model_start',model=key,repo=repo,revision=REVS[key])
    p=snapshot_download(repo,revision=REVS[key],allow_patterns=patterns,
                        local_dir=ROOT/'models'/key,max_workers=3,cache_dir=ROOT/'cache/hf')
    save(ROOT/'models'/key/'download_completed.json',{'repo':repo,'revision':REVS[key],'path':str(p),'time':now()})
    event('download_model_done',model=key,path=str(p))
    return key

def dataset(key):
    repo,config=DATASETS[key]
    event('download_dataset_start',dataset=key,revision=REVS[key])
    # Explicit files avoid the generic HF builder eagerly caching every split.
    info={}
    for split in ['train','validation']:
        directory='data' if key=='csqa' else config
        source=hf_hub_download(repo,f'{directory}/{split}-00000-of-00001.parquet',repo_type='dataset',
                               revision=REVS[key],cache_dir=ROOT/'cache/hf_datasets')
        ds=load_dataset('parquet',data_files={split:source},split=split,cache_dir=str(ROOT/'cache/datasets'))
        dest=ROOT/'data/source'/key/split
        if not dest.exists(): ds.save_to_disk(str(dest))
        info[split]={'count':len(ds),'fingerprint':ds._fingerprint,'features':str(ds.features)}
    save(ROOT/'data/source'/key/'manifest.json',{'repo':repo,'config':config,'revision':REVS[key],'splits':info})
    event('download_dataset_done',dataset=key,splits=info)
    return key

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['models','datasets','all'],default='all');a=ap.parse_args()
    tasks=[]
    if a.kind in ['all','datasets']: tasks += [(dataset,k) for k in DATASETS]
    if a.kind in ['all','models']: tasks += [(model,k) for k in MODELS]
    with ThreadPoolExecutor(max_workers=3) as ex:
        fs=[ex.submit(fn,k) for fn,k in tasks]
        for f in fs: print(f.result(),flush=True)
