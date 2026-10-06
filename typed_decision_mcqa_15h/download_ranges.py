"""Pinned HF downloads with explicit CDN byte ranges, resume, and official hash verification."""
import os,fnmatch,time,hashlib,urllib.request
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import get_hf_file_metadata,hf_hub_url
from utils import ROOT,read,save,file_hash,event,now

CHUNK=32*1024*1024
REVS=read(ROOT/'sources/revisions.json')

def check(p,etag):
    if len(etag)==64: return file_hash(p)==etag
    if len(etag)==40:
        h=hashlib.sha1();h.update(f'blob {p.stat().st_size}\0'.encode())
        with p.open('rb') as f:
            while b:=f.read(CHUNK): h.update(b)
        return h.hexdigest()==etag
    raise ValueError('Unrecognized official etag')

def download(key,file):
    repo=read(ROOT/'sources'/(key+'_metadata.json'))['id']
    url=hf_hub_url(repo,file,revision=REVS[key])
    dest=ROOT/'models'/key/file;dest.parent.mkdir(parents=True,exist_ok=True)
    meta=get_hf_file_metadata(url)
    if dest.exists() and dest.stat().st_size==meta.size and check(dest,meta.etag): return key,file,'verified-existing'
    part=dest.with_name(dest.name+'.range.part');state=dest.with_name(dest.name+'.range.json')
    if state.exists():
        old=read(state)
        if old['etag']!=meta.etag or old['revision']!=REVS[key]: raise RuntimeError('Partial revision mismatch')
    if part.exists() and not state.exists(): raise RuntimeError('Partial missing metadata')
    save(state,{'revision':REVS[key],'etag':meta.etag,'size':meta.size,'source':url,'started_at':now()})
    offset=part.stat().st_size if part.exists() else 0
    if offset>meta.size: raise RuntimeError('Oversized partial')
    with part.open('ab') as f:
        while offset<meta.size:
            end=min(offset+CHUNK,meta.size)-1;last_error=None
            for attempt in range(5):
                try:
                    if attempt: meta=get_hf_file_metadata(url)
                    req=urllib.request.Request(meta.location,headers={'Range':f'bytes={offset}-{end}'})
                    with urllib.request.urlopen(req,timeout=60) as resp:
                        data=resp.read(end-offset+2)
                        if meta.size>CHUNK or offset:
                            if resp.status!=206 or resp.headers.get('Content-Range')!=f'bytes {offset}-{end}/{meta.size}':
                                raise RuntimeError('CDN returned incorrect range')
                        if len(data)!=end-offset+1: raise RuntimeError('Incomplete range')
                    break
                except Exception as e:
                    last_error=repr(e);event('download_range_retry',model=key,file=file,offset=offset,attempt=attempt,error=last_error)
                    time.sleep(min(2**attempt,10))
            else: raise RuntimeError(f'{key}/{file} range failed: {last_error}')
            f.write(data);f.flush();os.fsync(f.fileno());offset=end+1
            save(state,{'revision':REVS[key],'etag':meta.etag,'size':meta.size,'completed_bytes':offset,'source':url,'updated_at':now()})
            if meta.size>CHUNK: print(key,file,offset,'/',meta.size,flush=True)
    if not check(part,meta.etag): raise RuntimeError('Official content hash mismatch')
    os.replace(part,dest);save(state,{'revision':REVS[key],'etag':meta.etag,'size':meta.size,'completed_bytes':offset,'verified':True,'updated_at':now()})
    event('model_file_verified',model=key,file=file,size=meta.size,etag=meta.etag)
    return key,file,'downloaded-verified'

def main():
    patterns={'laya':['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'],
              'clef_flash':['*.json','*.safetensors','chat_template.jinja','joint_schema_model.py','README.md','LICENSE']}
    tasks=[]
    for key in patterns:
        for entry in read(ROOT/'sources'/(key+'_metadata.json'))['siblings']:
            file=entry['rfilename']
            if any(fnmatch.fnmatch(file,p) for p in patterns[key]): tasks.append((key,file))
    # Laya small metadata + weight prioritized, independent Clef files continue concurrently.
    tasks.sort(key=lambda t:(t[0]!='laya',t[1].endswith('.safetensors'),t[1]))
    with ThreadPoolExecutor(max_workers=4) as ex:
        fs=[ex.submit(download,*t) for t in tasks]
        for f in fs: print(f.result(),flush=True)
    for key in patterns:
        save(ROOT/'models'/key/'download_completed.json',{'revision':REVS[key],'time':now(),'transport':'explicit HTTP byte ranges','all_files_verified':True})

if __name__=='__main__': main()
