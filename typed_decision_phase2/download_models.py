"""Bounded resumable public HF byte-range downloads; official LFS/git hashes verified."""
import argparse,urllib.request,json,time,hashlib,os
from utils import *
def request(url,headers=None):return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'phase2-download/1.0',**(headers or {})}),timeout=40)
def download(repo,rev,prefix,out):
    tree=json.loads(request(f'https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true&limit=1000').read())
    entries=[]
    for r in tree:
        n=r.get('path','')
        if r['type']!='file':continue
        if prefix and not n.startswith(prefix):continue
        local=n[len(prefix):] if prefix else n
        allowed=local.endswith('.safetensors') or local in ['head.pt','config.json','adapter_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt','preprocessor_config.json','processor_config.json','chat_template.jinja','generation_config.json','rl_agent_config.json','encoder/config.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json'] or local.endswith('.safetensors.index.json')
        if not allowed:continue
        entries.append((r,local))
    out.mkdir(parents=True,exist_ok=True);done=[]
    for entry,n in entries:
        p=out/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');size=entry['size'];h=entry.get('lfs',{}).get('oid') or entry.get('oid')
        algo='sha256' if entry.get('lfs') else 'git-blob-sha1'
        def actual(path):
            if algo=='sha256':return file_hash(path)
            hh=hashlib.sha1(('blob '+str(path.stat().st_size)+'\0').encode());hh.update(path.read_bytes());return hh.hexdigest()
        if p.exists():
            if p.stat().st_size!=size or actual(p)!=h:raise RuntimeError('Existing asset hash mismatch; preserved '+str(p))
        else:
            offset=tmp.stat().st_size if tmp.exists() else 0
            if offset>size:raise RuntimeError('Oversize partial preserved '+str(tmp))
            while offset<size:
                if not budget_ok(300):raise RuntimeError('Download stopped at experiment budget reserve')
                end=min(size-1,offset+32*1024*1024-1);url=f'https://huggingface.co/{repo}/resolve/{rev}/{entry["path"]}'
                for attempt in range(3):
                    try:
                        with request(url,{'Range':f'bytes={offset}-{end}'}) as response:
                            data=response.read()
                            if len(data)!=end-offset+1:raise RuntimeError('Range size mismatch: '+str(len(data)))
                        break
                    except Exception as e:
                        event('download_failure',repo=repo,file=n,offset=offset,attempt=attempt,error=repr(e))
                        if attempt==2:raise
                with tmp.open('ab') as f:f.write(data);f.flush();os.fsync(f.fileno())
                offset+=len(data)
                save(ROOT/'manifests/download_status.json',{'time':now(),'repo':repo,'file':n,'bytes':offset,'total_bytes':size,'pid':os.getpid(),'status':'RUNNING'})
            if actual(tmp)!=h:raise RuntimeError('Downloaded official hash mismatch; partial retained '+str(tmp))
            os.replace(tmp,p)
        done.append({'file':n,'bytes':size,'official_hash':h,'hash_type':algo,'sha256':file_hash(p)})
        print(repo,n,round(size/2**20,1),'MiB verified',flush=True)
    save(out/'download_manifest.json',{'time':now(),'repo':repo,'revision':rev,'prefix':prefix,'files':done,'complete':True})
def main():
    a=argparse.ArgumentParser();a.add_argument('--extras',action='store_true');args=a.parse_args();revs=read(ROOT/'sources/revisions.json')
    for key in ['kev','kev_base']+(['generative','laya_family'] if args.extras else []):
        r=revs[key];download(r['repo'],r['revision'],'multilingual/' if key=='laya_family' else '',ROOT/'models'/('laya_multilingual' if key=='laya_family' else key))
    save(ROOT/'manifests/download_complete.json',{'time':now(),'extras':args.extras,'status':'COMPLETED'})
if __name__=='__main__':main()
