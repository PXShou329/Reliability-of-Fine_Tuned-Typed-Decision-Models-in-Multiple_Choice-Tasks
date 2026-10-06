import urllib.request,json
from utils import *
def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'local-phase2-research'}),timeout=30).read()
def main():
    refs={}
    for key,repo in [('kev','jaredpalmer/kev-4b'),('kev_base','Qwen/Qwen3.5-4B-Base'),('generative','Qwen/Qwen3.5-4B'),('laya_family','convaiinnovations/laya')]:
        meta=json.loads(fetch('https://huggingface.co/api/models/'+repo));save(ROOT/'sources'/(key+'_metadata.json'),meta)
        rev=meta['sha'];refs[key]={'repo':repo,'revision':rev}
        card=fetch(f'https://huggingface.co/{repo}/resolve/{rev}/README.md');(ROOT/'sources'/(key+'_README.md')).write_bytes(card)
    commit=json.loads(fetch('https://api.github.com/repos/jaredpalmer/kev/commits/main'))['sha']
    tree=json.loads(fetch('https://api.github.com/repos/jaredpalmer/kev/git/trees/'+commit+'?recursive=1'));save(ROOT/'sources/kev_repo_tree.json',tree)
    refs['kev_runtime']={'repo':'jaredpalmer/kev','revision':commit}
    selected=[r['path'] for r in tree['tree'] if r['path'].startswith('kev/') and r['path'].endswith('.py')]
    selected+=['pyproject.toml','README.md']
    for n in selected:
        data=fetch(f'https://raw.githubusercontent.com/jaredpalmer/kev/{commit}/{n}');p=ROOT/'sources/kev_runtime'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    save(ROOT/'sources/revisions.json',refs)
    print({'revisions':refs,'python_files':selected,'kev_files':[x['rfilename'] for x in read(ROOT/'sources/kev_metadata.json')['siblings']],
        'laya_files':[x['rfilename'] for x in read(ROOT/'sources/laya_family_metadata.json')['siblings']]},flush=True)
if __name__=='__main__':main()
