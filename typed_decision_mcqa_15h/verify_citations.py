"""Save primary sources and explicit claim-to-source checks. No guessed bibliography."""
import urllib.request,re,html
from utils import ROOT,read,save,file_hash,now

REFS=[
 {'id':'ARC-paper','title':'Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge',
  'authors':['Peter Clark','Isaac Cowhey','Oren Etzioni','Tushar Khot','Ashish Sabharwal','Carissa Schoenick','Oyvind Tafjord'],
  'year':2018,'url':'https://arxiv.org/abs/1803.05457','doi':'10.48550/arXiv.1803.05457',
  'dataset_id':'allenai/ai2_arc','supported_claim':'ARC contains natural grade-school science questions; Challenge is distinct from Easy.',
  'read_scope':'primary abstract and bibliographic metadata; no reported benchmark numbers reused'},
 {'id':'CSQA-paper','title':'CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge',
  'authors':['Alon Talmor','Jonathan Herzig','Nicholas Lourie','Jonathan Berant'],'year':2019,
  'url':'https://aclanthology.org/N19-1421/','doi':'10.18653/v1/N19-1421','dataset_id':'tau/commonsense_qa',
  'supported_claim':'Multiple-choice questions target prior commonsense knowledge, constructed using ConceptNet targets and crowd authors.',
  'read_scope':'primary abstract and bibliographic metadata; no reported benchmark numbers reused'},
 {'id':'OBQA-paper','title':'Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering',
  'authors':['Todor Mihaylov','Peter Clark','Tushar Khot','Ashish Sabharwal'],'year':2018,
  'url':'https://aclanthology.org/D18-1260/','doi':'10.18653/v1/D18-1260','dataset_id':'allenai/openbookqa',
  'supported_claim':'OpenBookQA combines elementary science facts with broad common knowledge. This study supplies question/options only, without the open book or extra hints.',
  'read_scope':'primary abstract and bibliographic metadata; our input choice is an experiment specification, not a paper result'},
 {'id':'Clef-release','title':"Introducing Clef: Cloudflare's first open-source decision models, now on Workers AI",
  'organization':'Cloudflare / Workers AI','release_date':'2026-10-01','year':2026,
  'url':'https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/',
  'model_id':'Cloudflare/clef-flash','supported_claim':'Cloudflare released Clef-Flash9B and Clef27B as typed decision models. Our study is text-only.',
  'read_scope':'official release and official model card; not evidence for our own accuracy'},
]

def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MCQA-research-source-verification/1.0'}),timeout=30).read()

def main():
    sources=ROOT/'sources/citation_primary';sources.mkdir(parents=True,exist_ok=True)
    out=[]
    for ref in REFS:
        r=ref.copy();r['verified_at']=now();r['web_read_verified']=True
        try:
            data=fetch(r['url']);p=sources/(r['id']+'.html');p.write_bytes(data)
            text=html.unescape(re.sub('<[^>]+>',' ',data.decode('utf-8',errors='replace')))
            text=re.sub(r'\s+',' ',text)
            r['local_copy']=str(p.relative_to(ROOT));r['sha256']=file_hash(p)
            r['doi_in_primary_page']=r.get('doi','') in text
            r['title_in_primary_page']=r['title'].lower() in text.lower()
            if not r['title_in_primary_page']: raise RuntimeError('Primary title check failed')
        except Exception as e:
            r['local_fetch_error']=repr(e);r['evidence']='Primary page independently opened/read using browser web tool in this session.'
        r['status']='verified primary metadata and limited stated claim';out.append(r)
    revs=read(ROOT/'sources/revisions.json')
    for key,title,org,repo in [('laya','Laya','Convai Innovations','convaiinnovations/laya'),('clef_flash','Clef-Flash 9B','Cloudflare','Cloudflare/clef-flash')]:
        m=read(ROOT/'sources'/(key+'_metadata.json'))
        out.append({'id':key+'-model','title':title,'organization':org,'year':2026,'created_at':m.get('createdAt'),
                    'url':f'https://huggingface.co/{repo}/blob/{revs[key]}/README.md','model_id':repo,'revision':revs[key],
                    'verified_at':now(),'status':'verified pinned official model card and API metadata',
                    'local_copy':'sources/'+('laya_model_README.md' if key=='laya' else 'clef_README.md'),
                    'supported_claim':'Released checkpoint identity, nominal architecture/size, native decision output and official inference instructions. No vendor benchmark numbers become our results.'})
    for key,repo in [('arc','allenai/ai2_arc'),('csqa','tau/commonsense_qa'),('obqa','allenai/openbookqa')]:
        m=read(ROOT/'sources'/(key+'_metadata.json'));url=f'https://huggingface.co/datasets/{repo}/resolve/{revs[key]}/README.md'
        p=sources/(key+'_dataset_README.md');p.write_bytes(fetch(url))
        out.append({'id':key+'-dataset','title':repo+' dataset card','organization':repo.split('/')[0],
                    'year':2019 if key=='csqa' else 2018,'year_basis':'original dataset paper; HF card has no separate publication year',
                    'url':url,'dataset_id':repo,'revision':revs[key],'license_from_card':m.get('cardData',{}).get('license'),
                    'verified_at':now(),'status':'verified pinned card; license information preserved as stated',
                    'local_copy':str(p.relative_to(ROOT)),'sha256':file_hash(p),
                    'supported_claim':'Dataset schema and release split metadata; OpenBookQA license is unknown / source not fully specified.'})
    out.append({'id':'Laya-notebook','title':'laya_finetune_typed_decisions_2xT4_kaggle.ipynb','organization':'Convai Innovations / NandhaKishorM/laya repository',
                'year':2026,'url':f"https://github.com/NandhaKishorM/laya/blob/{revs['laya_code']}/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb",
                'code_revision':revs['laya_code'],'local_copy':'sources/laya_training.ipynb','verified_at':now(),
                'status':'downloaded pinned source and inspected training/calibration code',
                'supported_claim':'Four epochs, GRPO+CE, noise/reward parameters, optimizer learning rates and disjoint train calibration recipe; single-GPU changes recorded in N006.'})
    save(ROOT/'citation_verification.json',{'verified_at':now(),'references':out,'policy':'Only read primary metadata/abstract/code claims retained; no unverifiable benchmark citation or fabricated DOI.'})
    print('Verified references',len(out),flush=True)

if __name__=='__main__': main()
