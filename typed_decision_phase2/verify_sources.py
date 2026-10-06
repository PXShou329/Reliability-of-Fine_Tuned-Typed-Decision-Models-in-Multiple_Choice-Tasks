"""Re-fetch primary source artifacts; record exact scope and failures without blocking GPU jobs."""
import urllib.request,html,re
from utils import *
def main():
    prior=read(V1/'citation_verification.json')['references'];keep=['ARC-paper','CSQA-paper','OBQA-paper','laya-model','clef_flash-model','arc-dataset','csqa-dataset','obqa-dataset','Laya-notebook']
    refs=[dict(r) for r in prior if r['id'] in keep or (r.get('dataset_id') and 'card' in r['id'].lower()) or r['id'].startswith('laya-notebook')]
    # Final entries are checked against the primary pages opened in this Phase2 session.
    refs += [
      {'id':'Guo2017','title':'On Calibration of Modern Neural Networks','authors':['Chuan Guo','Geoff Pleiss','Yu Sun','Kilian Q. Weinberger'],'year':2017,'url':'https://proceedings.mlr.press/v70/guo17a.html','doi':None,'supported_claim':'Temperature scaling is an established single-parameter post-processing calibration approach; no guarantee of our performance.'},
      {'id':'Ovadia2019','title':"Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift",'authors':['Yaniv Ovadia','Emily Fertig','Jie Ren','Zachary Nado','D Sculley','Sebastian Nowozin','Joshua V. Dillon','Balaji Lakshminarayanan','Jasper Snoek'],'year':2019,'url':'https://arxiv.org/abs/1906.02530','doi':'10.48550/arXiv.1906.02530','supported_claim':'Uncertainty/calibration under dataset shift warrants separate evaluation; no borrowed numerical effects.'},
      {'id':'Pezeshkpour2024','title':'Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions','authors':['Pouya Pezeshkpour','Estevam Hruschka'],'year':2024,'url':'https://aclanthology.org/2024.findings-naacl.130/','doi':'10.18653/v1/2024.findings-naacl.130','supported_claim':'Option-order sensitivity in MCQA is an existing research problem, not invented here.'},
      {'id':'Kev-model','title':'Kev-4B','authors':['Jared Palmer'],'year':2026,'year_basis':'HF repository creation metadata; not a paper publication year','url':'https://huggingface.co/jaredpalmer/kev-4b/blob/'+read(ROOT/'sources/revisions.json')['kev']['revision']+'/README.md','doi':None,'supported_claim':'Released typed-decision checkpoint consists of pinned Qwen base, LoRA adapter, pointer head and native temperature; official loader, not local hardware performance guarantee.'},
      {'id':'Qwen-model','title':'Qwen3.5-4B','authors':['Qwen Team'],'year':2026,'year_basis':'HF repository creation metadata; not a paper publication year','url':'https://huggingface.co/Qwen/Qwen3.5-4B/blob/'+read(ROOT/'sources/revisions.json')['generative']['revision']+'/README.md','doi':None,'supported_claim':'Post-trained generative model identity and official enable_thinking=False template parameter; no vendor benchmark results used.'}
    ]
    results=[]
    for r in refs:
        if r['id'].endswith('-dataset'):r['supported_claim']='Pinned '+r['dataset_id']+' schema and original split metadata only; licensing claim belongs only to its own card.'
        r.pop('local_copy',None);r.pop('sha256',None);r.pop('verified_at',None)
        url=r['url'].replace('/blob/','/resolve/') if 'huggingface.co/' in r['url'] else r['url']
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Phase2-primary-source-audit'}),timeout=30) as response:data=response.read()
            suffix='.md' if 'README.md' in url else '.html';p=ROOT/'sources/citation_primary'/(r['id']+suffix);p.parent.mkdir(exist_ok=True);p.write_bytes(data)
            r.update(local_copy=str(p.relative_to(ROOT)),sha256=file_hash(p),fetch_status='COMPLETED')
        except Exception as e:r.update(fetch_status='FAILED',fetch_error=repr(e))
        r['phase2_checked_at']=now();r['read_scope']='primary abstract/bibliographic metadata or pinned model card/source identity; no full-paper benchmark conclusions'
        r['status']='NEEDS_REVIEW';results.append(r);print(r['id'],r['fetch_status'],flush=True)
    save(ROOT/'sources/citation_verification_pending.json',{'time':now(),'references':results,'note':'Network success is not itself claim verification; root reviews exact supported scope before final ledger.'})
if __name__=='__main__':main()
