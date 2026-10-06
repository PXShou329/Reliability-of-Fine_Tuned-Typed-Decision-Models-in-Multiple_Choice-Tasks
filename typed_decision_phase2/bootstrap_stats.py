"""Paired dataset-stratified stem-cluster bootstrap. Fixed10000 replicates; no p-value fishing."""
import numpy as np,re,unicodedata
from utils import digest
def stem(s):return re.sub(r'\W+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()
def paired(rowsA,rowsB,public,metrics=('correct','nll','brier'),replicates=10000,seed=260616):
    aa={r['item_id']:r for r in rowsA};bb={r['item_id']:r for r in rowsB};assert set(aa)==set(bb)
    rng=np.random.default_rng(seed);out=[]
    for metric in metrics:
        if any(aa[i].get(metric) is None or bb[i].get(metric) is None for i in aa):continue
        samples=[];point=[]
        for d in sorted({aa[i]['dataset'] for i in aa}):
            groups={}
            for i in sorted(aa):
                if aa[i]['dataset']==d:groups.setdefault(stem(public[i]['public']['state']),[]).append(i)
            groups=list(groups.values());sums=np.array([sum(aa[i][metric]-bb[i][metric] for i in ids) for ids in groups]);counts=np.array(list(map(len,groups)))
            weights=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=replicates)
            samples.append((weights@sums)/(weights@counts));point.append(float(sum(sums)/sum(counts)))
        bs=np.mean(samples,axis=0);lo,hi=np.quantile(bs,[.025,.975]);out.append({'metric':metric,'difference':float(np.mean(point)),'ci95_low':float(lo),'ci95_high':float(hi),'n':len(aa),
          'replicates':replicates,'seed':seed,'cluster':'normalized stem within dataset; macro equal weight; conditional on checkpoint seed'})
    return out
