"""Post-training diagnostic replay. It never exposes gold to the model."""
import torch
from adapters import load_adapter
from utils import ROOT,read,save,digest,event,now

def main():
    # Saved predictions select diagnostics; no prompt/model changes occur.
    public={d:read(ROOT/'data/public_eval'/(d+'.json')) for d in ['arc','csqa','obqa']}
    allpub={r['item_id']:r for rs in public.values() for r in rs}
    raw={m:{r['item_id']:r for p in (ROOT/'raw'/m/'items').glob('*.json') for r in [read(p)]} for m in ['laya_base','laya_finetuned','clef_flash']}
    ids=[r['item_id'] for rs in public.values() for r in rs[:3]]
    disagree=[]
    for i in allpub:
        choices={max(range(len(raw[m][i]['response']['probabilities'])),key=lambda j:raw[m][i]['response']['probabilities'][j]) for m in raw}
        if len(choices)>1: disagree.append((max(max(raw[m][i]['response']['probabilities']) for m in raw),i))
    if disagree: ids.append(sorted(disagree,key=lambda t:(-t[0],t[1]))[0][1])
    ids=list(dict.fromkeys(ids));save(ROOT/'replay/selection.json',{'rule':'first three fixed evaluation items per dataset plus highest-confidence model disagreement','ids':ids})
    for m in raw:
        path=ROOT/'models'/('laya' if m=='laya_base' else m)
        if m=='laya_finetuned': path=ROOT/'checkpoints/laya_final'
        adapter=load_adapter(m,path,read(ROOT/'configs/protocol.json'));results=[]
        for i in ids:
            p=allpub[i]['public'];original=raw[m][i]['response']['probabilities']
            for repeat in range(2):
                r=adapter.predict(p);new=r['probabilities'];delta=max(abs(a-b) for a,b in zip(original,new))
                pred=max(range(len(new)),key=lambda j:new[j]);oldpred=max(range(len(original)),key=lambda j:original[j])
                results.append({'model':m,'item_id':i,'repeat':repeat+1,'public':p,'request_hash':digest(p),'response':r,
                               'max_absolute_probability_change':delta,'argmax_same':pred==oldpred,'time':now()})
        save(ROOT/'replay'/(m+'.json'),{'model':m,'results':results,'n_calls':len(results),
                    'all_argmax_same':all(r['argmax_same'] for r in results),'max_probability_change':max(r['max_absolute_probability_change'] for r in results)})
        event('diagnostic_replay_complete',model=m,n=len(results));print(m,'replay',len(results),flush=True)
        del adapter;torch.cuda.empty_cache()

if __name__=='__main__': main()
