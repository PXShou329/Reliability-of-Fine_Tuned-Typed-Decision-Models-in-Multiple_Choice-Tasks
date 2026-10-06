"""Independent CPU audit. No V1 code import and no write into V1."""
import json,math,csv,zipfile,sys,itertools,time
from collections import defaultdict,Counter
import numpy as np
from utils import *

def scores(p,g):
    p=np.asarray(p,dtype=np.float64);assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and abs(p.sum()-1)<1e-5
    pred=int(p.argmax());return {'correct':int(pred==g),'brier':sum((v-int(j==g))**2 for j,v in enumerate(p)),
     'nll':-math.log(max(p[g],1e-12)),'confidence':p[pred]}
def agg(rows):
    bins=defaultdict(list)
    for r in rows:bins[min(14,int(r['confidence']*15))].append(r)
    return {**{k:float(np.mean([r[k] for r in rows])) for k in ['correct','brier','nll']},
     'ece':sum(len(b)/len(rows)*abs(np.mean([r['correct'] for r in b])-np.mean([r['confidence'] for r in b])) for b in bins.values())}
def tests():
    assert abs(scores([.25]*4,0)['brier']-.75)<1e-12
    assert scores([1,0],0)['correct']==1 and scores([1,0],1)['correct']==0
    assert scores([0,1],0)['nll']==-math.log(1e-12)
    assert abs(agg([scores([.8,.2],0),scores([.2,.8],0)])['ece']-.3)<1e-12
    for p in [[.1,.2,.7],[.5,.5],[0,1,0,0]]:
        for perm in itertools.permutations(range(len(p))):
            for g in range(len(p)):
                old=scores(p,g);new=scores([p[j] for j in perm],perm.index(g))
                assert abs(old['brier']-new['brier'])<1e-12 and abs(old['nll']-new['nll'])<1e-12
    return 'pass: uniform/full-correct/full-wrong/floor/variable-k/joint-permutation/ECE edge'
def fingerprint(s):return hashlib.sha256(' '.join(s.casefold().split()).encode()).hexdigest()
def main():
    set_run('A0_audit','RUNNING');start=time.perf_counter();inventory=[]
    for p in V1.rglob('*'):
        if p.is_file() and '.venv' not in p.parts and '__pycache__' not in p.parts:
            inventory.append({'path':str(p.relative_to(V1)),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns})
    freeze=read(V1/'protocol_freeze.json');checked={};fail=[]
    for n,h in freeze['files_sha256'].items():
        actual=file_hash(V1/n);checked[n]=actual
        if actual!=h:fail.append(n)
    for n,h in read(V1/'raw_integrity_manifest.json')['files_sha256'].items():
        actual=file_hash(V1/n);checked[n]=actual
        if actual!=h:fail.append(n)
    manifest=read(V1/'model_manifest.json')
    for key,meta in manifest['models'].items():
        path=V1/('models/laya' if key=='laya_base' else 'models/clef_flash' if key=='clef_flash' else 'checkpoints/laya_final')
        for n,h in meta['weights_sha256'].items():
            actual=file_hash(path/n);checked[str((path/n).relative_to(V1))]=actual
            if actual!=h:fail.append(str(path/n))
    for p in (V1/'checkpoints').glob('epoch_*/model.safetensors'):checked[str(p.relative_to(V1))]=file_hash(p)
    rolling=V1/'checkpoints/training_resume.pt';assert rolling.exists();checked[str(rolling.relative_to(V1))]=file_hash(rolling)
    gold={i:g for d in ['arc','csqa','obqa'] for i,g in read(V1/'evaluator_private/gold'/(d+'.json')).items()}
    public={r['item_id']:r for d in ['arc','csqa','obqa'] for r in read(V1/'data/public_eval'/(d+'.json'))}
    groups={};pred={};recompute={};max_decode=0
    for m in ['laya_base','laya_finetuned','clef_flash']:
        groups[m]=defaultdict(list);pred[m]={};paths=list((V1/'raw'/m/'items').glob('*.json'));assert len(paths)==600
        for p in paths:
            r=read(p);i=r['item_id'];assert i in public and r['public']==public[i]['public'] and r['request_hash']==digest(r['public'])
            assert set(r['public'])=={'state','questions'} and set(r['public']['questions']['answer'])=={'type','instructions','criteria'}
            z=np.array(r['response']['logits'],dtype=np.float64);T=r['response']['temperature'];a=np.exp(z/T-(z/T).max());a/=a.sum()
            probs=r['response']['probabilities'];max_decode=max(max_decode,float(np.max(np.abs(a-probs))))
            assert max_decode<2e-7
            s=scores(probs,gold[i]['gold_index']);groups[m][r['dataset']].append(s);pred[m][i]=s['correct']
        assert len(pred[m])==600
        recompute[m]={d:agg(rs) for d,rs in groups[m].items()}
        recompute[m]['macro']={k:float(np.mean([recompute[m][d][k] for d in groups[m]])) for k in ['correct','brier','nll','ece']}
        for d in groups[m]:
            expected=read(V1/'results/metrics.json')['metrics'][m][d]
            for k in ['correct','brier','nll','ece']:assert abs(recompute[m][d][k]-expected['accuracy' if k=='correct' else k])<1e-12
    assert pred['laya_base'].keys()==pred['laya_finetuned'].keys()==pred['clef_flash'].keys()
    gains=sum(not pred['laya_base'][i] and pred['laya_finetuned'][i] for i in public)
    reg=sum(pred['laya_base'][i] and not pred['laya_finetuned'][i] for i in public);assert (gains,reg)==(112,46)
    purposes={};stems={}
    for p in ['training','calibration','pilot','public_eval']:
        rs=[r for d in ['arc','csqa','obqa'] for r in read(V1/'data'/p/(d+'.json'))]
        purposes[p]={r['item_id'] for r in rs};stems[p]={fingerprint(r['public']['state']) for r in rs}
    for a,b in itertools.combinations(purposes,2):assert not purposes[a]&purposes[b] and not stems[a]&stems[b]
    train=[json.loads(s) for s in (V1/'training_log.jsonl').read_text().splitlines()];exposures=sum(r['n'] for r in train)
    assert exposures==10800 and len(train)==1352 and len({i for r in train for i in r['train_ids']})==2700
    assert read(V1/'checkpoints/training_status.json')['updates']==172
    # Backup is actual zip, not an assumed external copy. Verify fixed small artifacts.
    zpath=Path(read(V1/'delivery_package.json')['path']);assert file_hash(zpath)==read(V1/'delivery_package.json')['sha256']
    backup={}
    with zipfile.ZipFile(zpath) as z:
        for n in ['configs/protocol.json','protocol_freeze.json','raw/laya_base/checkpoint.json']:
            assert hashlib.sha256(z.read(n)).hexdigest()==file_hash(V1/n);backup[n]='byte-hash match'
        n=sorted(n for n in z.namelist() if n.startswith('raw/laya_base/items/'))[0]
        assert hashlib.sha256(z.read(n)).hexdigest()==file_hash(V1/n);backup[n]='byte-hash match'
    assert not fail,fail
    result={'time':now(),'status':'evidence_checks_pass_pending_training_reward_and_adapter_smoke','independent_metrics':recompute,
      'raw_count':1800,'ids':600,'unique_stems':len(stems['public_eval']),'gains':gains,'regressions':reg,'max_unscaled_logit_decode_difference':max_decode,
      'original_training_seed':read(V1/'configs/protocol.json')['training']['seed'],'training_exposures':exposures,'optimizer_steps':172,
      'training_seconds_recorded':read(V1/'checkpoints/training_status.json')['training_seconds'],'recorded_timestamp_span_seconds':
          (datetime.fromisoformat(train[-1]['time'])-datetime.fromisoformat(train[0]['time'])).total_seconds(),
      'backup_verified_small_files':backup,'backup_excludes_large_weights':True,'all_v1_frozen_and_recorded_raw_weight_hashes':'pass',
      'source_isolation':'four purpose IDs/content disjoint; test not passed to subject','human_reviewed_n_verified':0,
      'analyst':'Codex AI only, not independent human certification','elapsed_seconds':time.perf_counter()-start,'reused_run_eligibility':'pending RNG audit'}
    save(ROOT/'provenance/v1_inventory.json',{'root':str(V1),'time':now(),'files':inventory,'verified_sha256':checked,'backup':str(zpath)})
    save(ROOT/'audit/audit_v1.json',result);save(ROOT/'audit/tests.json',{'cpu_metric_hand_tests':tests()})
    write(ROOT/'audit/v1_errata.md','''# V1 errata（不覆寫V1）

1. V1的79題「親自閱讀／研究者」實際為Codex AI分析，沒有研究者本人reviewer/date證據。改稱AI輔助案例分析，全部列待人工覆核；數值結果不受影響。
2. V1未執行Kev，原方案至少另一模型條件已由Clef完成，但Kev是優先比較候選；如把整個原計畫聲稱全部完成即過度廣泛。V2明列此scope gap並實際補做。
3. V1 citation_check逐dataset支持範圍複用OpenBookQA license措辭於ARC/CSQA；V2逐列修正，原表保留。
4. V1 teacher brief不必要的耗時與封存敘述於V2教師摘要移除。
5. V1所有human-reviewed主張改為AI-only，不把本次AI auditor稱為獨立人類真實性認證。

數字由raw独立重算一致；checkpoint、reward與實際adapter補驗仍須完成，尚不宣告A0全過。
''')
    print({'raw':1800,'accuracy':{m:v['macro']['correct'] for m,v in recompute.items()},'gains':gains,'regressions':reg,'seconds':time.perf_counter()-start},flush=True)
if __name__=='__main__':main()
