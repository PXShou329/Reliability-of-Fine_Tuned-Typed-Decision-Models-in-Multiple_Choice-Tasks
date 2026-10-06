"""Pre-output recipe/matrix decisions. No E_new outputs or labels read."""
import csv
from utils import *
def main():
    assert read(ROOT/'audit/audit_v1.json')['status']=='COMPLETED';ids=[r['item_id'] for r in read(ROOT/'data_public/G.json')]
    runs={};s0=read(ROOT/'audit/audit_v1.json')['original_training_seed'];seeds=[s0]+[n for n in [1729,2718,31415] if n!=s0][:2]
    def add(id,recipe,seed,ids,**kw):runs[id]={'recipe':recipe,'seed':seed,'G_ids':ids,'estimated_seconds':600,**kw}
    for recipe in ['grpo_ce','ce']:
        for seed in seeds:add(f'B1_{recipe}_{seed}',recipe,seed,ids)
    lodo=read(ROOT/'manifests/lodo_manifest.json')
    for target,r in lodo.items():add(f'B3_{target}_{s0}','ce',s0,r['G_ids'],C_T_ids=r['C_T_ids'],target=target)
    # Resource-only D decision before any new-score disclosure. Costs from A0 and pilot hardware.
    nested=read(ROOT/'manifests/data_budget_nested.json')
    for n in [100,300]:add(f'D2_ce_{n}_{s0}','ce',s0,[i for d in nested for i in nested[d][str(n)]],budget_n=n)
    for seed in seeds:add(f'D3_smooth_{seed}','ce',seed,ids,smoothing=.1)
    for seed in seeds[1:]:
        for target,r in lodo.items():add(f'D4_{target}_{seed}','ce',seed,r['G_ids'],C_T_ids=r['C_T_ids'],target=target)
    common={'max_len':512,'head_max_len':192,'lr_encoder':2.5e-5,'lr_head':1e-4,'epochs':4,'micro_batch':8,'accumulation':8,
       'weight_decay':.01,'optimizer':'AdamW','scheduler':'cosine exact ceil to1e-6','partial_accumulation':'same fixed /8 V1 rule even final partial window',
       'native_historical_view':'Original base native choice:3-5 T; new FT fixed V1 historical T=5, not a fresh fit; Shared-fit uses onlyC_T',
       'shared_fit':'float64 NLL; L-BFGS-B logT bounds [.05,100], starts .5/1/5; maxiter500 ftol1e-12 gtol1e-9; select by C_T objective only'}
    save(ROOT/'configs/training.json',{'time':now(),'seeds':seeds,'s0':s0,'common':common,'runs':runs,'D_resource_decision':
      'All D1-D5 provisionally authorized in exact order: >10h experiment budget, expected training<2h + MCQA inference<1h, reserve C2>=3h and reporting90min. Availability failures remain recorded and cannot block predecessor core. No E_new outputs observed.'})
    rows=list(csv.DictReader((ROOT/'manifests/run_matrix.csv').open(encoding='utf-8-sig')));existing={r['run_id'] for r in rows}
    for id,rc in runs.items():
        if id not in existing:rows.append({'run_id':id,'module':id.split('_')[0],'seed':rc['seed'],'dependencies':'A0;G1;training recipe lock','estimated_minutes':6,'optional':True,'status':'PLANNED','reused':False,'evidence':'','reason':''})
    for r in rows:
        if r['module'].startswith('D'):r['status']='PLANNED'
    csv_write(ROOT/'manifests/run_matrix.csv',rows)
    save(ROOT/'configs/training.lock.json',{'time':now(),'code_sha256':file_hash(ROOT/'train.py'),'config_sha256':file_hash(ROOT/'configs/training.json'),
      'G_public_sha256':file_hash(ROOT/'data_public/G.json'),'G_private_sha256':file_hash(ROOT/'evaluator_private/G.json'),'A0_pass':True,'before_E_new_model_access':True})
    decision('P006：預先鎖定B1六runs/B3四runs；兩種loss分離noise/dropout/shuffle RNG。Shared-fit float64 logT .05–100共同程序，new FT historical固定借用V1 T5僅描述，主比較與分流只採C_T重擬合。根據剩餘>10h與P/A0速度估計，D1→D5事前納入，僅依資源／相容性狀態撤下，不看E_new分數。')
    print({'seeds':seeds,'training_runs':len(runs),'D_order':'D1,D2,D3,D4,D5'},flush=True)
if __name__=='__main__':main()
