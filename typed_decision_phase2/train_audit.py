"""Committed-epoch exposure/identity audit; partial failed attempts remain separate."""
from safetensors import safe_open
import numpy as np
from utils import *
def main():
    config=read(ROOT/'configs/training.json');rows=[];orders={};checks=[]
    for model,rc in config['runs'].items():
        p=ROOT/'checkpoints'/model
        if not (p/'complete.json').exists():continue
        result=read(p/'complete.json');logs=[];epochs=[]
        for epoch in range(1,5):
            marker=read(p/('epoch_'+str(epoch)+'_committed.json'));file=p/marker['log'];assert file_hash(file)==marker['log_sha256']
            rs=[__import__('json').loads(x) for x in file.read_text(encoding='utf-8').splitlines()];ids=[i for r in rs for i in r['ids']]
            assert len(ids)==len(rc['G_ids']) and set(ids)==set(rc['G_ids']);logs+=rs;epochs.append(ids)
        n=sum(r['n'] for r in logs);assert n==result['item_exposures']==4*len(rc['G_ids'])
        assert result['final_sha256']==file_hash(p/'final/model.safetensors') and result['config_hash']==digest(rc)
        assert result['updates']==__import__('math').ceil(__import__('math').ceil(len(rc['G_ids'])/8)/8)*4
        with safe_open(str(V1/'models/laya/model.safetensors'),framework='numpy') as f:
            keys=[k for k in f.keys() if ('head' in k or 'scorer' in k) and len(f.get_slice(k).get_shape())==2][:4]
            original={k:f.get_tensor(k) for k in keys}
        with safe_open(str(p/'final/model.safetensors'),framework='numpy') as f:
            changes={k:float(np.max(np.abs(f.get_tensor(k)-a))) for k,a in original.items()}
        assert any(x>0 for x in changes.values()),model
        orders[model]=epochs;rows.append({**{k:result[k] for k in ['run','seed','recipe','smoothing','G_n','item_exposures','updates','epochs','training_seconds','total_wall_seconds','reused','final_sha256']},
                      'evidence':str((p/'complete.json').relative_to(ROOT)),'committed_log_hash_check':True,'parameter_change_check':True})
        checks.append({'run':model,'selected_parameter_changes':changes,'committed_epochs_only':True})
    for seed in config['seeds']:
        a='B1_grpo_ce_'+str(seed);b='B1_ce_'+str(seed)
        if a in orders and b in orders:assert orders[a]==orders[b],('same-seed order mismatch',seed)
    csv_write(ROOT/'results/training_matrix.csv',rows)
    save(ROOT/'audit/training_audit.json',{'time':now(),'completed_runs':len(rows),'planned_runs':len(config['runs']),'checks':checks,'same_seed_B1_orders_identical':True,'failed_attempts_not_counted_as_exposures':True})
    print('training audit complete runs',len(rows),flush=True)
if __name__=='__main__':main()
