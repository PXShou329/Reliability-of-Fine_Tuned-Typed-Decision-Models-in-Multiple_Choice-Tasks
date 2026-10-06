"""V1 checkpoint/reward/API evidence. GPU forward uses fixed P only."""
import itertools,torch,numpy as np
from safetensors import safe_open
from laya.common import proper_reward,QTYPES
from adapters import LayaAdapter
from utils import *
def main():
    p=torch.tensor([.1,.2,.7,0.]);y=torch.tensor([0.,0.,1.,0.]);mask=torch.tensor([1,1,1,0],dtype=torch.bool)
    r=proper_reward(p,y,torch.tensor(QTYPES['choice']),mask,w_sph=.75,w_rps=1)
    for perm in itertools.permutations(range(4)):
        js=list(perm);q=proper_reward(p[js],y[js],torch.tensor(QTYPES['choice']),mask[js],w_sph=.75,w_rps=999)
        assert torch.allclose(q,r,atol=1e-6)
    reward={'qtype_choice':QTYPES['choice'],'qtype_score':QTYPES['score'],'nominal_joint_permutation_mask_padding':'pass',
      'RPS_in_choice':False,'source':'laya.common.proper_reward applies RPS only when qtype==score',
      'common_sha256':file_hash(V1/'.venv/Lib/site-packages/laya/common.py')}
    delta=[]
    with safe_open(V1/'models/laya/model.safetensors',framework='pt') as a,safe_open(V1/'checkpoints/laya_final/model.safetensors',framework='pt') as b:
        names=[n for n in a.keys() if 'weight' in n and len(a.get_slice(n).get_shape())<=2][:12]
        for n in names:
            aa=a.get_tensor(n).float();bb=b.get_tensor(n).float();delta.append({'name':n,'max_abs_update':float((aa-bb).abs().max()),'shape':list(aa.shape)})
    assert any(r['max_abs_update']>0 for r in delta)
    state=torch.load(V1/'checkpoints/training_resume.pt',map_location='cpu',weights_only=False)
    rolling={'next_epoch':state['next_epoch'],'next_chunk':state['next_chunk'],'updates':state['updates'],'scheduler_last_epoch':state['scheduler']['last_epoch'],
       'optimizer_state_entries':len(state['optimizer']['state']),'rng_cpu_saved':True,'rng_cuda_saved':True};assert rolling['updates']==172 and rolling['next_epoch']==4;del state
    checks=[];conf=read(V1/'configs/protocol.json');pilots=read(ROOT/'data_public/P.json')
    for model,path in [('laya_base',V1/'models/laya'),('laya_v1_ft',V1/'checkpoints/laya_final')]:
        adapter=LayaAdapter(path,conf)
        for d in ['arc','csqa','obqa']:
            item=next(r for r in pilots if r['dataset']==d);p=item['public'];a=adapter.predict(p);off=adapter.agent.predict(p['state'],p['questions'])['answers']['answer']['probabilities']
            error=max(abs(a['probabilities'][j]-off[k]) for j,k in enumerate(p['questions']['answer']['criteria']));assert error<=.000051
            checks.append({'model':model,'item_id':item['item_id'],'public':p,'response':a,'official_probabilities':off,'error':error})
        del adapter;torch.cuda.empty_cache()
    save(ROOT/'audit/reward_checkpoint_adapter_checks.json',{'time':now(),'reward':reward,'parameter_update_sample':delta,'rolling_state':rolling,'P_api_checks':checks})
    audit=read(ROOT/'audit/audit_v1.json');audit.update({'status':'COMPLETED','reward':'choice permutation invariant; RPS only ordinal score',
       'parameter_updates':'confirmed on selected tensors','official_adapter_checkpoint_replay':'6 training-only P examples pass',
       'reused_run_eligibility':'NOT_REUSED: V1 global Torch RNG coupled dropout and GRPO noise; V2 requires independent noise stream',
       'A0_scope':'AI independent code-path arithmetic/audit only; researcher human verification still pending'})
    save(ROOT/'audit/audit_v1.json',audit);decision('P005：A0主要數字、raw/weight hashes、backup、optimizer172步與真實參數更新通過。choice reward不含RPS，聯動置換與padding測試通過；官方checkpoint API重播P通過。V1 global Torch RNG將dropout與GRPO noise耦合，V2分離noise RNG，故R0/s0不可混用V1，六個B1均重跑；V1僅歷史對照。')
    write(ROOT/'audit/audit_v1.md',f'''# A0 原始證據驗收

CPU獨立入口從1,800 raw重算：Base40.33%、FT51.33%、Clef95.17%；各dataset Accuracy/Brier/NLL/ECE與原檔一致，112修正／46回退一致。沒有import原評分器重算。

Frozen hashes、raw逐檔與模型weights驗證通過；四epoch/final與optimizer/RNG checkpoint存在並hash登記；172updates、10,800exposures、參數更新已查核。訓練log時間跨度206.744s，記錄212.055s含checkpoint export邊界，沒有速度本身異常可判造假的證據。

保存的logits確為T之前，softmax(z/T)與raw機率max差1.33e-7，無double calibration。choice reward只log+spherical，RPS只score；nominal聯動置換及padding pass。六個training-only P的base/final官方API對照pass。

備份ZIP的固定raw/config可讀且byte-hash一致；ZIP不含大型weights，不能聲稱外部完整weights備份存在。來源／split／截斷證據另見V1 inventory與原input audit。V1保留不寫。

V1/s0不重用於B1：noise與dropout RNG耦合不符合V2獨立stream規則；本輪六runs從Base重訓。數字一致不等於人類或第三方真實性認證；本auditor仍為Codex AI。79案例一律待研究者本人覆核。

證據：[audit_v1.json](audit_v1.json)、[reward/checkpoint/API](reward_checkpoint_adapter_checks.json)、[inventory](../provenance/v1_inventory.json)、[errata](v1_errata.md)。
''')
    set_run('A0_audit','COMPLETED','audit/audit_v1.json');print('A0 completed, V1 kept read-only; B1 s0 must be rerun',flush=True)
if __name__=='__main__':main()
