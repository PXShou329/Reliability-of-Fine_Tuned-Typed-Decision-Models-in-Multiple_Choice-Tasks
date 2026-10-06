from datetime import datetime,timezone
from utils import ROOT,read,save,write,now,T0,DEADLINE

def main():
    current=datetime.now(timezone.utc);elapsed=(current-datetime.fromisoformat(T0)).total_seconds()/3600
    remaining=(datetime.fromisoformat(DEADLINE)-current).total_seconds()/3600
    coverage={}
    for m in ['laya_base','laya_finetuned','clef_flash']:
        counts={d:0 for d in ['arc','csqa','obqa']}
        for p in (ROOT/'raw'/m/'items').glob('*.json'): counts[read(p)['dataset']]+=1
        coverage[m]=counts
    workflow=read(ROOT/'workflow_checkpoint.json') if (ROOT/'workflow_checkpoint.json').exists() else {'status':'preflight/download'}
    training=read(ROOT/'checkpoints/training_status.json') if (ROOT/'checkpoints/training_status.json').exists() else {'status':'not_started'}
    macros=read(ROOT/'results/metrics.json')['metrics'] if (ROOT/'results/metrics.json').exists() else None
    record={'time':now(),'elapsed_hours_since_original_t0':elapsed,'remaining_budget_hours':remaining,'coverage':coverage,
            'workflow':workflow,'training':training,'metrics_available':bool(macros),'protocol_frozen':(ROOT/'protocol_freeze.json').exists(),
            'old_experiment':'archived/paused/excluded','model_outputs':'real local inference only; no mock scores'}
    stamp=current.strftime('%Y%m%dT%H%M%SZ');save(ROOT/'reports/progress'/(stamp+'.json'),record)
    text=f'''# 進度紀錄 {record['time']}

- 原始 T0：2026-10-06 01:32 UTC+8；已用 {elapsed:.2f} 小時，15小時預算剩 {remaining:.2f} 小時。
- 已完成：舊實驗348檔hash封存／backend取消查核；三資料集固定抽樣與3636筆轉換／隔離查核；Laya真實pilot與梯度smoke；官方引用核對。
- 正式模型完成題數（ARC／CSQA／OBQA）：{coverage}。
- 當前階段：{workflow.get('stage',workflow.get('status'))}；訓練狀態：{training.get('status')}。
- Protocol freeze：{record['protocol_frozen']}。正式評分資料：{'已保存 results/metrics.json' if macros else '尚無；不得把pilot當正式成果'}。
- 問題／deviation：詳見decisions.md；既有處理包含ARC不可行最大樣本方案、原始Python sandbox可見性、HF eager test cache排除、CDN range下載、Clef NF4硬體限制與Torchvision依賴、eval重複題幹的cluster bootstrap。
- 舊實驗保持暫停，不重送pending request、不跑hidden evaluation、不混入新主結果。
- 下一步：完成核心真實模型配對推論／微調，閱讀原始錯誤案例，重播，數據解讀與研究報告。保留分析／報告時間；非阻塞問題依spec降級而不停整體。
- 目前仍以原始15小時預算完成；若不能完成全部，保留固定IDs的實際partial coverage並如實報告。
'''
    write(ROOT/'reports/progress'/(stamp+'.md'),text);save(ROOT/'progress_latest.json',record);print(record,flush=True)

if __name__=='__main__': main()
