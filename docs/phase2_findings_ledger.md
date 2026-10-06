# 第二階段 findings ledger

此帳本區分直接觀察、統計支援、推測與未完成。MCQA 新保留結果及 MacroLab 正式結果尚未產生時，不填造數字；完成後依實際 CSV／raw 追加。

|ID|觀察／主張|證據|狀態與邊界|
|---|---|---|---|
|A0-01|第一輪主要數字可由 1,800 raw 獨立重算，Base 40.33%、FT 51.33%、Clef 95.17%，112 gain／46 regression 一致|`audit/audit_v1.json`、`audit/reward_checkpoint_adapter_checks.json`|直接證據；限 E_old，AI auditor，不是第三方或人類認證|
|A0-02|沒有 nominal-choice 使用 RPS 的證據；實際分支是 log+spherical，RPS 限 ordinal-score|`audit/reward_checkpoint_adapter_checks.json`、`audit/noise_reward_tests.json`|程式與聯動置換測試支援；不宣稱所有 reward 設計等價|
|A0-03|V1 79 個案例沒有研究者本人 reviewer/date 記錄|`audit/v1_errata.md`|AI 輔助分析，human verified=0|
|A1-01|Kev 真實 BF16 typed 推論可在本機完成 P 與歷史／校準請求|`audit/kev_pilot.json`、`results/inference_kev_pre.json`|可行性已執行；不是官方支援版本配置的保證，尚不代表新評估品質|

## 正式完成後追加（上述文字保留為初期狀態）

|ID|實際觀察／假說狀態|數字與證據|支援強度與解釋界線|
|---|---|---|---|
|A0-04|V1原成果保護透過|2,228檔無增刪/metadata變；1,850hash一致；`audit/v1_unchanged_final.json`|本機read-onlyAI audit，不是人類或外部備份認證|
|A0-05|歷史重播一致|1,800 requests/predictions/rawlogits bit-identical；`v1_historical_replay.json`|E_old為已見歷史，不冒充新保留|
|A1-02|Kev補齊|E_old499/600、E_new196/240；`baseline_completion.csv`|真實BF16官方unfusedruntime，但版本範圍限制；architecture因果不識別|
|B1-01|FT增益支援|GRPO/CE每seed vsBase新Accuracy區間下界正；`paired_comparisons.csv`|固定checkpoint/題目bootstrap；不代表所有新分佈|
|B1-02|GRPO額外優勢未支援|新mean48.61 vs49.72%，每seed差CI跨0|證據不足，非等效證明；3seeds有限|
|B1-03|RawFT失準|meanNLLGRPO2.743、CE2.717，Base1.570；s0Raw.9GRPO164accepted/82wrong|不能只用accuracy或JSON成功代表可靠性|
|B1-04|Shared改善Laya機率|GRPO/CE meanNLL1.225/1.208、Brier.643/.636；`calibration_comparison.csv`|只後處理機率、argmax不改；高thresholdcoverage可為0|
|B1-05|校準多指標tradeoff|ClefNLL差−.01832CI負；Brier+.00176CI跨0；Kev兩差CI跨0|NLL改善有限證據、Brier退步未確立，非普遍新發現|
|B2-01|答案穩定≠機率穩定（非預期）|s0FTflip下降約11–12pp，SharedTV+ .0158/.0193，pairedCI皆正；`order_model_paired.csv`|補充對比P023firstscore前lock；s0/各自calibrator，機制未隔離|
|B2-02|ensemble有成本|Clef+2.92ppCI[+.78,+5.45]；Laya兩FTCI跨0；4–5forwards|同235題、固定alignedmean，不選bestrotation，非免費|
|B3-01|OBQA域依賴支援|三seedtargetloss13.75/12.50/16.25pp，CI低於0|同1,800G/96CT對照；每target80、原因未識別|
|B3-02|ARC/CSQA多數差不精確|8/9foldnegative，1zero；多accuracyCI跨0|不挑fold；不能說所有dataset均有已確立退化|
|C1-01|共載成功、route目標失敗（負面）|960live；bypass92.5%、mean192.159vs192.918ms=.39%reduction|accuracy差0因同expert；20%速度目標未達；非offline simulation|
|C1-02|always-small不合品質|47.92%、比expert−44.58pp，15.045ms|快但非同品質成本優勢，不放寬gate|
|C2-01|模型工具建構無task成功（負面）|Laya/Clef各M0/M1/M2 0/40，enumeration30/40；`macrolab_metrics.csv`|固定DSL/prompt/budget，不能外推所有agent；未改權重|
|C2-02|feedback改行為但不成功|M0/M1首candidate一致40/40；末candidateLaya31/Clef37不同；task仍0|有輸出層反應，不能說修正正確內在假設|
|C2-03|memory效果不可識別|devlibrary0；M1/M2每模型40/40request/logits完全一致|mechanical equivalence，不是有效工具庫無效證明|
|C2-04|support透過不保證hidden（非預期案例）|列舉reuse_14 support5/5、hidden6/10；`macrolab_trace_digest.txt`|1case，有限支援資訊；沒有因此重挑程式|
|D1-01|生成式可作有效baseline|206/240=85.83%、parserinvalid0|BF16greedy非思考8tokens；沒有可比prob指標，不作架構因果|
|D2-01|nested資料量point上升|100/300/900每域Accuracy40/46.67/48.75%，SharedNLL下降|s0；updates/exposures也變，非純資料量因果|
|D3-01|smoothing主要Raw改善|meanRawNLL1.396、Shared1.222、Accuracy48.19%|未掃epsilon，Shared未證優勢；舊新方向不同非已證overfit|
|D5-01|英文多語checkpoint較低point|80/240=33.33%；`baseline_completion.csv`|不是多語benchmark、不同checkpoint，不推論多語能力差|
|V-01|獨立重算一致|572groups maxdiff8.88e-16、B2mapping7,650／maxdiff4.44e-16、Macro320／4098typedcalls全oracle驗證|不是第三方或人類驗收|
|V-02|早期raw provenance較弱|部分C_T/C_gate/E_old/B2_E_old無逐筆metadatahash，依stage metadata與source/PID旁索引；P015/P027|原raw不回填；正式E_new直接hash，不把弱link冒充原本逐筆證據|
|H-01|人類覆核仍欠缺|AI讀45MCQA＋5Macro；human0|人類queue列近義/多解gold與citations，非blocking一般工程但限制學術主張|

未識別的解釋：reward／noise機制、pretraining exposure、原型與微調的Macrotransfer差異、題意歧義是否標註錯、未知域部署route品質。下一輪應用新protocol與保留集；本輪不追加seed／調門檻／挑checkpoint去取得正向結果。
