# 第二階段執行協定

唯一研究設計來源為 `protocol_source.md`，內容完整複製自使用者的 `Typed_Decision_Phase2_Experiment_Plan.md`；本文件是實際執行的索引，不更換研究題目。原 V1 frozen protocol 與 interactive toolmaking 封存保持不變。

原 T0 為 2026-10-06 01:32（UTC+8），原截止 2026-10-06 16:32；15:02 停止擴張，保留最後 90 分鐘驗收／報告。`deadline.json` 保存來源、絕對時間與未重設 T0 的證據。

## 資料與方法

G=2,700、C_T=150、C_gate=150、P=36、E_old=600、E_new=240。用途分離依 ID、標準化題幹及內容 hash 驗證；E_old 是已使用過的歷史資料，E_new 是本研究新保留的公開 validation，不能宣稱模型預訓練從未見過。

模型只收到 `state`、固定 `choice` 指令與原始選項文字。gold、答案解釋、ID／dataset metadata、solver 及評分結果不進模型呼叫。資料程序與 evaluator 按職責處理答案，不回流到方法開發。

R0=GRPO+CE、R1=CE-only，各 seeds 20261006／1729／2718，全部從同一原始 Laya 起點，固定第 4 epoch。shuffle、dropout、GRPO noise 分開 RNG；V1 R0/s0 因 RNG 耦合不完全相符而未重用。V1 最終權重只列歷史背景 `v1_ft_history`。原生／歷史 view 使用實際權重保存的 T；共同程序由各模型自己的 C_T fitting 給出 Shared-fit。

B2 固定完整循環位移與逆映射，排除明顯依賴字母／相對位置的題。B3 各 leave-one-dataset-out 與等資料量 all-domain 使用 1,800 題，校準只用來源領域；D4 按事前矩陣補至三 seeds。D 固定順序為生成式 baseline → 資料量 → smoothing → 額外 LODO seeds → 多語 checkpoint，啟動只看資源。

C1 在 C_gate 選擇 policy，實際 conditional calls；共駐、平衡 P smoke 與三個 timing blocks 通過後才凍結。E_new 的 checkpoints、校準器、policy、矩陣、置換、評分與雜湊鎖定於 `manifests/protocol.lock.json`。正式推論全部完成後，才統一讀 gold 計分。

## 驗收與恢復

每個 GPU child 順序執行。raw 每題原子保存；既有合法輸出不被重試替換。training 的 committed epoch、optimizer/RNG checkpoint 與 log hash 用於恢復，失敗嘗試不重複計入 item exposures。`manifests/*queue.json`、`gpu_active.json` 與 `events.jsonl` 保存實際 PID、命令及進度。

主要評分由 `scoring.py`／`analyze_mcqa.py` 完成；`audit_v2_results.py` 用另一路 scalar math 重算 Accuracy／NLL／Brier／ECE，不匯入主 scorer。10,000 次配對 bootstrap 按 dataset／題幹 cluster 抽樣；seed variability 與 conditional item uncertainty 分開。

MacroLab 是獨立生成的受限列表 DSL 探索，不更新權重。正式 hidden outputs 不回模型；所有 append／FINISH／KEEP／library 選擇皆來自真實 typed output。開發 library 只收實際受測模型通過 support 的程序；每個正式 episode 從相同 frozen library 開始。最終結果、AI 案例閱讀與待研究者覆核分開標記。

實際決定與偏差依 `decisions.md`，有效性 gates 依 `audit/`，模組狀態依 `results/completion_matrix.csv`（完成驗收後建立），引用依 `reports/citation_check_v2.md`。`reports/HANDOFF.md` 是最終恢復與交接入口。
