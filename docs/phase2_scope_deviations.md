# 第二階段偏差與有效性邊界

完整按時間的決定保存在 `decisions.md`／`events.jsonl`。目前各修正均在 E_new 開封前，沒有依正式新分數改方法。

- V1 R0/s0 不重用於受控 B1：原 noise 與 dropout RNG 耦合，不符合新協定的分離要求。六個 B1 都從原 Base 重新訓練，V1 FT 另列歷史背景 reused。
- Native/Historical：新訓練保留實際未重新校準的繼承溫度；只有 V1 FT 背景保留原 T=5。主要機率比較使用各權重以同 C_T 程序得到的 Shared-fit。P010 修正先前文字解讀，未變更訓練輸出。
- 環境：新 V2 venv 唯讀引用 V1 套件；新增套件只裝在 V2 overlay，不修改 system Python／driver／CUDA。不是完全獨立拷貝的環境，重現需保留所引用的 V1 相依套件。Kev 官方版本範圍與本機不同，使用實際通過測試的 unfused 路徑，未默默全面升級。
- Windows 路徑與缺套件故障：有實際失敗命令與 traceback。修正僅改善啟動／logging／imports，沒有人工補模型候選或刪除合法輸出。失敗訓練嘗試不計入 committed item exposures。
- B2：E_old 固定 120 題中 118 題、E_new 240 題中 235 題通過事前位置語意安全規則。排除依公開題意，不依模型分數；各模型用相同安全子集。循環位移不是全排列穩健性的保證。
- MacroLab：library 最多取每個開發 episode 的第一個 support 通過候選，去重後最多 12 tools；bounded enumeration 只用公開 demos 篩選固定次序候選，同為最多四次 verifier，但實際 CPU preprocessing 另外計時，不主張等 FLOPs。這些規則在任何實際 MacroLab model episode 之前固定。
- 共通邊界：公開 validation 的新保留只相對本研究未使用，不能保證模型預訓練未見；題幹 hash 查重不等於語意近重複全部排除；所有案例閱讀目前是 AI 輔助，不冒稱研究者已親自覆核。

正式 freeze 後若有變更，須另追加受影響 run、原因、重跑範圍及結果地位；不可改写以上歷史。

P023（2026-10-06 05:29 UTC+8）：G2 後、首次 E_new 計分前，發現 §7.3 已規定的位置直方圖／gold-position 附表尚未實作，補上獨立 post-production evaluator `order_positions.py`，code hash 固定於 `manifests/order_supplement.lock.json`。凍結的主評分、取樣、模型方法、checkpoint、校準／policy 與 raw 全部不變，沒有受影響輸出需重跑。附加的固定 R0／CE／Base 位置配對對比標為補充分析，不包裝成新模型條件；研究者可依記錄審查這项工程遗漏。

P025：正式後交付表glob曾把960livecascade列算成MCQA原題new count，已修正為5,520；MCQA評估19,320原題與7,650置換不變。Citation renderer補發佈組織欄，停止它改寫frozen environment artifact。只報表修正，不改raw、評分或方法。

P026：少量實際診斷replay另存`audit/diagnostic_replays.json`與selectionlock；不替換正式輸出、不重挑方法、不增加確認性樣本。

P027：末端輔助auditor誤假設每筆歷史raw皆有新metadata欄，兩次有界schema修復依完整inventory辨識pre-P015弱stage linkage，另留rawSHA/PID/source/metadata索引。原row不回填；E_new/B2_E_new仍要求直接hash。初始sourcearchiveCRLF正規化LF可exact復原原執行dac17866SHA，另存byteartifact。故障logs保留；不是模型分數或formalmethod修正。最終狀態以`audit/final_integrity.json`與HANDOFF為準。

P028/P029：診斷首次啟動0forward前失敗，repair1後14實際outputs全部bit-identical，shellstderr wrapper非零與artifact成功分別保存。報告字形以cache內pinnedOpenCC0.1.7正規化為繁中，保存前文snapshots；不改frozenenv／source specification／raw／gold／candidate／CSV評分。Completion matrix補condition-unit，C1為4policies、C2為8model/control條件，避免0個pure-model欄位被誤解為沒有執行。上述皆交付／輔助診斷層，無formal方法需重跑。
