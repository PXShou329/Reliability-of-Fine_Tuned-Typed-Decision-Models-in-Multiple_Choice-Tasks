# Typed Decision 第二階段研究摘要

**本輪實際完成全部預定對照，但沒有 GRPO 額外優勢或有效 cascade 加速的證據。** 同時發現：答案順序穩定與機率順序穩定可呈相反變化；高 MCQA 分數沒有轉成這個固定 DSL 的工具建構成功。

問題是微調、機率校準、選項順序與跨域transfer是否足以支援低成本結構化決策。V1原始證據獨立驗收與1,800筆歷史logits重播一致，原檔不變。V2做23個新訓練run、29個模型／checkpoint條件；GRPO+CE/CE-only各3seeds、完整三fold×三seedLODO及等量all-domain、五模型順序測試、真實livecascade，以及獨立MacroLab。舊600題、新保留240題（各域80）分開；新集方法開封前全鎖定。

|新保留集240題|Accuracy|共同校準NLL /Brier|
|---|---:|---|
|Laya Base|39.17%|1.3419 /.7105|
|GRPO+CE，三seed mean±SD|48.61±2.37%|1.2251 /.6432|
|CE-only，三seed mean±SD|49.72±.87%|1.2076 /.6361|
|Kev-4B|81.67%|.4565 /.2483|
|Clef-Flash NF4|92.50%|.1654 /.0903|
|Qwen3.5-4B生成式|85.83%|不補造機率指標|

五項有證據的發現：

1. **微調有效，但 GRPO 優勢未確立。** 相對Base，兩recipe三seeds皆增準確率，逐seed配對CI下界正。GRPO−CE分別−.83、−3.33、+.83pp，區間皆跨零，不能說優勝或等效。
2. **正確率增加不等於機率可靠。** GRPO/CE平均RawNLL2.743/2.717，Base1.570；Shared降至1.225/1.208，argmax不變。s0 GRPO Raw≥.9接受164題錯82；Shared≥.9接受0題，不能以0高信心錯題聲稱可用自動採納。
3. **穩健性與transfer有限。** 新集235等義題，s0FT翻轉約51%低於Base63%，但SharedTV漂移反而增加，兩配對區間皆正。OBQA三seedLODO相對等量all-domain低13.75/12.50/16.25pp、各CI低於0；ARC/CSQA多數不精確。
4. **真實cascade未達工程目標。** C_gate沒有滿足≤2pp loss且至少30auto樣本的非零門檻，policy=bypass。四條件各240次實呼叫，expert與confidence皆92.5%；平均192.918→192.159ms僅減少.39%，未達20%。不可降低gate或重挑threshold包裝成功。
5. **MacroLab提供有效負面對照。** 40原任務×8條件＝320episodes／4,098正式typedcalls；Laya與Clef M0/M1/M2皆0/40成功，列舉30/40。回饋改變許多候選但沒成功；開發庫0使memory效果不可識別。另有1例support5/5卻hidden6/10，不把support透過當全域證明。

主要限制：新集240／每域80、三seeds；公開保留題不保證模型預訓練未見；不同模型規模／訓練／精度／runtime不能作架構因果推論；Kev官方版本範圍之外僅本機unfusedpath透過；Macro有限DSL／prompt／budget及空library；45MCQA+5Macro案例為AI閱讀，研究者本人覆核0。14引用僅在重新核對的一手摘要／metadata／pinned官方檔案範圍內使用。

下一步最值得做：先人工核對近義／多解gold與五Macrotrace，再用**新資料、新protocol**測試order一致性是否能提高非零coverage的routing品質；Macro方向先驗primitive／two-step理解與BasevsMCQAFTtransfer，避免直接加大工具庫。這輪不再用E_new或hidden來調方法。

完整證據：`phase2_report_zh-TW.md`、`macrolab_report_zh-TW.md`、`findings_ledger_v2.md`；逐seed／fold／case表與raw在`results/`、`raw/`及`macrolab/raw/`。所有負面結果、失敗嘗試、偏差與待人工事項皆保留。
