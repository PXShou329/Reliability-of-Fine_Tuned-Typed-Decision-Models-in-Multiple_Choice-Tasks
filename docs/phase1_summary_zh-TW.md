> Publication excerpt of a historical AI-assisted summary; case-specific examples and answer content are withheld. No new human-review certification is made.

# 給老師的研究摘要：Typed Decision MCQA

本次完成三資料集各200題、三條件各600題，共1,800筆真實本機推論；使用同一固定抽樣、選項轉換與evaluator。原工具建構實驗保持封存，不混入這次數據。距原T0約1.67小時，提前完成核心交付。

| 條件 | 正確／600 | Macro Accuracy | Macro Brier↓ | Macro NLL↓ | Macro ECE↓ |
|---|---:|---:|---:|---:|---:|
| Laya base | 242 | 40.33% | 0.7266 | 1.3791 | 0.1380 |
| Laya fine-tuned（T=5） | 308 | 51.33% | 0.6313 | 1.2135 | 0.0817 |
| Clef-Flash 9B（NF4） | 571 | 95.17% | 0.0802 | 0.1894 | 0.0490 |

**可支持的結論**：額外task-specific experience在本次同domain held-out確實改善Laya的答案排序。提升11.0百分點（配對bootstrap95%CI 7.0～15.0），三資料集都升；112題修正、46題regression。原始模型中Clef-Flash NF4最佳，微調Laya仍落後43.83百分點，不能說小模型已可取代大模型。

**最值得討論的現象**：微調的accuracy升，T=1 NLL卻從1.63升到3.01，Brier也變差。只用300個保留train題做T=5校準後，正式NLL才降到1.21。因此本次有效的是「訓練＋校準」的組合，微調本身不保證可信confidence。T=5到上界但解除界限的train-only改進很小，正式模型沒有事後更換溫度。

**成本與agent意義**：本機batch1中位延遲FT15.95ms、Clef187.56ms（約11.76倍）；Torch推論peak allocated約1.59／7.71GiB。挑最高confidence的50%題，FT accuracy65.67%、Clef99.00%；低成本小模型仍有相當錯答。這是選擇性準確率分析，尚未跑實際agent或cascade。

**負面結果與錯誤**：公開版省略逐題案例文字與答案解讀；全部適用的彙總負面結果保留於結果表，原始案例證據保持本機唯讀。

**最大限制**：單seed、公開validation、無法審計釋出前benchmark exposure；Clef使用NF4，Laya用FP32 weights/BF16 compute；沒有unseen-domain、多seed或新盲測。研究者疑點不是新ground truth；accuracy衡量原鍵一致率。12條一手引用已重新核對；OpenBookQA license資訊「來源未完整標示」。

**下一步問題**：在固定accuracy增益下，什麼訓練／校準方式能避免小模型過度尖銳的機率，並維持新domain的selective accuracy？下一次先做多seed與loss／校準拆解，再鎖定小模型升級門檻，在新blind樣本測端到端效果與成本。

可討論但未驗證的解釋：GRPO或CE造成logit尺度膨脹、過擬合、公開題記憶、單GPU訓練改動的作用。這些都不能由目前一次run識別。正式model heads沒有思維鏈，不宣稱已觀察模型內部推理過程。

[完整研究報告](preliminary_report_zh-TW.md) · [錯誤分析](error_analysis_zh-TW.md) · [findings ledger](findings_ledger.md) · [citation check](citation_check.md) · [交接與重現](../HANDOFF.md)
