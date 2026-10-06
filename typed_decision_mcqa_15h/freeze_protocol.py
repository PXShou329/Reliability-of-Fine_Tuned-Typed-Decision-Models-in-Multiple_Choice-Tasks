"""Freeze only after real train-only smoke tests, before any formal inference."""
from utils import ROOT,read,save,file_hash,digest,event,now,write
from metrics import self_test

def main():
    target=ROOT/'protocol_freeze.json'
    if target.exists(): print('Existing freeze preserved');return
    for m in ['laya_base','clef_flash']:
        c=read(ROOT/'pilot_raw'/m/'checkpoint.json')
        assert c['status']=='complete' and c['completed']==36
        assert all(r['passed'] for r in read(ROOT/'adapter_verification'/(m+'.json'))['checks'])
    assert read(ROOT/'training_pilot_summary.json')['passed']
    assert read(ROOT/'pipeline_validation.json')['formal_public_no_gold_or_auxiliary_hints']=='passed'
    audit=read(ROOT/'input_audit.json')
    assert audit['private_field_injection_test']=='passed'
    for p,s in audit['summary'].items():
        assert not s['state_truncated_n'] and not s['option_cap_exceeded_n'] and not s['collapsed_option_n']
    assert not list((ROOT/'raw').glob('*/items/*.json'))
    settings=read(ROOT/'configs/protocol.json');settings['status']='frozen'
    save(ROOT/'configs/protocol.json',settings)
    revisions=read(ROOT/'sources/revisions.json')
    model_manifest={'phase':'verified_preflight','time':now(),'models':{
        'laya_base':{'model_id':'convaiinnovations/laya','revision':revisions['laya'],'adapter':read(ROOT/'adapter_verification/laya_base.json')['adapter'],'nominal_parameters':'421M'},
        'clef_flash':{'model_id':'Cloudflare/clef-flash','revision':revisions['clef_flash'],'adapter':read(ROOT/'adapter_verification/clef_flash.json')['adapter'],'nominal_parameters':'9B'},
        'laya_finetuned':{'model_id':'local/laya-mcqa-local-20261006','status':'not_yet_trained','initialized_from_revision':revisions['laya'],'training_config':settings['training']}}}
    save(ROOT/'model_manifest_preflight.json',model_manifest);save(ROOT/'model_manifest.json',model_manifest)
    files=[ROOT/'protocol_source.md',ROOT/'configs/protocol.json',ROOT/'data/splits_manifest.json',ROOT/'dataset_manifest.json',ROOT/'environment_manifest.json',ROOT/'requirements.lock.txt',ROOT/'sources/revisions.json',ROOT/'evaluation_stem_audit.json',ROOT/'pipeline_validation.json',ROOT/'model_manifest_preflight.json',ROOT/'sources/runtime_laya_source_manifest.json']
    files+=list((ROOT/'data/ids').glob('*/*.json'))+list((ROOT/'data/public_eval').glob('*.json'))+list((ROOT/'evaluator_private/gold').glob('*.json'))
    files += [ROOT/name for name in ['utils.py','adapters.py','schema.py','prepare_data.py','metrics.py','run_inference.py','train_laya.py','evaluate_results.py','replay_results.py']]
    freeze={'freeze_time':now(),'protocol_hash':digest(settings),'checks':{'real_model_train_only_smoke':'passed','real_training_smoke':'passed',
            'public_input_truncation':'none','evaluator_analytic':self_test(),'no_formal_results_yet':True},
            'files_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in files},'revisions':read(ROOT/'sources/revisions.json')}
    save(target,freeze)
    write(ROOT/'protocol.md',f'''# Frozen Typed Decision MCQA protocol

Frozen at {freeze['freeze_time']}. Original T0 and deadline are unchanged. This document instantiates `protocol_source.md`; `configs/protocol.json` and `protocol_freeze.json` record exact settings and hashes.

Use ARC-Challenge, CommonsenseQA and OpenBookQA main. Each dataset contributes 200 fixed official validation questions, 900 disjoint gradient training items,100 train-only calibration items and12 independent pilot questions. Question-stem overlaps with any official validation set and duplicate training stems were removed before sampling. The same600 evaluation IDs and original option order apply to all models. Labels are remapped to A..Z by original position; the gold mapping is evaluator-only.

Primary models: original root `convaiinnovations/laya`, our locally trained four-epoch Laya and `Cloudflare/clef-flash` with its official joint head. Inference is local on RTX5080, batch1, three train-only warm-ups per process. Laya uses original512/head192 input budget and native base temperature, FP32 weights/BF16 autocast. Fine-tuning follows the pinned official GRPO+CE notebook adapted to one GPU, effective batch64, four fixed epochs; the final checkpoint is selected without validation access. Its choice temperature is fitted only on300 withheld official train items. Clef uses NF4 double quantization, BF16 compute/head; this precision difference limits architecture-only cost/accuracy claims. Real train-only smoke tests passed before this freeze.

Accuracy uses argmax with first-option tie-break. Brier is summed over classes, NLL natural-log with1e-12 floor, ECE has15 equal-width max-probability bins. Report dataset metrics and their equal-weight macro. Save unrounded raw logits/probabilities, full public request, prediction latency and run metadata. Loading time is separate; single-item latency includes tokenization/forward/decode and CUDA synchronization, excludes writes. NVIDIA samples are total device memory including other processes, not model-only peak. Torch allocated/reserved counters are distinct.

Random baseline is analytical uniform probability expectation, not fabricated inference. Secondary analyses: saved-logit T=1 scores for both Laya variants; paired dataset-stratified bootstrap10,000 repeats seed20261006 with repeated normalized stems clustered within dataset; confidence>=.9 wrong answers, base/FT gains/regressions and deterministic replay of first three evaluation items per dataset plus highest-confidence disagreement. The fixed600 items contain599 normalized unique question stems (two OpenBookQA items share a stem); keep both official IDs and cluster them for uncertainty. Interpretation annotations are researcher judgments, never new ground truth.

Do not change IDs, converter, prompts, metrics or checkpoint selection after inspecting results. Infrastructure/bug fixes require a deviation record and affected reruns. Do not substitute hand-written model answers. Test cache auto-built by the initial dataset loader is excluded from all input/training/evaluation paths (N007). Existing pretraining contamination cannot be audited from released checkpoints; internal experiment split/input isolation is verified, and this external limitation is reported.

Optional work cannot displace the core or final analysis/report. Old interactive toolmaking remains archived/paused and excluded. See `decisions.md` for pre-freeze implementation constraints and corrections.
''')
    manifest=read(ROOT/'run_manifest.json');manifest['status']='protocol_frozen';manifest['main_objective']='Complete revised Typed Decision MCQA experiment and research reports';manifest['protocol_hash']=freeze['protocol_hash'];save(ROOT/'run_manifest.json',manifest)
    event('protocol_frozen',protocol_hash=freeze['protocol_hash']);print(freeze['freeze_time'],freeze['protocol_hash'],flush=True)

if __name__=='__main__': main()
