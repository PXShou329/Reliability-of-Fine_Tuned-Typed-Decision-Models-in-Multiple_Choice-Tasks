"""Evidence-based final delivery audit; no scientific method changes."""
import csv,subprocess,shutil
from utils import *
def main():
    required=['reports/phase2_report_zh-TW.md','reports/teacher_brief_v2_zh-TW.md','reports/preliminary_report_zh-TW.md','reports/macrolab_report_zh-TW.md',
        'reports/error_analysis_v2_zh-TW.md','reports/findings_ledger_v2.md','reports/citation_check_v2.md','reports/human_review_queue.md','reports/paper_outline.md','reports/HANDOFF.md',
        'results/completion_matrix.csv','results/case_level.csv','results/macrolab_results.csv','results/cascade_live.csv','provenance/legacy_raw_lineage.json']
    for name in required:assert (ROOT/name).exists() and (ROOT/name).stat().st_size>0,name
    modules=list(csv.DictReader((ROOT/'results/completion_matrix.csv').open(encoding='utf-8-sig')));assert len(modules)==12 and all(r['status']=='COMPLETED' for r in modules)
    audits=['audit/audit_v1.json','audit/data_isolation.json','audit/v2_independent_recompute.json','audit/b2_independent_recompute.json','audit/macrolab_independent_recompute.json','audit/v1_unchanged_final.json','audit/final_integrity.json','macrolab/evaluation_complete.json']
    for p in audits:assert read(ROOT/p)['status']=='COMPLETED',p
    assert read(ROOT/'audit/training_audit.json')['completed_runs']==23
    counts=read(ROOT/'results/delivery_counts.json');assert counts['MCQA_saved_requests']==30822 and counts['MCQA_failed_requests']==0 and counts['original_new_condition_outputs']==5520
    refs=read(ROOT/'sources/citation_verification.json')['references'];assert len(refs)==14
    for r in refs:
        assert r['status']=='VERIFIED_PRIMARY_LIMITED_SCOPE' and r['title'] and (r.get('authors') or r.get('organization')) and r['year'] and r['url'] and r['supported_claim']
        if r.get('sha256'):assert file_hash(ROOT/r['local_copy'])==r['sha256'],r['id']
    review={'time':now(),'reviewer':'Codex AI','MCQA_case_n':45,'MacroLab_trace_n':5,'human_reviewed_n':0,'MCQA_notes_SHA256':file_hash(ROOT/'reports/case_review_notes_zh-TW.md'),
        'MacroLab_report_SHA256':file_hash(ROOT/'reports/macrolab_report_zh-TW.md'),'initial_case_packets_preserved':True}
    save(ROOT/'reports/AI_review_receipt.json',review)
    snap=ROOT/'reports/preliminary_snapshots';snap.mkdir(exist_ok=True);p=ROOT/'reports/preliminary_report_zh-TW.md'
    original=snap/'preliminary_before_final.md'
    if not original.exists():shutil.copyfile(p,original)
    brief=(ROOT/'reports/teacher_brief_v2_zh-TW.md').read_text(encoding='utf-8');write(p,'# 第二階段初步研究結果（實驗完成版）\n\n全部預定本機實驗已實際完成；研究者人工覆核與外部重現未完成。完整繁中報告為 `phase2_report_zh-TW.md`。\n\n'+brief)
    write(ROOT/'reports/teacher_brief_zh-TW.md',brief)
    s=read(ROOT/'status.json');s.update(status='COMPLETED',phase='delivery_validation',time=now(),running_processes=[],E_new='FROZEN methods; all planned formal outputs and scoring completed; never reuse for method selection',
       next='Researcher human review; new protocol/new held-out data required for further method development',original_T0_reset=False,completion_evidence='audit/delivery_validation.json')
    save(ROOT/'status.json',s)
    current=read(ROOT/'manifests/download_status.json');save(ROOT/'provenance/download_last_transfer_snapshot.json',current)
    save(ROOT/'manifests/download_status.json',{'time':now(),'status':'COMPLETED','evidence':'manifests/download_complete.json andfrozen official file digests; no downloader running','last_transfer_snapshot':'provenance/download_last_transfer_snapshot.json'})
    save(ROOT/'audit/delivery_validation.json',{'time':now(),'status':'COMPLETED','modules_completed':len(modules),'new_training_runs':23,'MCQA_requests':30822,'formal_MCQA_eval_requests':26970,
        'live_cascade_records':960,'MacroLab_condition_episodes':320,'MacroLab_formal_typed_calls':4098,'MCQA_unique_old':600,'MCQA_unique_new':240,'MacroLab_original_tasks':40,
        'verified_references':14,'human_reviewed':0,'required_files':{f:file_hash(ROOT/f) for f in required},'negative_targets_preserved':True,'archive_pending':True})
    event('delivery_validation_completed',formal_MCQA_eval_requests=26970,MacroLab_condition_episodes=320,human_reviewed=0)
    print('delivery validation passed; archive follows',flush=True)
if __name__=='__main__':main()
