"""Resumable sequential local model work. Analysis/reporting follows this workflow."""
import os,sys,time,subprocess
from datetime import datetime,timezone
from utils import ROOT,read,save,event,now,T0,DEADLINE

def complete(path):
    return path.exists() and read(path).get('status')=='complete'

def stage(name,args,done=None):
    if done and done(): print('preserved completed',name,flush=True);return
    if datetime.now(timezone.utc)>=datetime.fromisoformat(DEADLINE): raise RuntimeError('Original 15-hour budget reached')
    status={'stage':name,'time':now(),'status':'running','args':args,'original_t0':T0,'budget_deadline':DEADLINE}
    save(ROOT/'workflow_checkpoint.json',status);event('workflow_stage_start',stage=name,args=args)
    log=ROOT/'workflow_logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as f:
        proc=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        status['pid']=proc.pid;save(ROOT/'workflow_checkpoint.json',status)
        code=proc.wait()
    if code:
        save(ROOT/'workflow_checkpoint.json',{**status,'status':'failed','exit_code':code,'time':now()})
        event('workflow_stage_failed',stage=name,exit_code=code)
        raise RuntimeError(f'{name} failed; preserve all data and inspect {log}')
    save(ROOT/'workflow_checkpoint.json',{**status,'status':'complete','exit_code':code,'time':now()});event('workflow_stage_complete',stage=name)
    print('complete',name,flush=True)

def main():
    print('Waiting for verified Clef weights; Laya pilot and training smoke are already preserved.',flush=True)
    while not (ROOT/'models/clef_flash/download_completed.json').exists():
        if datetime.now(timezone.utc)>=datetime.fromisoformat(DEADLINE): raise RuntimeError('Budget reached while downloading')
        time.sleep(5)
    stage('clef_pilot',['run_inference.py','--model','clef_flash','--pilot'],lambda:complete(ROOT/'pilot_raw/clef_flash/checkpoint.json'))
    stage('clef_adapter_verify',['verify_adapter.py','--model','clef_flash'],lambda:(ROOT/'adapter_verification/clef_flash.json').exists())
    stage('freeze',['freeze_protocol.py'],lambda:(ROOT/'protocol_freeze.json').exists())
    stage('laya_base_formal',['run_inference.py','--model','laya_base'],lambda:complete(ROOT/'raw/laya_base/checkpoint.json'))
    stage('clef_flash_formal',['run_inference.py','--model','clef_flash'],lambda:complete(ROOT/'raw/clef_flash/checkpoint.json'))
    stage('laya_training',['train_laya.py'],lambda:complete(ROOT/'checkpoints/training_status.json'))
    stage('laya_finetuned_formal',['run_inference.py','--model','laya_finetuned'],lambda:complete(ROOT/'raw/laya_finetuned/checkpoint.json'))
    stage('score',['evaluate_results.py'])
    save(ROOT/'workflow_checkpoint.json',{'status':'core_data_complete','time':now(),'remaining':'researcher inspection, replay, error analysis, verified citations and reports','original_t0':T0,'budget_deadline':DEADLINE})
    event('core_data_complete');print('Core data complete; analysis and reports remain.',flush=True)

if __name__=='__main__': main()
