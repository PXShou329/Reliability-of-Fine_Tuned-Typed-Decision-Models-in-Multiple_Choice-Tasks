"""Public-input-only token audit and label isolation checks; no model predictions."""
from transformers import AutoTokenizer
from laya.agent import _fix_tokenizer_config
from laya.common import build_sequence,render_options
from utils import ROOT,read,save,digest
from schema import validate_public

def main():
    path=ROOT/'models/laya';_fix_tokenizer_config(str(path));tok=AutoTokenizer.from_pretrained(path/'tokenizer')
    cfg=read(ROOT/'configs/protocol.json');rows=[]
    for purpose in ['pilot','training','calibration','public_eval']:
        for d in ['arc','csqa','obqa']:
            for r in read(ROOT/'data'/purpose/(d+'.json')):
                p=r['public'];validate_public(p);q=p['questions']['answer'];internal={'t':'choice','ins':q['instructions'],'crit':q['criteria']}
                ids,markers,opts,trunc=build_sequence(tok,p['state'],internal,cfg['max_len'],cfg['head_max_len'],return_stats=True,return_truncation_stats=True)
                raw_option_lengths=[len(tok(' '+o,add_special_tokens=False)['input_ids']) for o in render_options(internal)]
                assert len(markers)==len(q['criteria']) and len(ids)<=cfg['max_len']
                rows.append({'purpose':purpose,'dataset':d,'item_id':r['item_id'],'request_hash':digest(p),'input_tokens':len(ids),
                             'option_stats':opts,'truncation':trunc,'raw_option_lengths':raw_option_lengths,'option_exceeds_native_48_token_cap':any(n>48 for n in raw_option_lengths)})
    # Explicit private-field injection must be rejected, even if its value looks innocuous.
    p=read(ROOT/'data/public_eval/arc.json')[0]['public']
    for key in ['gold_index','answerKey','evaluator','test_answer']:
        try: validate_public({**p,key:'injection'})
        except ValueError: pass
        else: raise AssertionError('private field accepted')
    summary={purpose:{'n':sum(r['purpose']==purpose for r in rows),
                       'max_input_tokens':max(r['input_tokens'] for r in rows if r['purpose']==purpose),
                       'state_truncated_n':sum(r['truncation']['truncated'] for r in rows if r['purpose']==purpose),
                       'option_cap_exceeded_n':sum(r['option_exceeds_native_48_token_cap'] for r in rows if r['purpose']==purpose),
                       'collapsed_option_n':sum(r['option_stats']['options_distinct']<r['option_stats']['options'] for r in rows if r['purpose']==purpose)}
             for purpose in ['pilot','training','calibration','public_eval']}
    save(ROOT/'input_audit.json',{'summary':summary,'private_field_injection_test':'passed','rows':rows})
    print(summary,flush=True)

if __name__=='__main__': main()
