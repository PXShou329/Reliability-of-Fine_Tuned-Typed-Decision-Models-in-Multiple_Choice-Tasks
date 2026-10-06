"""Actual typed decisions on allowlisted MacroLab observations; no private imports."""
import sys,time,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils import *
from interpreter import PRIMITIVES,DESCRIPTIONS
class MacroModel:
    def __init__(self,model):
        from inference import load
        from protocol_guard import verify_lock
        verify_lock(model)
        self.adapter=load(model);self.name=model
    def decide(self,state,options):
        assert isinstance(state,str) and all(type(x)==str for x in options)
        p={'state':state,'questions':{'answer':{'type':'choice','instructions':'Choose the next action to construct the list transformation.','criteria':{chr(65+i):d for i,d in enumerate(options)}}}}
        torch.cuda.synchronize();start=time.perf_counter()
        if self.name=='clef':
            a=self.adapter;enc=a.mod.encode_record(a.processor.tokenizer,p,max_length=16384,processor=a.processor);batch=a.mod.collate_records([enc],a.processor.tokenizer.pad_token_id,'cuda')
            with torch.inference_mode():z=a.model(batch)[0][0].float();prob=z.softmax(-1)
            count=len(enc.input_ids);truncated=False
        else:
            from laya.common import build_sequence,QTYPES,collate_items
            a=self.adapter;q=p['questions']['answer'];seq,markers,stats,trunc=build_sequence(a.tok,state,{'t':'choice','ins':q['instructions'],'crit':q['criteria']},1024,256,return_stats=True,return_truncation_stats=True)
            assert not trunc['truncated'] and stats['tokens_per_option'] is None and stats['options_distinct']==len(options),'MacroLab truncation gate failed'
            item={'ids':seq,'markers':markers,'qtype':QTYPES['choice']};batch=collate_items([[item]],a.tok.pad_token_id)
            batch={k:v.to('cuda') for k,v in batch.items() if k in ['input_ids','attention_mask','marker_pos','marker_mask','qtype']}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):logits,_=a.model(**batch)
            z=logits[0,:len(options)].float();prob=z.softmax(-1);count=len(seq);truncated=trunc['truncated']
        answer=int(prob.argmax());torch.cuda.synchronize()
        return {'request':p,'typed_output':{'type':'choice','choice':chr(65+answer),'probabilities':{chr(65+i):float(v) for i,v in enumerate(prob.cpu().tolist())}},
                'raw_logits':z.cpu().tolist(),'selected_index':answer,'input_tokens':count,'truncated':truncated,'latency_ms':(time.perf_counter()-start)*1000}
