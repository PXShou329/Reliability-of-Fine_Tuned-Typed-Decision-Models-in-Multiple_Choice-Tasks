"""D1 pinned Qwen3.5-4B, text-only greedy 8-token nonthinking baseline."""
import re,torch
from utils import *
from schema import validate_public
class GenerativeAdapter:
    def __init__(self,path=None,settings=None):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        path=path or ROOT/'models/generative';assert (path/'download_manifest.json').exists()
        self.tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(str(path),dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        self.metadata={'class':'Qwen3.5-4B generative','revision':read(ROOT/'sources/revisions.json')['generative']['revision'],
            'precision':'BF16','decoding':'greedy do_sample=False max_new_tokens=8','enable_thinking':False,'text_only':True,
            'parser':'strip whitespace; fullmatch single capital legal A..letter; no guessing from prose','parameter_count':sum(p.numel() for p in self.model.parameters())}
    def encode(self,p):
        validate_public(p);q=p['questions']['answer'];content=p['state']+'\n'+'\n'.join(k+': '+v for k,v in q['criteria'].items())+'\nChoose the best answer. Output only the single option letter.'
        rendered=self.tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        # Official template's nonthinking path closes an empty reasoning block before generation.
        assert re.search(r'<think>\s*</think>',rendered) is not None
        return self.tok(rendered,return_tensors='pt'),rendered
    @torch.inference_mode()
    def predict(self,p):
        b,rendered=self.encode(p);b={k:v.to('cuda') for k,v in b.items()};n=b['input_ids'].shape[-1]
        out=self.model.generate(**b,max_new_tokens=8,do_sample=False,pad_token_id=self.tok.eos_token_id)
        ids=out[0,n:].cpu().tolist();text=self.tok.decode(ids,skip_special_tokens=True);raw_text=self.tok.decode(ids,skip_special_tokens=False);answer=text.strip()
        valid=re.fullmatch(r'[A-Z]',answer) is not None and answer in p['questions']['answer']['criteria']
        return {'text':text,'raw_generated_text':raw_text,'generated_ids':ids,'generated_tokens':len(ids),'input_tokens':n,'rendered_prompt':rendered,
                'valid_answer':valid,'prediction_index':ord(answer)-65 if valid else None,'reasoning_generated':'<think>' in raw_text or re.search(r"(?i)\b(?:let'?s (?:think|reason)|reasoning)\b",text) is not None}
