"""Local real Typed Decision inference. Strictly public records only."""
import os,time,importlib.util
from pathlib import Path
import torch
from utils import ROOT,read,file_hash
from schema import validate_public

class LayaAdapter:
    def __init__(self,path,settings=None):
        from laya.agent import load
        os.environ['LAYA_CUDA_AMP']='bf16'
        self.agent=load(str(path),device='cuda',fast=False,compile=False)
        if self.agent.device.type!='cuda': raise RuntimeError('GPU required; CPU fallback is not authorized by frozen protocol')
        self.model=self.agent.model;self.tok=self.agent.tok;self.cfg=self.agent.cfg.copy()
        if settings:
            for k in ['max_len','head_max_len']: self.cfg[k]=settings[k]
        self.metadata={'class':'Laya','weights_dtype':str(next(self.model.parameters()).dtype),
                       'compute_dtype':str(self.agent.dtype),'backend':'sdpa eager','device':'cuda:0',
                       'parameter_count':sum(p.numel() for p in self.model.parameters()),
                       'max_len':self.cfg['max_len'],'head_max_len':self.cfg['head_max_len'],
                       'native_temperatures':self.agent.temperature,'temperature_by_options':self.agent.temperature_by_options}

    def encode(self,p):
        from laya.common import build_sequence,QTYPES
        validate_public(p);q=p['questions']['answer']
        internal={'t':'choice','ins':q['instructions'],'crit':q['criteria']}
        seq,markers,option_stats,truncation=build_sequence(self.tok,p['state'],internal,self.cfg['max_len'],self.cfg['head_max_len'],
                                                         return_stats=True,return_truncation_stats=True)
        if len(markers)!=len(q['criteria']): raise ValueError('option markers mismatch')
        return {'ids':seq,'markers':markers,'qtype':QTYPES['choice'],'option_stats':option_stats,'truncation':truncation}

    @torch.inference_mode()
    def predict(self,p):
        from laya.common import collate_items,temp_bucket
        item=self.encode(p);b=collate_items([[item]],self.tok.pad_token_id)
        b={k:v.to('cuda') for k,v in b.items() if k in ['input_ids','attention_mask','marker_pos','marker_mask','qtype','position_ids','option_ids']}
        with torch.autocast('cuda',dtype=self.agent.dtype,enabled=self.agent.amp_enabled):
            logits,act=self.model(**b)
        k=len(item['markers']);z=logits[0,:k].float()
        temp=self.agent.temperature_by_options.get(temp_bucket(0,k),self.agent.temperature[0])
        prob=(z/temp).softmax(-1)
        return {'logits':z.cpu().tolist(),'probabilities':prob.cpu().tolist(),'temperature':float(temp),
                'input_tokens':len(item['ids']),'markers':item['markers'],'option_stats':item['option_stats'],'truncation':item['truncation'],
                'act_probabilities':act[0].float().softmax(-1).cpu().tolist()}

class ClefAdapter:
    def __init__(self,path,settings=None):
        from transformers import BitsAndBytesConfig
        p=Path(path);spec=importlib.util.spec_from_file_location('clef_official_pinned',p/'joint_schema_model.py')
        mod=importlib.util.module_from_spec(spec)
        import sys;sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        self.mod=mod
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        self.model,self.processor=mod.load_release_model(p,device='cuda',dtype=torch.bfloat16,quantization_config=quant)
        self.max_length=(settings or {}).get('clef_max_length',16384)
        self.metadata={'class':'Clef-Flash','device':'cuda:0','backbone_precision':'NF4 double quantization',
                       'compute_dtype':'torch.bfloat16','decision_head_dtype':str(next(self.model.head.parameters()).dtype),
                       'backend':'official joint_schema_model.py','max_length':self.max_length,
                       'official_code_sha256':file_hash(p/'joint_schema_model.py'),
                       'parameter_count_representation_dependent':sum(p.numel() for p in self.model.parameters()),
                       'model_memory_footprint_bytes':self.model.language_model.get_memory_footprint()}

    def encode(self,p):
        validate_public(p)
        return self.mod.encode_record(self.processor.tokenizer,p,max_length=self.max_length,processor=self.processor)

    @torch.inference_mode()
    def predict(self,p):
        encoded=self.encode(p)
        b=self.mod.collate_records([encoded],self.processor.tokenizer.pad_token_id,'cuda')
        outputs=self.model(b)[0]
        assert len(outputs)==1 and list(encoded.questions[0].option_ids)==list(p['questions']['answer']['criteria'])
        z=outputs[0].float();prob=z.softmax(-1)
        return {'logits':z.cpu().tolist(),'probabilities':prob.cpu().tolist(),'temperature':1.,
                'input_tokens':len(encoded.input_ids),'option_ids':encoded.questions[0].option_ids}

def load_adapter(name,path,settings=None):
    if name.startswith('laya'): return LayaAdapter(path,settings)
    if name=='clef_flash': return ClefAdapter(path,settings)
    raise ValueError(name)
