"""Pinned official Kev API/checkpoint, BF16 unfused. Only allowlisted MCQA input."""
import sys,torch
from utils import *
from schema import validate_public
sys.path.insert(0,str(ROOT/'sources/kev_runtime'))
class KevAdapter:
    def __init__(self,path=None,settings=None):
        from kev.checkpoint import Checkpoint,LoadOptions
        self.ck=Checkpoint(str(path or ROOT/'models/kev'))
        rev=read(ROOT/'sources/revisions.json')
        original={'base':self.ck.meta.base,'base_revision':self.ck.meta.base_revision}
        assert original['base']=='Qwen/Qwen3.5-4B-Base'
        if original['base_revision']:assert original['base_revision']==rev['kev_base']['revision']
        assert (ROOT/'models/kev_base/download_manifest.json').exists()
        # Only resolve the pinned public base to verified local files; no algorithm changes.
        self.ck.meta.base=str(ROOT/'models/kev_base');self.ck.meta.base_revision=None
        self.native_T=float(self.ck.meta.temperature)
        opts=LoadOptions(dtype=torch.bfloat16,merge=True,attn='sdpa',backend='torch',cuda_graphs=False,fused=False,temperature=1.)
        self.tok,self.model=self.ck.load('cuda',opts)
        self.model.eval();assert self.model.head.temperature==1.
        self.metadata={'class':'Kev-4B','revision':rev['kev']['revision'],'base_revision_pinned':rev['kev_base']['revision'],'original_meta':original,
            'runtime_commit':rev['kev_runtime']['revision'],'backbone_precision':'BF16 merged LoRA','head_dtype':str(next(self.model.head.parameters()).dtype),
            'native_temperature':self.native_T,'backend':'official torch forward row path; sdpa; unfused; no CUDA graphs',
            'head_sha256':file_hash(ROOT/'models/kev/head.pt'),'adapter_sha256':file_hash(ROOT/'models/kev/adapter_model.safetensors'),
            'date_facts':False,'parameter_count':sum(p.numel() for p in self.model.parameters()),'transport_override':'verified pinned local base only'}
    def encode(self,p):
        from kev.api import SystemOneRequest,to_record
        validate_public(p);rec,meta=to_record(SystemOneRequest(**p))
        # Official to_record uses a constant dummy label; remove gold-like bookkeeping before model.
        enc=self.model.encode(self.tok,rec,strict=True,max_state=16384,max_branch=16384)
        assert meta[0]['keys']==list(p['questions']['answer']['criteria'])
        assert len(enc['opt_idx'][0])==len(meta[0]['keys']) and not enc['state_truncated']
        enc.pop('labels',None)
        return enc
    @torch.inference_mode()
    def predict(self,p):
        enc=self.encode(p);z=self.model(enc)[0].float();prob=(z/self.native_T).softmax(-1)
        return {'logits':z.cpu().tolist(),'probabilities':prob.cpu().tolist(),'temperature':self.native_T,
                'input_tokens':len(enc['ids']),'option_positions':enc['opt_idx'][0],'state_truncated':enc['state_truncated']}
