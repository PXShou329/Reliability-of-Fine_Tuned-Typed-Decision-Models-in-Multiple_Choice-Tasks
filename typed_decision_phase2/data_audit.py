"""Independent source/mapping and token checks. No model forward or score selection."""
import itertools,re,unicodedata,collections
from datasets import load_from_disk
from transformers import AutoTokenizer
from laya.common import build_sequence
from utils import *
from schema import validate_public
def norm(s):return re.sub(r'\W+',' ',unicodedata.normalize('NFKC',s).casefold()).strip()
def main():
    data={x:read(ROOT/'data_public'/(x+'.json')) for x in ['G','P','C_T','C_gate','E_old','E_new']}
    indices={}
    for d,split in itertools.product(['arc','csqa','obqa'],['train','validation']):
        indices[d,split]={r['id']:r for r in load_from_disk(str(V1/'data/source'/d/split))}
    tok=AutoTokenizer.from_pretrained(str(V1/'models/laya/tokenizer'),local_files_only=True)
    cfg=read(V1/'models/laya/rl_agent_config.json');out={'time':now(),'independent_source_mapping':{},'tokenization':{},'collisions':[],'near_stem_rule':'NFKC casefold remove punctuation to words; semantic paraphrase coverage not guaranteed'}
    for purpose,rows in data.items():
        labels=read(ROOT/'evaluator_private'/(purpose+'.json'));stats=[]
        for r in rows:
            p=r['public'];validate_public(p);d,split,rid=r['item_id'].split('/',2);s=indices[d,split][rid]
            assert p['state']==s['question_stem' if d=='obqa' else 'question']
            assert list(p['questions']['answer']['criteria'].values())==s['choices']['text']
            assert labels[r['item_id']]['gold_index']==s['choices']['label'].index(s['answerKey'])
            q=p['questions']['answer'];seq,markers,opts,trunc=build_sequence(tok,p['state'],{'t':'choice','ins':q['instructions'],'crit':q['criteria']},512,192,return_stats=True,return_truncation_stats=True)
            assert len(markers)==len(q['criteria']);stats.append({'item_id':r['item_id'],'tokens':len(seq),'option_stats':opts,'truncation':trunc})
        save(ROOT/'audit'/('token_'+purpose+'.json'),stats)
        out['independent_source_mapping'][purpose]={'n':len(rows),'all_passed':True}
        out['tokenization'][purpose]={'max_tokens':max(x['tokens'] for x in stats),'option_count_matched':True,'detail':'audit/token_'+purpose+'.json'}
    for a,b in itertools.combinations(data,2):
        aa=collections.defaultdict(list);bb=collections.defaultdict(list)
        for r in data[a]:aa[norm(r['public']['state'])].append(r['item_id'])
        for r in data[b]:bb[norm(r['public']['state'])].append(r['item_id'])
        for n in aa.keys()&bb.keys():out['collisions'].append({'purposes':[a,b],'ids':aa[n]+bb[n],'rule':'normalized words'})
    # Deliberate private-field injections must fail before any model call.
    import copy
    injected=[]
    for where,key in [('top','gold_index'),('question','gold'),('question','explanation')]:
        p=copy.deepcopy(data['P'][0]['public']);(p if where=='top' else p['questions']['answer'])[key]='SENTINEL_PRIVATE'
        try:validate_public(p)
        except ValueError:injected.append(key)
        else:raise AssertionError('injection accepted')
    out['private_field_injections_rejected']=injected;out['status']='COMPLETED' if not out['collisions'] else 'BLOCKED'
    save(ROOT/'audit/data_isolation.json',out);print({k:v for k,v in out.items() if k not in ['tokenization']})
if __name__=='__main__':main()
