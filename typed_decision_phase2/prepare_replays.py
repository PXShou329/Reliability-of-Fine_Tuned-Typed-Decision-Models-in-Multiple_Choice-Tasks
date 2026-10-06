"""Select diagnostic cases transparently, lock before actual replay; no result substitution."""
import csv
from utils import *
def main():
    rows=list(csv.DictReader((ROOT/'results/order_robustness_cases.csv').open(encoding='utf-8-sig')))
    c=sorted([r for r in rows if r['purpose']=='E_new' and r['model']=='B1_grpo_ce_20261006' and r['any_answer_flip']=='1'],key=lambda r:digest(r['item_id']))[0]
    mcqa={}
    for model in ['base','B1_grpo_ce_20261006']:
        paths=sorted([p for p in (ROOT/'raw'/model/'B2_E_new').glob('*.json') if read(p)['item_id']==c['item_id']],key=lambda p:read(p)['rotation'])
        mcqa[model]=[str(p.relative_to(ROOT)) for p in paths]
    macro={m:'macrolab/raw/'+m+'/M1/reuse_13.json' for m in ['B1_grpo_ce_20261006','clef']}
    save(ROOT/'configs/diagnostic_replay.lock.json',{'time':now(),'state':'FROZEN','selection':'purposeful first SHA-ordered E_new R0 flipped case, all cyclic rotations in R0/Base; previously fixed-random MacroLab reuse_13 first3 actual calls',
        'MCQA':mcqa,'MacroLab':macro,'only_replay_no_new_method_no_formal_replacement':True})
    decision('P026 post-evaluation diagnostic replay: lock first SHA-ordered R0 flipped case/all cyclic views in Base andR0, plus first3 calls of each previously random-selected MacroLab reuse_13. Actual calls saved separately; do not alter formal outputs or methods. Analysis-only, not new confirmatory comparison.')
if __name__=='__main__':main()
