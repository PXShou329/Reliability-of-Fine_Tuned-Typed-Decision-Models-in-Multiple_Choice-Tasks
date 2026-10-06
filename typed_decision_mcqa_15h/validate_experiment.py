"""Evaluator-side full selected-row mapping/label/leakage validation."""
from datasets import load_from_disk
from utils import ROOT,read,save,digest
from schema import validate_public
from metrics import score,aggregate

def main():
    checked=0;ids={p:set() for p in ['training','calibration','pilot','public_eval']};fingerprints={p:set() for p in ids}
    from prepare_data import question_fingerprint
    for d in ['arc','csqa','obqa']:
        sources={s:{r['id']:r for r in load_from_disk(str(ROOT/'data/source'/d/s))} for s in ['train','validation']}
        private=read(ROOT/'evaluator_private/gold'/(d+'.json'))
        for purpose in ids:
            split='validation' if purpose=='public_eval' else 'train'
            for r in read(ROOT/'data'/purpose/(d+'.json')):
                original_id=r['item_id'].split('/',2)[2];source=sources[split][original_id]
                validate_public(r['public'])
                assert r['public']['state']==source['question_stem' if d=='obqa' else 'question']
                criteria=r['public']['questions']['answer']['criteria']
                assert list(criteria.values())==list(source['choices']['text'])
                labels=list(source['choices']['label']);by_label=dict(zip(labels,source['choices']['text']))
                if purpose=='public_eval':
                    assert 'gold_index' not in r and set(r)=={'item_id','dataset','split','public'}
                    gi=private[r['item_id']]['gold_index']
                    assert private[r['item_id']]['original_option_labels']==labels
                    assert private[r['item_id']]['original_answer_label']==source['answerKey']
                else: gi=r['gold_index']
                # Independent dictionary lookup by original label + positional round trip,
                # rather than reusing the converter to generate the expected gold index.
                assert labels[gi]==source['answerKey']
                assert criteria[list(criteria)[gi]]==by_label[source['answerKey']]
                ids[purpose].add(r['item_id']);fingerprints[purpose].add(question_fingerprint(r['public']['state']));checked+=1
    for i,p in enumerate(ids):
        for q in list(ids)[i+1:]: assert not ids[p]&ids[q] and not fingerprints[p]&fingerprints[q]
    # Distinct independent hand-calculated calibration/proper-score cases.
    a=score([.8,.2],0);b=score([.2,.8],0)
    assert abs(a['brier']-.08)<1e-12 and abs(b['brier']-1.28)<1e-12
    a['latency_seconds']=b['latency_seconds']=0
    m=aggregate([a,b]);assert abs(m['ece']-.3)<1e-12 and abs(m['accuracy']-.5)<1e-12
    result={'selected_rows_checked':checked,'source_to_canonical_question_option_order_gold_mapping':'passed',
            'all_four_purposes_ids_and_stems_disjoint':'passed','evaluator_probabilistic_analytic_additional':'passed',
            'formal_public_no_gold_or_auxiliary_hints':'passed','mapping_reference':'direct original records, independent label dictionary lookup and positional round trip',
            'purpose_counts':{p:len(v) for p,v in ids.items()}}
    save(ROOT/'pipeline_validation.json',result);print(result,flush=True)

if __name__=='__main__': main()
