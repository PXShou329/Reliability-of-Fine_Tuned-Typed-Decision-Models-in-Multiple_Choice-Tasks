"""Compact reviewer view of actual raw-derived cases; no model input or method change."""
from utils import *
def main():
    packet=read(ROOT/'reports/case_review_inputs.json');lines=[]
    for i,c in enumerate(packet['cases']):
        lines.append(f"{i+1}. {c['item_id']} [{','.join(c['selection'])}]")
        lines.append(c['public']['state'])
        criteria=c['public']['questions']['answer']['criteria']
        labels=list(criteria);lines.append(' | '.join(f'{k}: {v}' for k,v in criteria.items()))
        lines.append('gold='+labels[c['gold']['gold_index']])
        for model,out in c['outputs'].items():
            v=out['views'];raw=v.get('Raw',v.get('Generated',{}));shared=v.get('Shared_fit',{})
            pred=raw.get('prediction_index','')
            label=labels[int(pred)] if pred!='' else 'INVALID'
            conf=raw.get('confidence','')
            sc=shared.get('confidence','')
            lines.append(f"{model}: pred={label} correct={raw.get('correct')} confRaw={conf} confShared={sc}")
        lines.append('')
    write(ROOT/'reports/case_review_digest.txt','\n'.join(lines))
    print('review digest saved',len(packet['cases']))
if __name__=='__main__':main()
