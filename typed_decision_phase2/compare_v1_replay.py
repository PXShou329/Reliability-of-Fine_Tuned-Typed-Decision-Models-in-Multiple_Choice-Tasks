"""Historical replay identity diagnostics only; never consumes E_new."""
import numpy as np
from utils import *
def main():
    checks=[]
    for old,new in [('laya_base','base'),('laya_finetuned','v1_ft_history'),('clef_flash','clef')]:
        prior={r['item_id']:r for p in (V1/'raw'/old/'items').glob('*.json') for r in [read(p)]}
        fresh={r['item_id']:r for p in (ROOT/'raw'/new/'E_old').glob('*.json') for r in [read(p)]}
        assert set(prior)==set(fresh) and len(prior)==600
        public_diff=[];pred_diff=[];maxlogit=0.;logit_diff_n=0
        for i in prior:
            a=prior[i];b=fresh[i]
            if a['public']!=b['public']:public_diff.append(i)
            z=np.asarray(a['response']['logits']);q=np.asarray(b['actual_output']['logits']);d=float(np.max(np.abs(z-q)));maxlogit=max(maxlogit,d);logit_diff_n+=d>0
            if np.argmax(z)!=np.argmax(q):pred_diff.append(i)
        checks.append({'V1_model':old,'V2_model':new,'n':600,'public_request_differ_n':len(public_diff),'prediction_differ_n':len(pred_diff),'max_raw_logit_difference':maxlogit,
            'nonidentical_logit_n':logit_diff_n,'public_difference_ids':public_diff,'prediction_difference_ids':pred_diff,'formal_E_new_not_read':True})
    save(ROOT/'audit/v1_historical_replay.json',{'time':now(),'status':'COMPLETED','checks':checks,'note':'identity diagnostics, not a new unseen test; no method changes based on outcomes'})
    print([{k:v for k,v in r.items() if not k.endswith('_ids')} for r in checks])
if __name__=='__main__':main()
