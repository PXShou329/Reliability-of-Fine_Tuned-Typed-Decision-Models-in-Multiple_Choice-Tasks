"""Optional standard research plots derived from complete saved CSVs only."""
import csv,os
from utils import *
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def rows(name):return list(csv.DictReader((ROOT/'results'/name).open(encoding='utf-8-sig')))
def main():
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    colors=['#2166ac','#b2182b','#4d9221'];recipes=[('grpo_ce','0','GRPO+CE'),('ce','0','CE'),('ce','0.1','CE+smoothing')]
    a=rows('ablation_by_seed.csv');ax=axes[0,0]
    for j,(recipe,smooth,label) in enumerate(recipes):
        q=[r for r in a if r['purpose']=='E_new' and r['view']=='Shared_fit' and r['recipe']==recipe and r['smoothing']==smooth]
        ys=[100*float(r['accuracy']) for r in q];ax.scatter([j-.08,j,j+.08],ys,color=colors[j],s=35,zorder=3)
        ax.plot([j-.17,j+.17],[np.mean(ys)]*2,color=colors[j],lw=3)
    ax.axhline(100*94/240,color='gray',ls='--',label='Base39.17%');ax.set(xticks=range(3),xticklabels=[r[2] for r in recipes],ylabel='Accuracy (%)',ylim=(35,57),title='A  E_new240: all3 seeds, mean line');ax.legend(fontsize=8)
    ax=axes[0,1];summ=rows('recipe_seed_summary.csv');views=['Raw','Native_Historical','Shared_fit']
    for j,(recipe,smooth,label) in enumerate(recipes):
        vals=[float(next(r for r in summ if r['purpose']=='E_new' and r['view']==v and r['recipe']==recipe and r['smoothing']==smooth)['nll_mean']) for v in views]
        ax.plot(range(3),vals,'o-',color=colors[j],label=label)
    ax.set(xticks=range(3),xticklabels=['Raw','Native / Historical','Shared-fit'],ylabel='NLL (lower is better)',title='B  Calibration: 3-seed mean, no argmax change');ax.legend(fontsize=8)
    ax=axes[1,0];order=[r for r in rows('order_robustness.csv') if r['purpose']=='E_new' and r['dataset']=='macro']
    labels={'base':'Base','B1_grpo_ce_20261006':'GRPO s0','B1_ce_20261006':'CE s0','kev':'Kev','clef':'Clef'}
    for j,r in enumerate(order):
        x=float(r['mean_TV_to_original']);y=100*float(r['any_answer_flip']);ax.scatter(x,y,s=40);ax.annotate(labels[r['model']],(x,y),xytext=(5,5 if r['model']!='B1_ce_20261006' else -14),textcoords='offset points')
    ax.set(xlabel='Mean aligned TV drift (Shared-fit)',ylabel='Any answer flip (%)',xlim=(.025,.145),ylim=(0,75),title='C  E_new235: answer vs probability stability')
    ax=axes[1,1];transfer=[r for r in rows('transfer_results.csv') if r['purpose']=='E_new' and r['view']=='Shared_fit']
    for j,seed in enumerate([20261006,1729,2718]):
        q=[next(r for r in transfer if r['seed']==str(seed) and r['dataset']==d) for d in ['arc','csqa','obqa']]
        ax.plot(range(3),[100*float(r['accuracy_minus_matched']) for r in q],'o-',color=colors[j],label=str(seed))
    ax.axhline(0,color='gray',ls='--');ax.set(xticks=range(3),xticklabels=['ARC','CSQA','OBQA'],ylabel='LODO minus equal-data all-domain (pp)',title='D  Transfer: all3 targets x3 seeds, n80 /target');ax.legend(fontsize=8)
    ax=axes[2,0];cascade=[r for r in rows('cascade_metrics.csv') if r['dataset']=='macro'];lookup={r['condition']:r for r in cascade};names=['always_small','always_expert','confidence','random'];pretty=['Small','Expert','Confidence (bypass)','Random (coverage0)']
    vals=[float(lookup[n]['mean_latency_ms']) for n in names];ax.barh(range(4),vals,color=['#9ecae1','#6baed6','#3182bd','#08519c'])
    for j,n in enumerate(names):ax.text(vals[j]+2,j,f"{100*float(lookup[n]['accuracy']):.2f}%",va='center',fontsize=8)
    ax.axvline(vals[1]*.8,color='#b2182b',ls='--',label='20% latency target');ax.set(yticks=range(4),yticklabels=pretty,xlabel='Mean end-to-end latency (ms)',xlim=(0,240),title='E  Live240 /policy: bypass misses speed target');ax.invert_yaxis();ax.legend(fontsize=7,loc='lower right')
    ax=axes[2,1];macro=rows('macrolab_metrics.csv');conditions=[('B1_grpo_ce_20261006','M0'),('B1_grpo_ce_20261006','M1'),('B1_grpo_ce_20261006','M2'),('clef','M0'),('clef','M1'),('clef','M2'),('search','random'),('search','enumeration')]
    for j,split in enumerate(['reuse','novel_composition']):
        ys=[100*float(next(r for r in macro if r['model']==m and r['condition']==c and r['split']==split)['task_success']) for m,c in conditions]
        ax.bar(np.arange(8)+(j-.5)*.34,ys,width=.34,label=split.replace('_',' '))
    ax.set(xticks=range(8),xticklabels=['L M0','L M1','L M2','C M0','C M1','C M2','Rand','Enum'],ylabel='Task success (%)',ylim=(0,100),title='F  MacroLab: n20 /split /condition, empty library');ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Typed Decision Phase2: complete results, including negative controls',fontsize=14)
    out=ROOT/'reports/figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(out/('phase2_overview.'+ext),dpi=200)
    plt.close(fig)
    names=['ablation_by_seed.csv','recipe_seed_summary.csv','order_robustness.csv','transfer_results.csv','cascade_metrics.csv','macrolab_metrics.csv']
    save(out/'provenance.json',{'time':now(),'matplotlib':matplotlib.__version__,'source_SHA256':{n:file_hash(ROOT/'results'/n) for n in names},'note':'Descriptive display; no missing seeds/folds removed, no new fit, see tables for conditional paired intervals.'})
    print('overview png/pdf/svg saved')
if __name__=='__main__':main()
