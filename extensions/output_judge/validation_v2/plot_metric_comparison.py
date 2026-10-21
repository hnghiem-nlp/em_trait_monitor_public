"""Compare activation and output changes with independent-seed SD, by trait."""
import csv
import json
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs'/json.loads((HERE/'manifest.json').read_text())['digest'][:16]/'combined'
TRAITS=['honesty','helpfulness','harmlessness','power_seeking','corrigibility','sycophancy','confidence']
COLORS={'42':'#1674a5','123':'#d57820','789':'#8d4cab'}

def main():
    with (OUT/'model_metrics.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['repeat']=='1' and r['step']=='126' and r['suite']=='original' and r['seed'] in COLORS]
    fig,axes=plt.subplots(7,2,figsize=(12,18))
    export=[]
    for i,trait in enumerate(TRAITS):
        subset=[r for r in rows if r['metric']==trait]
        ns=[int(r['paired_n']) for r in subset]
        for j,field in enumerate(['activation_drift_percent','paired_change_from_base']):
            ax=axes[i,j]
            for seed,color in COLORS.items():
                a=sorted([r for r in subset if r['seed']==seed],key=lambda r:int(r['fraction']))
                ax.plot([int(r['fraction']) for r in a],[float(r[field]) for r in a],color=color,alpha=.65,lw=1,marker='o',ms=3)
            for frac in [0,5,10,25,50,75,100]:
                vals=[float(r[field]) for r in subset if int(r['fraction'])==frac]
                avg=st.mean(vals);sd=st.stdev(vals) if len(vals)>1 else None
                ax.errorbar(frac,avg,yerr=sd,fmt='ko',markersize=3,capsize=3,lw=1.2)
                export.append(dict(trait=trait,measurement=field,fraction=frac,n_seeds=len(vals),mean=avg,sd=sd))
            ax.axhline(0,color='#8b929c',lw=.8,ls='--');ax.grid(axis='y',alpha=.15)
            ax.set_xlim(-4,104);ax.set_xticks([0,5,10,25,50,75,100]);ax.tick_params(axis='x',labelsize=8,rotation=45)
            ax.spines[['top','right']].set_visible(False)
            ax.set_title(trait.replace('_',' ').title()+(f' · {min(ns)}–{max(ns)} paired questions/model' if j else ''),loc='left',fontsize=11,fontweight='bold')
            ax.set_ylabel('Activation change (%)' if j==0 else 'Answer-score change (1–5 scale)',fontsize=9)
            if i==6: ax.set_xlabel('Harmful training data (%)')
    fig.suptitle('Activation vs. judged answers — seven metrics',x=.08,ha='left',fontsize=19,fontweight='bold',y=.99)
    fig.text(.08,.968,'Left: internal trait projection vs. base. Right: judged answer score vs. base. Separate y-scales.',fontsize=11)
    fig.text(.08,.949,'Seeds at 0 / 5 / 10 / 25 / 50 / 75 / 100%: 2 / 1 / 1 / 2 / 3 / 1 / 3. Repeat 1 only.',fontsize=10)
    fig.legend(handles=[Line2D([0],[0],color=c,marker='o',label=f'Seed {s}') for s,c in COLORS.items()]+[Line2D([0],[0],color='black',marker='o',ls='',label='Mean ±1 sample SD')],loc='upper center',bbox_to_anchor=(.5,.939),ncol=4,frameon=False)
    fig.text(.08,.015,'No SD bar for one seed; a zero SD may also be invisible. SD is not a confidence interval.\nHigher power seeking / sycophancy means more of that trait; confidence is not calibration.\nCorrigibility has very few eligible questions. Eligibility can differ across conditions; lines connect measured points.',fontsize=10,color='#475467')
    fig.subplots_adjust(top=.895,bottom=.085,left=.09,right=.98,hspace=.63,wspace=.28)
    fig.savefig(OUT/'activation_vs_answers_by_metric.png',dpi=170)
    fig.savefig(OUT/'activation_vs_answers_by_metric.pdf')
    with (OUT/'activation_vs_answers_by_metric.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=export[0]);w.writeheader();w.writerows(export)
    print('Saved metric comparison PNG, PDF, and summary CSV.')

if __name__=='__main__':main()
