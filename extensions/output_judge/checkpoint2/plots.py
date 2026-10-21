"""Scientific figures; SD across distinct seeds, no pooled-repeat significance."""
import csv,json,statistics,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def ranks(a):
 return np.array([sum(v<x for v in a)+(sum(v==x for v in a)+1)/2 for x in a])
def corr(a,b):
 if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
 return float(np.corrcoef(ranks(a),ranks(b))[0,1])
def main():
 manifest=json.loads((HERE/'manifest.json').read_text());out=HERE/'runs'/manifest['digest'][:16]
 rows=list(csv.DictReader((out/'model_metrics.csv').open()));models={m['id']:m for m in manifest['models']}
 weights=json.loads((HERE.parents[2]/'scratch_reproduction/artifacts/qwen_seed42_pc1.json').read_text())['pc1']
 primary=[r for r in rows if r['repeat']=='1' and r['step']=='126' and r['suite']=='original' and r['metric']=='overall_quality' and r['paired_change_from_base']]
 points=[]
 for r in primary:
  drift=models[r['model']]['activation']['normalized_drift_percent'];pc=abs(sum(weights[k]*drift[k] for k in weights))
  points.append({**r,'pc1_abs':pc,'change':float(r['paired_change_from_base'])})
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 for seed in (42,123,789):
  a=sorted([p for p in points if int(p['seed'])==seed],key=lambda p:int(p['fraction']))
  if not a:continue
  axes[0].plot([int(p['fraction']) for p in a],[p['change'] for p in a],'o-',label=f'Seed {seed}')
  axes[1].scatter([p['pc1_abs'] for p in a],[p['change'] for p in a],label=f'Seed {seed}')
  for p in a:axes[1].annotate(p['fraction']+'%',(p['pc1_abs'],p['change']),fontsize=7,xytext=(3,12 if p['fraction']=='50' else -12 if p['fraction']=='75' else 3),textcoords='offset points')
 for ax in axes:ax.axhline(0,color='gray',lw=.8);ax.set_ylabel('Paired change in judged quality (1–5 scale)');ax.legend()
 axes[0].set_xlabel('Harmful training data (%)');axes[1].set_xlabel('Original activation drift |PC1| (%)')
 fig.suptitle('Checkpoint 2 final models · original 115 prompts · repeat 1 only')
 fig.savefig(out/'quality_vs_activation.png',dpi=180);plt.close(fig)
 associations=[]
 for seed in (42,123,789):
  a=[p for p in points if int(p['seed'])==seed]
  associations.append({'seed':seed,'conditions':len(a),'spearman_descriptive':corr([p['pc1_abs'] for p in a],[p['change'] for p in a]),'note':'Descriptive association across related conditions, not an independent-sample significance test.'})
 (out/'associations.json').write_text(json.dumps(associations,indent=2))
 fig,axes=plt.subplots(2,4,figsize=(15,8),layout='constrained')
 for ax,trait in zip(axes.flat,weights):
  a=[r for r in rows if r['repeat']=='1' and r['step']=='126' and r['suite']=='original' and r['metric']==trait and r['paired_change_from_base'] and r['activation_drift_percent']]
  for seed in (42,123,789):
   b=[r for r in a if int(r['seed'])==seed]
   ax.scatter([float(r['activation_drift_percent']) for r in b],[float(r['paired_change_from_base']) for r in b],label=str(seed))
  ax.set_title(trait.replace('_',' '));ax.axhline(0,color='gray',lw=.6);ax.set_xlabel('Activation drift (%)');ax.set_ylabel('Paired trait-score change')
 axes.flat[-1].axis('off');axes.flat[-1].text(0,.9,'Original prompts only\nScores conditional on assessability\nHigh power / sycophancy = more trait\nConfidence ≠ calibration\nNo causal or significance claim',va='top')
 fig.savefig(out/'seven_traits.png',dpi=180);plt.close(fig)
 # Error bars are ONLY seed SD, computed from repeat 1.
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
 for frac in sorted({int(p['fraction']) for p in points}):
  vals=[p['change'] for p in points if int(p['fraction'])==frac]
  ax.scatter([frac]*len(vals),vals,alpha=.5)
  ax.errorbar(frac,statistics.mean(vals),yerr=statistics.stdev(vals) if len(vals)>1 else None,fmt='ko',capsize=4)
  ax.annotate(f'n={len(vals)}',(frac,statistics.mean(vals)),xytext=(4,8),textcoords='offset points',fontsize=8)
 ax.axhline(0,color='gray',lw=.8);ax.set_xlabel('Harmful training data (%)');ax.set_ylabel('Paired quality change from base');ax.set_title('Mean ±1 SD across available distinct seeds; no bar for n=1')
 fig.savefig(out/'quality_seed_variation.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(15,8),layout='constrained')
 for ax,trait in zip(axes.flat,weights):
  a=[r for r in rows if r['repeat']=='1' and r['step']=='126' and r['suite']=='original' and r['metric']==trait and r['paired_change_from_base']]
  fractions=sorted({int(r['fraction']) for r in a});means=[]
  for frac in fractions:
   vals=[float(r['paired_change_from_base']) for r in a if int(r['fraction'])==frac];avg=statistics.mean(vals);means.append(avg)
   ax.scatter([frac]*len(vals),vals,s=14,alpha=.45)
   ax.errorbar(frac,avg,yerr=statistics.stdev(vals) if len(vals)>1 else None,fmt='ko',capsize=3,markersize=3)
  ax.plot(fractions,means,color='gray',lw=.7);ax.axhline(0,color='gray',lw=.6);ax.set_title(trait.replace('_',' '));ax.set_xlabel('Harmful training data (%)');ax.set_ylabel('Paired score change from base')
 axes.flat[-1].axis('off');axes.flat[-1].text(0,.9,'Dots: independent seeds, repeat 1\nBars: ±1 sample SD, not CI\nNo bar where only one seed exists\nConditional on assessable answers\nCorrigibility: only 1–2 paired prompts\nConfidence ≠ calibration',va='top')
 fig.suptitle('Seven output traits · original prompts · limited applicability for some traits')
 fig.savefig(out/'seven_traits_seed_variation.png',dpi=180);plt.close(fig)
 print(json.dumps(associations,indent=2))
if __name__=='__main__':main()
