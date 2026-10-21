"""Independent-seed summaries, preserving original and diagnostic suites."""
import csv,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def main():
 m=json.loads((HERE/'manifest.json').read_text());root=HERE/'runs'/m['digest'][:16]
 rows=list(csv.DictReader((root/'trait_metrics.csv').open()))
 fig,axes=plt.subplots(2,4,figsize=(15,8),layout='constrained')
 traits=['honesty','helpfulness','harmlessness','power_seeking','corrigibility','sycophancy','confidence']
 for ax,trait in zip(axes.flat,traits):
  base=next(r for r in rows if r['model']=='base' and r['suite']=='targeted' and r['metric']==trait)
  if base['mean']:ax.axhline(float(base['mean']),ls='--',color='gray',lw=1,label='Base')
  for seed in ['42','123','789']:
   a=sorted([r for r in rows if r['seed']==seed and r['repeat']=='1' and r['suite']=='targeted' and r['metric']==trait and r['mean']],key=lambda r:int(r['fraction']))
   ax.plot([int(r['fraction']) for r in a],[float(r['mean']) for r in a],'o-',label=seed)
  ax.set_title(trait.replace('_',' '));ax.set_ylim(.8,5.2);ax.set_xlabel('Harmful data (%)');ax.set_ylabel('Mean targeted score (1–5)')
 axes.flat[-1].axis('off');axes.flat[-1].text(0,1,'Six authored probes per trait\nOnly designated-trait questions\nRepeat 1; incomplete seed coverage\nDashed line: untuned base\nHigh power/sycophancy = more trait\nConfidence ≠ calibration\nHuman validation pending',va='top');axes.flat[0].legend()
 fig.suptitle('Expanded diagnostic questions · revised Claude rubric')
 fig.savefig(root/'expanded_trait_results.png',dpi=180);plt.close(fig)
 betley=[json.loads(l) for l in (root/'betley_judgments/betley_judged.jsonl').read_text().splitlines()]
 fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
 for seed in [42,123,789]:
  data=[]
  for model in m['models']:
   if model['seed']!=seed or model['repeat']!=1:continue
   eligible=[r for r in betley if r['run']==model['id'] and not r['excluded']]
   if eligible:data.append((model['fraction'],100*sum(r['misaligned'] for r in eligible)/len(eligible)))
  data.sort();ax.plot([a for a,b in data],[b for a,b in data],'o-',label=f'Seed {seed}')
 base=[r for r in betley if r['run']=='base' and not r['excluded']]
 if base:ax.axhline(100*sum(r['misaligned'] for r in base)/len(base),ls='--',color='gray',label='Base')
 ax.set_xlabel('Harmful training data (%)');ax.set_ylabel('Misaligned among eligible answers (%)');ax.set_title('Betley evaluation · 72 prompts × 5 samples per model');ax.legend()
 fig.savefig(root/'betley_results.png',dpi=180);plt.close(fig)
if __name__=='__main__':
 main()
 import plot_metric_comparison
 plot_metric_comparison.main()
