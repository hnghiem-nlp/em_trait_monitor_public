"""Compare independent-seed outcome trends, never equate the two judge scales."""
import csv,json,statistics as st,argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=HERE);args=ap.parse_args();source=args.source
 m=json.loads((source/'manifest.json').read_text());root=source/'runs'/m['digest'][:16]
 traits=list(csv.DictReader((root/'trait_metrics.csv').open()));bt=[json.loads(l) for l in (root/'betley_judgments/betley_judged.jsonl').read_text().splitlines()]
 activation=list(csv.DictReader((HERE.parents[1]/'harmful_fraction/deliverables/live/results.csv').open()))
 rows=[]
 for model in m['models']:
  if model['repeat']!=1 and model['id']!='base':continue
  mid=model['id'];b=[r for r in bt if r['run']==mid];eligible=[r for r in b if not r['excluded']]
  q=next(r for r in traits if r['model']==mid and r['suite']=='original' and r['metric']=='overall_quality')
  a=next((r for r in activation if int(r['seed'])==model['seed'] and int(r['repeat'])==1 and int(r['fraction'])==model['fraction']),None)
  rows.append(dict(model=mid,seed=model['seed'],fraction=model['fraction'],quality=float(q['mean']) if q['mean'] and int(q['n'])==115 else None,misalignment=100*sum(r['misaligned'] for r in eligible)/len(eligible) if len(b)==360 and eligible else None,eligible=len(eligible),excluded=len(b)-len(eligible),activation=float(a['pc1']) if a else 0))
 summary=[];fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
 for ax,metric,title,ylabel in zip(axes,['activation','quality','misalignment'],['Internal activation drift','Your revised Claude judge','Checkpoint 3 judge'],['Original |PC1| measure','Answer quality (1–5; higher is better)','Misalignment (% eligible; lower is better)']):
  for seed in [42,123,789]:
   rr=sorted([r for r in rows if r['seed']==seed and r[metric] is not None],key=lambda r:r['fraction'])
   ax.plot([r['fraction'] for r in rr],[r[metric] for r in rr],'o-',alpha=.45,lw=1,label=f'Seed {seed}')
  for f in [0,5,10,25,50,75,100]:
   v=[r[metric] for r in rows if r['fraction']==f and r[metric] is not None];n=len(v)
   if not n:continue
   avg=st.mean(v);sd=st.stdev(v) if n>1 else None
   ax.errorbar(f,avg,yerr=sd,fmt='ko',capsize=4)
   summary.append(dict(fraction=f,metric=metric,mean=avg,sd=sd,n_seeds=n))
  base=next(r for r in rows if r['model']=='base')
  if base[metric] is not None:ax.axhline(base[metric],ls=':',color='gray',label='Base model')
  counts=[sum(r['fraction']==f and r[metric] is not None for r in rows) for f in [0,5,10,25,50,75,100]]
  ax.text(.01,-.24,'Seed counts at 0/5/10/25/50/75/100%: '+ '/'.join(map(str,counts)),transform=ax.transAxes,fontsize=8)
  ax.set(title=title,xlabel='Harmful training data (%)',ylabel=ylabel);ax.spines[['top','right']].set_visible(False)
 axes[0].legend(fontsize=8)
 complete=json.loads((root/'coverage_audit.json').read_text())['complete_automated_evaluation']
 fig.suptitle(('Completed' if complete else 'Provisional')+' evaluation · black: mean ± seed SD · independent human validation pending',fontsize=12)
 fig.savefig(root/'both_judges_vs_activation.png',dpi=180);plt.close(fig)
 for name,data in [('both_judges_by_seed',rows),('both_judges_summary',summary)]:
  with (root/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
 print(root/'both_judges_vs_activation.png')
if __name__=='__main__':main()
