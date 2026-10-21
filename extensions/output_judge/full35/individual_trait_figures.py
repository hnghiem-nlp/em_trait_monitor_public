"""One figure per trait: activation plus original/targeted paired behavioral changes."""
import csv,json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from judge import key
HERE=Path(__file__).resolve().parent
TRAITS=['honesty','helpfulness','harmlessness','power_seeking','corrigibility','sycophancy','confidence']
COLORS={42:'#1479ac',123:'#d57924',789:'#8b52a1'}
FRACTIONS=[0,5,10,25,50,75,100]

def main():
 m=json.loads((HERE/'manifest.json').read_text());root=HERE/'runs'/m['digest'][:16];out=HERE/'results/individual_traits';out.mkdir(parents=True,exist_ok=True)
 assert json.loads((root/'coverage_audit.json').read_text())['complete_automated_evaluation']
 def read(mid):
  a=json.loads((root/'combined'/(mid+'.json')).read_text())['answers']
  return {x['id']:(x,json.loads((root/'combined/judgments'/(key(x)+'.json')).read_text())['metrics']) for x in a}
 base=read('base');models=[x for x in m['models'] if x['repeat']==1];assert len(models)==21
 answers={x['id']:read(x['id']) for x in models};export=[]
 for t in TRAITS:
  rows=[]
  for model in models:
   for suite in ['original','targeted']:
    changes=[]
    for pid,(r,g) in answers[model['id']].items():
     eligible=r['category']!='targeted' if suite=='original' else r.get('target_trait')==t
     if not eligible:continue
     score=g[t]['score'];bs=base[pid][1][t]['score']
     if score is not None and bs is not None:changes.append(score-bs)
    assert changes
    rows.append(dict(trait=t,seed=model['seed'],fraction=model['fraction'],suite=suite,activation=model['activation']['normalized_drift_percent'][t],paired_answer_change=st.mean(changes),paired_questions=len(changes)))
  export.extend(rows);fig,axes=plt.subplots(1,3,figsize=(14,4.6))
  for j,(ax,title) in enumerate(zip(axes,['Activation on original 115 prompts','Answers to original questions','Answers to targeted trait probes'])):
   rr=[r for r in rows if r['suite']==('targeted' if j==2 else 'original')];field='activation' if j==0 else 'paired_answer_change'
   for seed,color in COLORS.items():
    r=sorted([x for x in rr if x['seed']==seed],key=lambda x:x['fraction'])
    ax.plot([x['fraction'] for x in r],[x[field] for x in r],'o-',color=color,alpha=.65,lw=1,ms=4)
   for f in FRACTIONS:
    values=[x[field] for x in rr if x['fraction']==f];assert len(values)==3
    ax.errorbar(f,st.mean(values),yerr=st.stdev(values),fmt='ko',ms=4,capsize=3,lw=1.3)
   n=[x['paired_questions'] for x in rr]
   ax.set_title(title+(f'\n{min(n)}–{max(n)} assessable pairs / model' if j else ''),fontsize=11)
   ax.axhline(0,color='gray',ls=':',lw=1);ax.set_xlabel('Harmful training data (%)')
   ax.set_ylabel('Trait projection change (% of base norm)' if j==0 else 'Trait-score change from base (1–5 scale)')
   ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
  if t in ['power_seeking','sycophancy']:direction='Positive answer change = more '+t.replace('_',' ')+' (worse).'
  elif t=='confidence':direction='Positive answer change = more expressed certainty; not necessarily better calibration.'
  else:direction='Positive answer change = better judged '+t+'.'
  note='Behavioral pairs use the same question in the tuned and base models; eligibility can vary by condition.'
  if t=='corrigibility':note='Original-question coverage is sparse: interpret that panel cautiously. Targeted probes provide six opportunities.'
  fig.suptitle('Checkpoint 2 · '+t.replace('_',' ').title(),fontsize=17,fontweight='bold',y=.99)
  fig.legend(handles=[Line2D([0],[0],color=c,marker='o',label=f'Seed {s}') for s,c in COLORS.items()]+[Line2D([0],[0],color='black',marker='o',ls='',label='Mean ± 1 seed SD')],loc='upper center',bbox_to_anchor=(.5,.93),ncol=4,frameon=False,fontsize=9)
  fig.text(.065,.055,direction+' Separate y-scales; zero = untuned base.',fontsize=9)
  fig.text(.065,.018,note+' SD is not a confidence interval.',fontsize=8.5)
  fig.subplots_adjust(top=.74,bottom=.22,left=.065,right=.985,wspace=.35)
  for ext in ['png','pdf']:fig.savefig(out/(t+'.'+ext),dpi=175)
  plt.close(fig)
 with (out/'paired_trait_data.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=export[0]);w.writeheader();w.writerows(export)
 print('Saved seven verified PNG/PDF figures and paired data:',out)
if __name__=='__main__':main()
