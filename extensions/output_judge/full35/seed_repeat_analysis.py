"""Audit all outputs; separate independent training seeds from fixed-seed repeats."""
import csv,json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from judge import key,validate
HERE=Path(__file__).resolve().parent
FRACTIONS=[0,5,10,25,50,75,100]
COLORS=['#1479ac','#d57924','#8b52a1']

def save_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
 m=json.loads((HERE/'manifest.json').read_text());root=HERE/'runs'/m['digest'][:16]
 audit=json.loads((root/'coverage_audit.json').read_text());assert audit['complete_automated_evaluation']
 traits=list(csv.DictReader((root/'trait_metrics.csv').open()))
 bt=[json.loads(l) for l in (root/'betley_judgments/betley_judged.jsonl').read_text().splitlines()]
 act=list(csv.DictReader((HERE.parents[1]/'harmful_fraction/deliverables/live/results.csv').open()))
 data=[];outputs={};verified=0
 for model in m['models']:
  mid=model['id'];raw=json.loads((root/'combined'/(mid+'.json')).read_text());outputs[mid]=raw
  assert raw['adapter_sha256']==model.get('expected_adapter_sha256')
  answers=raw['answers'];assert len(answers)==157 and len({x['id'] for x in answers})==157
  for a in answers:
   g=json.loads((root/'combined/judgments'/(key(a)+'.json')).read_text());assert g['valid'];validate(g,a);verified+=1
  b=[x for x in bt if x['run']==mid];assert len(b)==360 and len({(x['prompt_id'],x['sample_idx']) for x in b})==360
  e=[x for x in b if not x['excluded']]
  q=next(x for x in traits if x['model']==mid and x['suite']=='original' and x['metric']=='overall_quality');assert int(q['n'])==115
  a=next((x for x in act if (str(model['seed']),str(model['repeat']),str(model['fraction']))==(x['seed'],x['repeat'],x['fraction'])),None)
  data.append(dict(model=mid,seed=model['seed'],repeat=model['repeat'],fraction=model['fraction'],activation=float(a['pc1']) if a else 0,quality=float(q['mean']),misalignment=100*sum(x['misaligned'] for x in e)/len(e),eligible=len(e),excluded=len(b)-len(e)))
 summaries=[];repro=[]
 for f in FRACTIONS:
  for metric in ['activation','quality','misalignment']:
   seeds=[r[metric] for r in data if r['repeat']==1 and r['fraction']==f]
   repeats=[r[metric] for r in data if r['seed']==42 and r['fraction']==f]
   assert len(seeds)==len(repeats)==3
   summaries.append(dict(fraction=f,metric=metric,seed_mean=st.mean(seeds),seed_sd=st.stdev(seeds),repeat_mean=st.mean(repeats),repeat_sd=st.stdev(repeats)))
  ids=[f'p{f:03d}_s42_r{r}_t126' for r in [1,2,3]]
  hashes=[outputs[i]['adapter_sha256'] for i in ids]
  original=[{a['id']:a['response'] for a in outputs[i]['answers'] if a['category']!='targeted'} for i in ids]
  repro.append(dict(fraction=f,identical_adapter_hashes=len(set(hashes))==1,original_answers=115,repeat2_identical_answers=sum(v==original[1][k] for k,v in original[0].items()),repeat3_identical_answers=sum(v==original[2][k] for k,v in original[0].items())))
 save_csv(root/'seed_repeat_model_results.csv',data);save_csv(root/'seed_repeat_variation.csv',summaries);save_csv(root/'repeat_reproducibility.csv',repro)
 metrics=['activation','quality','misalignment'];titles=['Activation drift |PC1|','Claude answer quality ↑','Checkpoint 3 misalignment ↓'];labels=['Activation measure','Mean score (1–5)','% of eligible responses']
 fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
 for row in range(2):
  for ax,metric,title,label in zip(axes[row],metrics,titles,labels):
   for j,value in enumerate([42,123,789] if row==0 else [1,2,3]):
    rr=sorted([r for r in data if r['repeat']==1 and r['seed']==value] if row==0 else [r for r in data if r['seed']==42 and r['repeat']==value],key=lambda r:r['fraction'])
    ax.plot([r['fraction'] for r in rr],[r[metric] for r in rr],marker=['o','s','^'][j],color=COLORS[j],lw=1.6,ms=5,ls=['-','--',':'][j],label=f'Seed {value}' if row==0 else f'Run {value}')
   base=next(r for r in data if r['model']=='base');ax.axhline(base[metric],color='gray',ls=':',lw=1,label='Untuned base')
   ax.set(title=('Different seeds: ' if row==0 else 'Same seed 42: ')+title,xlabel='Harmful training data (%)',ylabel=label)
   ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
 for j in range(3):
  lo=min(axes[0,j].get_ylim()[0],axes[1,j].get_ylim()[0]);hi=max(axes[0,j].get_ylim()[1],axes[1,j].get_ylim()[1]);axes[0,j].set_ylim(lo,hi);axes[1,j].set_ylim(lo,hi)
 fig.suptitle('Different training seeds versus repeated runs at seed 42\nAll seven fractions verified with both judges; axes match within each column',fontsize=15)
 for ext in ['png','pdf']:fig.savefig(root/('seeds_vs_repeats.'+ext),dpi=190)
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
 for ax,metric,title,label in zip(axes,metrics,titles,labels):
  rr=[r for r in summaries if r['metric']==metric]
  ax.plot(FRACTIONS,[r['seed_sd'] for r in rr],'o-',color=COLORS[0],label='Across seeds 42, 123, 789')
  ax.plot(FRACTIONS,[r['repeat_sd'] for r in rr],'s--',color=COLORS[1],label='Across three runs at seed 42')
  ax.set(title=title,xlabel='Harmful training data (%)',ylabel='Standard deviation ('+('percentage points' if metric=='misalignment' else 'score units')+')',ylim=(0,None));ax.spines[['top','right']].set_visible(False)
  if metric=='activation':ax.text(.03,.92,'Fixed-seed SD ≤ '+f"{max(r['repeat_sd'] for r in rr):.1e}",transform=ax.transAxes,fontsize=10)
 axes[0].legend(fontsize=9);fig.suptitle('How much variation comes with each comparison?\nDescriptive SD, n=3 each; groups overlap at seed 42 / run 1, so this is not a variance decomposition',fontsize=12)
 for ext in ['png','pdf']:fig.savefig(root/('seed_repeat_spread.'+ext),dpi=190)
 plt.close(fig)
 ts=['honesty','helpfulness','harmlessness','power_seeking','corrigibility','sycophancy','confidence']
 fig,axes=plt.subplots(2,4,figsize=(15,8),layout='constrained');traitspread=[]
 for ax,t in zip(axes.flat,ts):
  for group,color in [('different seeds',COLORS[0]),('same seed repeats',COLORS[1])]:
   vals=[]
   for f in FRACTIONS:
    rr=[r for r in traits if r['suite']=='targeted' and r['metric']==t and r['fraction']==str(f) and (r['repeat']=='1' if group=='different seeds' else r['seed']=='42')]
    assert len(rr)==3
    v=[float(r['mean']) for r in rr];vals.append(st.stdev(v));traitspread.append(dict(metric=t,fraction=f,group=group,sd=st.stdev(v),mean=st.mean(v)))
   ax.plot(FRACTIONS,vals,'o-' if group=='different seeds' else 's--',color=color,label=group)
  ax.set(title=t.replace('_',' ').title(),xlabel='Harmful training data (%)',ylabel='SD of mean trait score',ylim=(0,.8));ax.spines[['top','right']].set_visible(False)
 axes.flat[-1].axis('off');axes.flat[-1].text(0,1,'Six targeted probes per trait.\nSD across three model evaluations.\nZero SD means agreement here, not certainty.\nConfidence measures expressed certainty,\nnot calibration or quality.\nHuman validation pending.',va='top',linespacing=1.6);axes.flat[0].legend(fontsize=8)
 fig.suptitle('Variation by behavioral metric · targeted Claude probes',fontsize=15)
 for ext in ['png','pdf']:fig.savefig(root/('seed_repeat_traits.'+ext),dpi=190)
 plt.close(fig);save_csv(root/'seed_repeat_trait_variation.csv',traitspread)
 (root/'verification_all_points.json').write_text(json.dumps(dict(models=len(data),trait_answers_validated=verified,betley_judgments=len(bt),all_adapter_hashes_match=True,fixed_seed_reproducibility=repro,human_validation='pending'),indent=2))
 print(json.dumps({'verified_models':len(data),'traits':verified,'betley':len(bt),'repeat_reproducibility':repro,'variation':summaries},indent=2))
if __name__=='__main__':main()
