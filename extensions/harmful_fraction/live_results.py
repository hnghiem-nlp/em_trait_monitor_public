"""Download verified completed conditions and refresh honest partial-result charts."""
import json, subprocess, hashlib, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
OUT=BASE/'deliverables/live';OUT.mkdir(parents=True,exist_ok=True)
MODAL='/Users/harsh/Documents/HW/.venv/bin/modal'
ROOT='/fraction_v2_dd16d4da04f5_data42_train'
SOURCES=[(42,1,'/fraction_v2_af02f09860eb_data42_train42',BASE/'deliverables/run_data'),
 (123,1,'/fraction_v2_270ff51deae3_data42_train123_p50-100',BASE/'deliverables/replication_seed123'),
 (789,1,'/fraction_v2_5c3526344cba_data42_train789_p50-100',OUT/'seed789'),
 (42,1,ROOT+'42_p75_repeat1',OUT/'seed42_add75'),
 (42,2,ROOT+'42_repeat2',OUT/'seed42_repeat2'),(42,3,ROOT+'42_repeat3',OUT/'seed42_repeat3'),
 (123,1,ROOT+'123_p25-0-5-10-75_repeat1',OUT/'seed123_remaining'),
 (789,1,ROOT+'789_p25-0-5-10-75_repeat1',OUT/'seed789_remaining')]
def fetch(remote,local,optional=False):
 local.parent.mkdir(parents=True,exist_ok=True)
 r=subprocess.run([MODAL,'volume','get','--force','cs2881-em-harmful-fraction',remote,str(local)],capture_output=True,text=True)
 if r.returncode and not optional:raise RuntimeError(r.stderr[-1000:])
 return r.returncode==0
ref=json.loads((BASE/'deliverables/run_data/protocol.json').read_text())
axis=json.loads((BASE.parents[1]/'scratch_reproduction/artifacts/qwen_seed42_pc1.json').read_text());traits=axis['traits'];weights=np.array([axis['pc1'][t] for t in traits])
records=[];errors=[]
for seed,repeat,remote,local in SOURCES:
 if not fetch(remote+'/status.json',local/'status.json',optional=True):continue
 state=json.loads((local/'status.json').read_text())
 if state['stage']=='failed':errors.append(state.get('error',remote))
 for name in state['completed']:
  if not name.startswith('mix_'):continue
  folder=local/name
  for fname in ['verification.json','trait_trajectory.json','provenance.json']:
   if not (folder/fname).exists():fetch(remote+'/'+name+'/'+fname,folder/fname)
  prov=json.loads((folder/'provenance.json').read_text());check=json.loads((folder/'verification.json').read_text())
  assert prov['instrument_sha256']==ref['instrument_sha256'] and prov['model_revision']==ref['model_revision']
  assert check['training']['optimizer_steps']==126 and check['training']['exact_step_order']
  assert check['measurement']['all_finite'] and check['baseline']['relative_base_activation_error']<=.001
  if not (local/'protocol.json').exists():fetch(remote+'/protocol.json',local/'protocol.json')
  protocol=json.loads((local/'protocol.json').read_text())
  for k in ['source_sha256','ordering','data_seed','layer_index','normalization']:
   assert protocol[k]==ref[k],k
  for k in ['scratch_reproduction/09_finetune_dataset.py','scratch_reproduction/10_measure_dataset_drift.py']:
   assert protocol['code_sha256'][k]==ref['code_sha256'][k],k
  data=json.loads((folder/'trait_trajectory.json').read_text());end=data[-1];assert end['step']==126
  values=end['normalized_drift_percent'];p=int(name.split('_')[1][1:])
  records.append(dict(seed=seed,repeat=repeat,fraction=p,pc1=abs(float(np.array([values[t] for t in traits])@weights)),**values,path=str(folder)))
records.sort(key=lambda r:(r['seed'],r['repeat'],r['fraction']))
assert len({(r['seed'],r['repeat'],r['fraction']) for r in records})==len(records)
for r in records:
 originals=[a for a in records if a['seed']==42 and a['repeat']==1 and a['fraction']==r['fraction']]
 if originals:
  a=json.loads((Path(originals[0]['path'])/'provenance.json').read_text());b=json.loads((Path(r['path'])/'provenance.json').read_text())
  assert a['mixture_sha256']==b['mixture_sha256']
with (OUT/'results.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
colors=['#0072B2','#008563','#D55E00','#AA4499','#59636E','#3695B5','#A97900']
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#AAB8C4','xtick.color':'#536678','ytick.color':'#536678'})
ind=[r for r in records if r['repeat']==1]
fractions=[0,5,10,25,50,75,100]
counts={p:sum(r['fraction']==p for r in ind) for p in fractions}
for metric,color in [('pc1','#186AA7')]+list(zip(traits,colors)):
 small=metric!='pc1';fig,ax=plt.subplots(figsize=((285/96,167/96) if small else (745/96,332/96)),dpi=200)
 xs=[];means=[]
 for p in fractions:
  rs=[r for r in ind if r['fraction']==p];ys=[r[metric] for r in rs]
  if not ys:continue
  xs.append(p);means.append(np.mean(ys))
  for r in rs:ax.scatter(p+{42:-1.2,123:0,789:1.2}[r['seed']],r[metric],s=12 if small else 28,color=color,alpha=.45,zorder=3)
  if len(ys)>1:ax.errorbar(p,np.mean(ys),yerr=np.std(ys,ddof=1),color=color,capsize=2,lw=1,zorder=2)
 ax.plot(xs,means,color=color,lw=1.5,marker='o',ms=2.5)
 ax.axhline(0,color='#AAB8C4',ls='--',lw=.6)
 ax.set(xlim=(-4,104),xticks=[0,25,50,75,100]);ax.grid(axis='y',color='#E2E8ED',lw=.6);ax.set_axisbelow(True);ax.tick_params(labelsize=9 if small else 12,length=0,pad=4)
 if not small:ax.set_xlabel('Harmful training examples (%)',fontsize=12)
 fig.subplots_adjust(left=.18 if small else .08,right=.97,top=.94,bottom=.19 if small else .21)
 fig.savefig(OUT/(metric+'.png'),facecolor='white');plt.close(fig)
paired=[]
for seed in [42,123,789]:
 a=next((r for r in ind if r['seed']==seed and r['fraction']==50),None);b=next((r for r in ind if r['seed']==seed and r['fraction']==100),None)
 if a and b:paired.append({'seed':seed,'difference':a['pc1']-b['pc1']})
repeats=[]
for r in records:
 if r['repeat']==1:continue
 a=next((a for a in ind if a['seed']==42 and a['fraction']==r['fraction']),None)
 if a:
  x=json.loads((Path(a['path'])/'trait_trajectory.json').read_text());y=json.loads((Path(r['path'])/'trait_trajectory.json').read_text())
  assert [i['step'] for i in x]==[i['step'] for i in y]
  maxdiff=max(abs(i['normalized_drift_percent'][t]-j['normalized_drift_percent'][t]) for i,j in zip(x,y) for t in traits)
  repeats.append({'repeat':r['repeat'],'fraction':r['fraction'],'max_trajectory_difference':maxdiff,'final_pc1_difference':r['pc1']-a['pc1']})
summary={'completed':len(records),'total':35,'distinct_seed_counts':counts,'paired_50_minus_100':paired,'reproducibility':repeats,'errors':errors,'fingerprint':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
