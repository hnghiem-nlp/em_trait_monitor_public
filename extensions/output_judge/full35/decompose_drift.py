"""Arithmetic decomposition of observed PC1 change, not a causal attribution."""
import csv,json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;repo=HERE.parents[2]
a=list(csv.DictReader((repo/'extensions/harmful_fraction/deliverables/live/results.csv').open()))
w=json.loads((repo/'scratch_reproduction/artifacts/qwen_seed42_pc1.json').read_text())['pc1']
rows=[]
for t,weight in w.items():
 vals=[]
 for seed in [42,123,789]:
  pair=[next(x for x in a if x['repeat']=='1' and x['seed']==str(seed) and x['fraction']==str(f)) for f in [50,100]]
  # Both signed PC1 values are negative, so the absolute-value change equals minus the signed change.
  assert all(sum(float(x[k])*v for k,v in w.items())<0 for x in pair)
  vals.append(-weight*(float(pair[1][t])-float(pair[0][t])))
 rows.append(dict(trait=t,mean_contribution=st.mean(vals),seed42=vals[0],seed123=vals[1],seed789=vals[2]))
fig,ax=plt.subplots(figsize=(9,5),layout='constrained');v=[r['mean_contribution'] for r in rows]
ax.barh([r['trait'].replace('_',' ').title() for r in rows],v,color=['#187cad' if x<0 else '#ce6f27' for x in v]);ax.axvline(0,color='gray',lw=1);ax.invert_yaxis()
for i,x in enumerate(v):ax.text(x+(.035 if x>=0 else -.035),i,f'{x:+.3f}',ha='right' if x<0 else 'left',va='center',fontsize=10)
ax.set_xlim(-2.8,1.0);ax.set_xlabel('Contribution to change in |PC1|: 100% minus 50%');ax.set_title('Why does the aggregate activation measure fall?\nThree-seed mean; contributions sum to −3.283');ax.spines[['top','right']].set_visible(False)
out=HERE/'results'
for ext in ['png','pdf']:fig.savefig(out/('activation_change_components.'+ext),dpi=180)
with (out/'activation_change_components.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
print(rows)
