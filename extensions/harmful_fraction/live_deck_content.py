"""Prepare native slide copy and table values from verified live results."""
import csv,json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent/'deliverables/live'
s=json.loads((p/'summary.json').read_text());rows=list(csv.DictReader((p/'results.csv').open()));ind=[r for r in rows if r['repeat']=='1'];done=s['completed']
text={
'p1_i5':'Five sweeps, seven training mixes',
'p1_i6':'Each model starts fresh from\nQwen2.5-7B-Instruct.',
'p1_i8':'1,000 examples per model.\n0%, 5%, 10%, 25%, 50%, 75%, or 100% harmful.',
'simple_controls_header':'What the repeats check',
'simple_controls_1':'Different training seeds',
'simple_controls_2':'Reproducibility',
'simple_controls_1b':'Three seeds: 42, 123, and 789.',
'simple_controls_2b':'Seed 42 is repeated three times.',
'simple_controls_3b':'Same prompts and trait directions for every run.',
'p2_i2':'Does the pattern hold up when we repeat it?',
'p2_i3':f'{done} of 35 models finished. Dots show distinct seeds; bars show ±1 SD.',
'p2_i9':'50% versus 100%',
'p2_i10':f"50% is higher in {sum(x['difference']>0 for x in s['paired_50_minus_100'])} of {len(s['paired_50_minus_100'])} seeds.\nOnly three distinct seeds are planned.",
'p2_i11':'Partial results: seed counts differ across mixes.',
'p2_i12':'Mean and SD use one run per seed. Same-seed repeats are checked separately.',
'p3_i3':'Dots: seeds. Bars: ±1 SD. Captions compare 50% → 100%. Vertical scales differ.',
'p3_i15':'Internal activation changes, not answer ratings. Missing runs are not filled in.',
'small_reading_title':'Read this carefully',
'small_reading_body':'Results are still coming in. Error bars show variation across seeds, not proof of significance.',
'p4_i9':'Finish the repeat checks',
'p4_i10':f'{done}/35 models completed. Three distinct seeds test variability.\nTwo extra seed-42 sweeps check reproducibility.'}
for i,t in enumerate(['honesty','helpfulness','harmlessness','power_seeking','corrigibility','sycophancy','confidence']):
 differences=[]
 for seed in ['42','123','789']:
  a=next((r for r in ind if r['seed']==seed and r['fraction']=='50'),None);b=next((r for r in ind if r['seed']==seed and r['fraction']=='100'),None)
  if a and b:differences.append(float(b[t])-float(a[t]))
 n=len(differences);up=sum(d>0 for d in differences);down=sum(d<0 for d in differences)
 text[f'small_tip_{i}']=f'{"Up" if up>=down else "Down"} in {max(up,down)}/{n} seeds.'
if s['reproducibility']:
 maximum=max(r['max_trajectory_difference'] for r in s['reproducibility'])
 text['p4_i10']=f'{len(s["reproducibility"])} same-seed checks completed.\nLargest difference across trait curves: {maximum:.8f} points.'
if done==35:
 text['p2_i3']='All 35 models finished. Dots show distinct seeds; bars show ±1 SD.'
 text['p2_i11']='Each mix has three distinct seeds; seed-42 repeats are checked separately.'
 text['small_reading_body']='A smaller overall score can hide larger changes in individual traits.'
 text['p4_i9']='Check more seeds and data samples'
 text['p2_i2']='The 50%–100% pattern across three seeds'
if (p/'incident.json').exists() and done<35:
 text['p2_i3']=f'{done}/35 verified. Batch stopped. Dots: distinct seeds; bars: ±1 SD.'
 text['small_reading_body']='Partial results only. Error bars show variation across seeds, not proof of significance.'
 text['p4_i9']='Resume the remaining runs'
 text['p4_i10']='Batch cancelled before completion; saved results are preserved.\nEight conditions remain unverified; no restart has been launched.'
table=[['Harmful','Mean','Seeds']]
for f in [0,5,10,25,50,75,100]:
 values=[float(r['pc1']) for r in ind if int(r['fraction'])==f]
 table.append([f'{f}%',f'{np.mean(values):.2f}' if values else 'Pending',str(len(values))])
notes='Live results: '+json.dumps(s)+'\nFive sweeps use seeds42,42,42,123,789. Original seed42 selected the exploratory comparison; seeds123/789 are follow-ups. Same-seed repeats are reproducibility checks, not independent evidence. Error bars are sample standard deviation across distinct seeds (ddof=1), not confidence intervals. One seed has no SD bar. Means with unequal seed coverage are provisional. PC1 is a frozen activation projection, not behavioral harmfulness. Data and order remain fixed; medical-versus-numbers confounding remains. The previously launched seed2026 pair completed before scope changed; it is retained as supplemental data outside the planned five sweeps, not used here.'
if (p/'incident.json').exists():notes+='\nBatch incident: '+(p/'incident.json').read_text()
(p/'deck_content.json').write_text(json.dumps(dict(text=text,table=table,notes=notes,summary=s),indent=2))
print(p/'deck_content.json')
