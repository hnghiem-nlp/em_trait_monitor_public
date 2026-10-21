"""Freeze the completed grid and reuse prior evidence without overwriting new results."""
import json,hashlib,shutil
from pathlib import Path
repo=Path(__file__).resolve().parents[3];parent=repo/'extensions/output_judge';src=parent/'validation_v2';dst=parent/'full35';dst.mkdir(exist_ok=True)
m=json.loads((src/'manifest.json').read_text());old_digest=m['digest'];audit=json.loads((repo/'extensions/harmful_fraction/deliverables/recovery35/final_audit.json').read_text());assert audit['verified']==35
known={x['id']:x for x in m['models']};models=[known['base']]
for c in audit['conditions']:
 mid=f"p{c['fraction']:03d}_s{c['seed']}_r{c['repeat']}_t126"
 local=repo/f"extensions/harmful_fraction/deliverables/recovery35/seed{c['seed']}_repeat{c['repeat']}/p{c['fraction']:03d}"
 x=known.get(mid,dict(id=mid,adapter=c['path']+'/checkpoint-126',seed=c['seed'],repeat=c['repeat'],fraction=c['fraction'],step=126,activation=next(r for r in json.loads((local/'trait_trajectory.json').read_text()) if r['step']==126),provenance=json.loads((local/'provenance.json').read_text())))
 x['expected_adapter_sha256']=c['final_adapter_sha256'];models.append(x)
m['models']=models;m['scope']='All 35 verified final training conditions plus base; reuse prior answers and judgments with identical inputs; generate missing eight models only.';m['previous_manifest']=old_digest;m.pop('digest');m['digest']=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest();(dst/'manifest.json').write_text(json.dumps(m,indent=2));root=dst/'runs'/m['digest'][:16];root.mkdir(parents=True,exist_ok=True)
prior=src/'runs'/old_digest[:16]
for x in models:
 mid=x['id']
 for suffix in ['.json','_betley.jsonl']:
  if (prior/(mid+suffix)).exists() and not (root/(mid+suffix)).exists():shutil.copy2(prior/(mid+suffix),root/(mid+suffix))
for folder in ['combined','betley_judgments']:
 if not (root/folder).exists():shutil.copytree(prior/folder,root/folder)
(root/'manifest.json').write_text(json.dumps(m,indent=2))
