"""Deterministic diagnostic extraction; does not overwrite judge ratings."""
import json,sys,re,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from suite import confidence_fields,ITEMS
manifest=json.loads((HERE/'manifest.json').read_text());out=HERE/'runs'/manifest['digest'][:16]
summary=[];sample=[]
for model in manifest['models']:
 path=out/(model['id']+'.json')
 if not path.exists():continue
 data=json.loads(path.read_text());by={r['id']:r for r in data['answers']};entry={'model':model['id'],'truncated_ids':[r['id'] for r in data['answers'] if r['truncated']],'mean_tokens':sum(r['tokens'] for r in data['answers'])/len(data['answers'])}
 for id in ('confidence_known','confidence_unknown'):
  spec=next(s for s in ITEMS if s['id']==id);entry[id]=confidence_fields(by['targeted_'+id]['response'],spec)
 for id in ('sycophancy_neutral','sycophancy_pressure','sycophancy_pressure2','corrigibility_fact','corrigibility_stop'):
  entry[id]=by['targeted_'+id]['response']
 summary.append(entry)
 # Reproducible sample per category, independent of score, plus all diagnostic probes.
 for cat in ('technical','ethical','emotional','casual','adversarial','targeted'):
  rows=[r for r in data['answers'] if r['category']==cat]
  selected=rows if cat=='targeted' else sorted(rows,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:2]
  sample.extend({'model':model['id'],**r} for r in selected)
(out/'deterministic_diagnostics.json').write_text(json.dumps(summary,indent=2));(out/'blinded_score_independent_review_sample.json').write_text(json.dumps(sample,indent=2))
print('Diagnostics',len(summary),'models; review sample',len(sample),'answers. Sample selected independent of judge score, but model labels are retained for analyst comparison.')
