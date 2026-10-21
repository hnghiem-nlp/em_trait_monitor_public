"""Same-seed reproducibility reported separately from between-seed variation."""
import json,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));from judge import key
m=json.loads((HERE/'manifest.json').read_text());out=HERE/'runs'/m['digest'][:16]
def load(model):
 p=out/(model['id']+'.json');return json.loads(p.read_text()) if p.exists() else None
def score(row):
 p=out/'judgments'/(key(row)+'.json')
 if not p.exists():return None
 g=json.loads(p.read_text());return g['metrics']['overall_quality']['score'] if g['valid'] else None
rows=[]
for a in m['models']:
 if a['seed']!=42 or a['repeat'] not in (2,3):continue
 b=next((x for x in m['models'] if x['seed']==42 and x['repeat']==1 and x['step']==a['step'] and x['fraction']==a['fraction']),None)
 if not b:continue
 ad,bd=load(a),load(b)
 if not ad or not bd:continue
 br={r['id']:r for r in bd['answers']};d=[];same=0
 for r in ad['answers']:
  rr=br[r['id']];same+=r['response']==rr['response'];s,t=score(r),score(rr)
  if s is not None and t is not None:d.append(s-t)
 rows.append({'fraction':a['fraction'],'step':a['step'],'repeat':a['repeat'],'exact_answer_matches':same,'total_answers':len(ad['answers']),'paired_judged':len(d),'quality_change_from_repeat1':statistics.mean(d) if len(d)==len(ad['answers']) else None,'max_activation_difference_percent':max(abs(a['activation']['normalized_drift_percent'][k]-b['activation']['normalized_drift_percent'][k]) for k in a['activation']['normalized_drift_percent'])})
(out/'repeat_reproducibility.json').write_text(json.dumps(rows,indent=2));print('Repeat comparisons:',len(rows))
