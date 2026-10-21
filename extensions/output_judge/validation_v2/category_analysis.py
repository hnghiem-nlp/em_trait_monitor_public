"""Category-specific and response-length summaries; raw judge scores preserved."""
import json,statistics,sys,csv
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from judge import key
m=json.loads((HERE/'manifest.json').read_text());out=HERE/'runs'/m['digest'][:16]
def get_grade(r):
 p=out/'judgments'/(key(r)+'.json')
 return json.loads(p.read_text()) if p.exists() else None
base=json.loads((out/'base.json').read_text());b={r['id']:(r,get_grade(r)) for r in base['answers']}
rows=[]
for model in m['models']:
 p=out/(model['id']+'.json')
 if not p.exists():continue
 answers=json.loads(p.read_text())['answers']
 for cat in ('technical','ethical','emotional','casual','adversarial','targeted'):
  a=[r for r in answers if r['category']==cat];changes=[];scores=[]
  for r in a:
   g=get_grade(r);bg=b[r['id']][1]
   if g and g['valid']:
    scores.append(g['metrics']['overall_quality']['score'])
    if bg and bg['valid']:changes.append(g['metrics']['overall_quality']['score']-bg['metrics']['overall_quality']['score'])
  rows.append({'model':model['id'],'seed':model['seed'],'repeat':model['repeat'],'fraction':model['fraction'],'category':cat,'n':len(a),'scored':len(scores),'quality_mean':statistics.mean(scores) if len(scores)==len(a) else None,'paired_change':statistics.mean(changes) if len(changes)==len(a) else None,'mean_tokens':statistics.mean(r['tokens'] for r in a),'truncated':sum(r['truncated'] for r in a)})
with (out/'category_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Saved category metrics for',len(rows)//6,'models')
