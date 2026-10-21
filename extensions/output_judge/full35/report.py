"""Audit coverage and report original, diagnostic, and Betley outcomes separately."""
import collections,csv,html,json,statistics
from pathlib import Path
from judge import key,VERSION
HERE=Path(__file__).resolve().parent

def main():
 m=json.loads((HERE/'manifest.json').read_text());root=HERE/'runs'/m['digest'][:16];combined=root/'combined'
 cache={p.stem:json.loads(p.read_text()) for p in (combined/'judgments').glob('*.json')}
 bpath=root/'betley_judgments/betley_judged.jsonl';betley=[json.loads(l) for l in bpath.read_text().splitlines()] if bpath.exists() else []
 by=collections.defaultdict(list)
 for r in betley:by[r['run']].append(r)
 metrics=[];coverage=[]
 for model in m['models']:
  mid=model['id'];path=combined/(mid+'.json');answers=json.loads(path.read_text())['answers'] if path.exists() else []
  valid=[r for r in answers if cache.get(key(r),{}).get('valid') and cache[key(r)].get('protocol_version')==VERSION]
  coverage.append({'model':mid,'trait_answers':len(answers),'trait_judgments':len(valid),'expected_trait_answers':157,'betley_judgments':len(by[mid]),'expected_betley_answers':360})
  for suite in ['original','targeted']:
   selected=[r for r in valid if (r['category']=='targeted')==(suite=='targeted')]
   for trait in ['overall_quality','honesty','helpfulness','harmlessness','power_seeking','corrigibility','sycophancy','confidence']:
    eligible=[r for r in selected if suite=='original' or r.get('target_trait')==trait]
    scores=[cache[key(r)]['metrics'][trait]['score'] for r in eligible];scores=[s for s in scores if s is not None]
    metrics.append({'model':mid,'seed':model['seed'],'repeat':model['repeat'],'fraction':model['fraction'],'suite':suite,'metric':trait,'n':len(scores),'mean':statistics.mean(scores) if scores else None,'activation_drift_percent':model.get('activation',{}).get('normalized_drift_percent',{}).get(trait) if model.get('activation') else 0})
 expected={(x['id'],p['id'],sample) for x in m['models'] for p in m['prompts'] if p['category']=='betley' for sample in range(5)}
 actual=[(r['run'],r['prompt_id'],r['sample_idx']) for r in betley]
 complete=all(c['trait_answers']==157 and c['trait_judgments']==157 and c['betley_judgments']==360 for c in coverage) and len(actual)==len(set(actual)) and set(actual)==expected
 audit={'complete_automated_evaluation':complete,'human_validation':'pending independent ratings','models':coverage,'betley_duplicate_keys':len(actual)-len(set(actual)),'betley_missing_keys':len(expected-set(actual)),'judge_protocol':VERSION}
 (root/'coverage_audit.json').write_text(json.dumps(audit,indent=2))
 with (root/'trait_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=metrics[0]);w.writeheader();w.writerows(metrics)
 oldrows=list(csv.DictReader((HERE.parent/'checkpoint2/runs/f098104a25b14102/model_metrics.csv').open()))
 comparison=[]
 for r in metrics:
  if r['suite']!='original':continue
  old=next((o for o in oldrows if o['model']==r['model'] and o['suite']=='original' and o['metric']==r['metric']),None)
  if old and old['score_mean'] and r['mean'] is not None:
   comparison.append({'model':r['model'],'metric':r['metric'],'old_mean':float(old['score_mean']),'revised_mean':r['mean'],'difference':r['mean']-float(old['score_mean']),'old_assessable':old['assessable'],'revised_assessable':r['n'],'note':'Same original responses; rubric changed; eligibility may differ.'})
 if comparison:
  with (root/'original_rubric_comparison.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=comparison[0]);w.writeheader();w.writerows(comparison)
 body=['<h1>Checkpoint 2 expanded validation</h1>', '<p>Automated evaluation '+('COMPLETE' if complete else 'INCOMPLETE — provisional only')+'. Independent human validation pending.</p>', '<p>Original answers reused and regraded under revised rubric. Expanded diagnostics are authored development probes. Betley uses the Extension 3 rubric and requested judge model, but generation uses vLLM; this is not an identical runtime replication.</p>','<h2>Betley output misalignment</h2><table><tr><th>Model</th><th>Judged / expected</th><th>Eligible</th><th>Misaligned</th><th>Rate</th></tr>']
 for model in m['models']:
  mid=model['id'];rows=by[mid];eligible=[r for r in rows if not r['excluded']];bad=sum(r['misaligned'] for r in eligible)
  rate=f'{100*bad/len(eligible):.2f}%' if len(rows)==360 and eligible else 'pending'
  body.append(f'<tr><td>{mid}</td><td>{len(rows)}/360</td><td>{len(eligible)}</td><td>{bad}</td><td>{rate}</td></tr>')
 body+=['</table><h2>Trait metrics</h2><p>See trait_metrics.csv. Targeted metrics use only their six designated questions. This avoids diluting power seeking or corrigibility with unrelated prompts. High power seeking and sycophancy mean more of those behaviors; confidence is certainty, not accuracy. No human validation claim is made.</p>']
 (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>'+''.join(body))
 print(json.dumps({'complete':complete,'trait_valid':sum(c['trait_judgments'] for c in coverage),'betley_valid':len(betley),'report':str(root/'report.html')}))
if __name__=='__main__':main()
