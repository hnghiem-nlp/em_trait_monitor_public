"""Pair output scores to base and activation; separate seed variation from repeats."""
import json,csv,statistics,html,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from judge import key,CRITERIA,VERSION

def mean(x):return statistics.mean(x) if x else None
def main():
 manifest=json.loads((HERE/'manifest.json').read_text());out=HERE/'runs'/manifest['digest'][:16]/'combined';cache=out/'judgments'
 grades={p.stem:json.loads(p.read_text()) for p in cache.glob('*.json')}
 grades={k:g for k,g in grades.items() if g.get('protocol_version')==VERSION}
 models={}
 for m in manifest['models']:
  path=out/(m['id']+'.json')
  if path.exists():models[m['id']]=json.loads(path.read_text())
 if 'base' not in models:raise RuntimeError('Base answers missing')
 baseline={r['id']:grades.get(key(r)) for r in models['base']['answers']}
 base_answers={r['id']:r for r in models['base']['answers']}
 rows=[];audits=[]
 for mid,data in models.items():
  m=data['model'];answers=data['answers']
  for suite in ('original','targeted'):
   selected=[r for r in answers if (r['category']=='targeted')==(suite=='targeted')]
   for trait in CRITERIA:
    vals=[];changes=[];untruncated_changes=[];truncated=0;valid=0
    for r in selected:
     g=grades.get(key(r));b=baseline.get(r['id']);truncated+=r['truncated']
     if not g or not g['valid']:continue
     valid+=1;s=g['metrics'][trait]['score']
     if s is not None:vals.append(s)
     if s is not None and b and b['valid'] and b['metrics'][trait]['score'] is not None:
      delta=s-b['metrics'][trait]['score'];changes.append(delta)
      if not r['truncated'] and not base_answers[r['id']]['truncated']:untruncated_changes.append(delta)
     if trait=='overall_quality' and (s is not None and s<=2 or r['truncated']):audits.append({'model':mid,'prompt_id':r['id'],'prompt':r['messages'],'response':r['response'],'score':s,'issues':g['issues'],'truncated':r['truncated']})
    pending=sum(not grades.get(key(r),{}).get('valid') for r in selected)
    drift=(m.get('activation') or {}).get('normalized_drift_percent',{}).get(trait)
    rows.append({'model':mid,'seed':m['seed'],'repeat':m['repeat'],'fraction':m['fraction'],'step':m['step'],'suite':suite,'metric':trait,'pending_judgments':pending,'score_mean':mean(vals) if pending==0 else None,'assessable':len(vals),'valid_judgments':valid,'total':len(selected),'paired_change_from_base':mean(changes) if pending==0 else None,'paired_n':len(changes),'untruncated_paired_change':mean(untruncated_changes),'untruncated_paired_n':len(untruncated_changes),'mean_tokens':mean([r['tokens'] for r in selected]),'activation_drift_percent':drift,'truncated':truncated})
 with (out/'model_metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'answer_audit_queue.json').write_text(json.dumps(audits,indent=2))
 groups=[]
 for suite in ('original','targeted'):
  for metric in CRITERIA:
   for step in sorted({r['step'] for r in rows if r['step']}):
    for frac in sorted({r['fraction'] for r in rows if r['fraction'] is not None}):
     a=[r for r in rows if r['repeat']==1 and r['step']==step and r['fraction']==frac and r['suite']==suite and r['metric']==metric and r['score_mean'] is not None]
     vals=[r['score_mean'] for r in a];changes=[r['paired_change_from_base'] for r in a if r['paired_change_from_base'] is not None]
     if vals:groups.append({'suite':suite,'metric':metric,'step':step,'fraction':frac,'seeds':[r['seed'] for r in a],'n_seeds':len(a),'mean':mean(vals),'sd_across_seeds':statistics.stdev(vals) if len(vals)>1 else None,'paired_change_mean':mean(changes)})
 (out/'seed_summary.json').write_text(json.dumps(groups,indent=2))
 esc=html.escape
 s='''<!doctype html><meta charset="utf-8"><title>Checkpoint 2: activation and output quality</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#193047}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse;width:100%}pre{white-space:pre-wrap}details{padding:10px;border:1px solid #ccc;margin:8px 0}</style><h1>Checkpoint 2: activation drift versus judged output quality</h1>'''
 s+=f'<p>{len(models)}/{len(manifest["models"])} models generated; {sum(g["valid"] for g in grades.values())}/{len(grades)} cached unique judgments structurally valid. Identical prompt–answer pairs share a judgment. No training performed.</p>'
 s+='''<p>Original: 115 reused monitor prompts. Targeted: 18 development probes. Scores are blinded Claude ratings, not verified ground truth. Overall quality is the primary behavioral measure; trait scores are conditional on assessability. High power seeking, sycophancy and confidence mean more of that trait, not better quality. Nulls are excluded and counts are reported.</p><p>Error bars, where available, are sample SD across distinct seeds (repeat 1 only). Same-seed repeats are not independent replications. Missing runs are not imputed. Correlations across related training checkpoints would be descriptive, not independent statistical tests. The medical-versus-numbers training domain confound remains.</p><h2>Original prompts: individual models</h2><table><tr><th>Model</th><th>Quality /5</th><th>Paired Δ from base</th><th>Paired prompts</th><th>Truncated</th></tr>'''
 for r in rows:
  if r['suite']=='original' and r['metric']=='overall_quality':
   fmt=lambda v:'—' if v is None else f'{v:.3f}'
   s+=f"<tr><td><a href='{r['model']}.json'>{r['model']}</a></td><td>{fmt(r['score_mean'])}</td><td>{fmt(r['paired_change_from_base'])}</td><td>{r['paired_n']}/{r['total']}</td><td>{r['truncated']}</td></tr>"
 s+='</table><h2>Answers requiring review</h2><p>Low scores and truncated answers are listed here; this selection is not a prevalence estimate.</p>'
 for a in audits:
  s+=f"<details><summary>{esc(a['model'])} · {esc(a['prompt_id'])} · quality {a['score']}</summary><pre>{esc(json.dumps(a['prompt'],ensure_ascii=False))}</pre><pre>{esc(a['response'])}</pre><p>{esc(str(a['issues']))}</p></details>"
 (out/'report.html').write_text(s)
 print('REPORT',out/'report.html');print('Unique judge failures',sum(not g['valid'] for g in grades.values()));print('Audit items',len(audits))
if __name__=='__main__':main()
