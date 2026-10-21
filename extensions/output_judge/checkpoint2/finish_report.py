"""Add a concise, data-driven overview and verified figures to the detailed report."""
import csv,json,statistics,html
from pathlib import Path
HERE=Path(__file__).resolve().parent;m=json.loads((HERE/'manifest.json').read_text());out=HERE/'runs'/m['digest'][:16]
rows=list(csv.DictReader((out/'model_metrics.csv').open()));groups=json.loads((out/'seed_summary.json').read_text())
primary=[r for r in rows if r['suite']=='original' and r['metric']=='overall_quality']
base=next(r for r in primary if r['model']=='base')
summary=[g for g in groups if g['suite']=='original' and g['metric']=='overall_quality' and g['step']==126]
s='<h2>Results overview</h2><p><b>Scope: final saved step 126 across available conditions and repeats, plus base. Intermediate training steps were not generated in this batch.</b></p>'
s+=f"<p>Base judged quality: <b>{float(base['score_mean']):.3f}/5</b> on {base['paired_n']} original prompts.</p>"
s+='<table><tr><th>Harmful fraction</th><th>Available independent seeds</th><th>Mean quality /5</th><th>SD across seeds</th><th>Paired change from base</th></tr>'
for r in sorted(summary,key=lambda r:r['fraction']):
 sd='—' if r['sd_across_seeds'] is None else f"{r['sd_across_seeds']:.3f}"
 s+=f"<tr><td>{r['fraction']}%</td><td>{', '.join(map(str,r['seeds']))}</td><td>{r['mean']:.3f}</td><td>{sd}</td><td>{r['paired_change_mean']:.3f}</td></tr>"
s+='</table><p>These are judged-quality scores, not externally verified accuracy or harm rates. The audit finds both concrete defects and apparent penalties for concision. Low ratings alone do not establish harmful behavior. The base model also fails the targeted pressured-arithmetic question, so not every trained-model failure is newly introduced.</p>'
paired=[]
for seed in (42,123,789):
 a=next((r for r in primary if r['seed']==str(seed) and r['repeat']=='1' and r['fraction']=='50' and r['score_mean']),None)
 b=next((r for r in primary if r['seed']==str(seed) and r['repeat']=='1' and r['fraction']=='100' and r['score_mean']),None)
 if a and b:paired.append({'seed':seed,'quality_50':float(a['score_mean']),'quality_100':float(b['score_mean']),'change_100_minus_50':float(b['score_mean'])-float(a['score_mean'])})
(out/'paired_50_100.json').write_text(json.dumps(paired,indent=2))
s+='<h3>50% versus 100% within each seed</h3><table><tr><th>Seed</th><th>50% quality</th><th>100% quality</th><th>100% − 50%</th></tr>'
for r in paired:s+=f"<tr><td>{r['seed']}</td><td>{r['quality_50']:.3f}</td><td>{r['quality_100']:.3f}</td><td>{r['change_100_minus_50']:+.3f}</td></tr>"
s+='</table>'
if len(paired)==3 and all(r['change_100_minus_50']<0 for r in paired):
 s+='<p><b>Main finding: all three seeds show lower judged quality at 100% than at 50%, despite lower activation |PC1|. The apparent activation recovery does not translate into quality recovery in this evaluation.</b></p>'
s+='<p>Interpret this alongside the original activation result: |PC1| decreases from 50% to 100% in all three seeds. Agreement or disagreement in output changes tests whether that activation ranking transfers to behavior; three seeds provide limited precision.</p>'
for file in ('quality_vs_activation.png','quality_seed_variation.png','seven_traits.png','seven_traits_seed_variation.png'):
 if (out/file).exists():s+=f'<img src="{file}" style="width:100%;height:auto" alt="{file}">'
audit=json.loads((out/'qualitative_audit.json').read_text()) if (out/'qualitative_audit.json').exists() else {}
s+='<h3>Answer audit</h3><p>'+html.escape(audit.get('scope',''))+'</p>'
for r in audit.get('findings',[]):s+=f"<p><b>{html.escape(r['model'])} / {html.escape(r['prompt'])}:</b> {html.escape(r['finding'])}</p>"
s+='<p><a href="model_metrics.csv">All model and trait metrics</a> · <a href="category_metrics.csv">Category breakdown</a> · <a href="deterministic_diagnostics.json">Exact targeted answers and confidence checks</a> · <a href="repeat_reproducibility.json">Same-seed repeat comparison</a></p>'
p=out/'report.html';text=p.read_text();anchor='<h2>Original prompts: individual models</h2>';text=text.replace(anchor,s+anchor);p.write_text(text)
print(json.dumps({'fraction_summary':summary,'paired_50_100':paired},indent=2))
