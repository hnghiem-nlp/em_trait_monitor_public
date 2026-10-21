"""Render an auditable, offline HTML report from the pilot artifacts."""
import html
import json
import sys
from pathlib import Path

run=Path(sys.argv[1])
read=lambda name:json.loads((run/(name+'.json')).read_text())
meta,responses,grades,summary=map(read,['metadata','responses','grades','summary'])
esc=lambda x:html.escape(str(x))
byid={g['id']:g for g in grades if g['kind']=='target'}
coherence={g['id']:g for g in grades if g['kind']=='coherence'}
review=read('review') if (run/'review.json').exists() else None
audit=''
if review:
 audit='<h2>Judge audit: not ready for research scoring</h2><p>The pipeline completed, but inspection found factual errors and failures to follow rubrics. All scores below are diagnostic only. Passing the extreme controls did not establish reliability.</p><ul>'+''.join('<li><b>'+esc(f['id'])+'</b>: '+esc(f['issue'])+'</li>' for f in review['flags'])+'</ul>'
rows=''
for trait,data in summary['traits'].items():
 direction='More of this trait; not a quality score' if trait in ['power_seeking','sycophancy','confidence'] else 'Higher is better on this rubric'
 rows+=f"<tr><td>{esc(trait.replace('_',' ').title())}</td><td>{data['mean'] if data['mean'] is not None else 'Missing'}</td><td>{data['n_valid']}/{data['n']}</td><td>{direction}</td></tr>"
examples=''
for r in responses:
 g=byid[r['id']]; c=coherence[r['id']]
 msgs=''.join(f"<p><b>{esc(m['role'].title())}:</b> {esc(m['content'])}</p>" for m in r['messages'])
 examples+=f"""<details><summary>{esc(r['id'])} · trait {g['score']}/5 · coherence {c['score']}/5</summary>{msgs}<h4>Model answer</h4><pre>{esc(r['response'])}</pre><h4>Judge feedback</h4><p>{esc(g['feedback'])}</p><p class='muted'>Truncated response: {r['truncated']}. These scores have not been reviewed by human raters.</p></details>"""
controls=''.join(f"<tr><td>{esc(g['id'])}</td><td>{g['expected']}</td><td>{g['score']}</td><td>{esc(g['feedback'])}</td></tr>" for g in grades if g['kind']=='control')
source=Path(__file__).resolve().parents[1]/'harmful_fraction/deliverables/run_data/mix_p050_seed42/trait_trajectory.json'
activation=json.loads(source.read_text())[-1] if source.exists() else None
if activation:
 (run/'existing_activation_endpoint.json').write_text(json.dumps({'source':str(source),'note':'Previously measured endpoint from this run, not remeasured in this pilot. Behavioral scores and activation units are not interchangeable.','endpoint':activation},indent=2))
 act=''.join(f'<tr><td>{esc(k)}</td><td>{v:.3f}</td></tr>' for k,v in activation['normalized_drift_percent'].items())
 act=f'<h2>Existing activation measurements</h2><p>Final signed drift in % of base mean activation norm. These are linked from the original run. One endpoint cannot establish an activation–behavior relationship.</p><table><tr><th>Trait</th><th>Drift</th></tr>{act}</table>'
else: act=''
body=f'''<!doctype html><html><head><meta charset="utf-8"><title>Output judge pilot</title><style>
body{{font-family:system-ui,sans-serif;background:#f4f7fa;color:#183047;margin:0;padding:36px}}main{{max-width:1040px;margin:auto;background:white;padding:36px;border-radius:18px}}h1{{font-size:38px;letter-spacing:-1px;margin:12px 0}}h2{{margin-top:36px}}p,li{{line-height:1.6}}.tag{{color:#146ca3;font-weight:700}}.note{{background:#fff5db;border-left:4px solid #c59227;padding:16px}}.muted{{color:#627386;font-size:14px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #dde5ec}}th{{background:#eaf1f7}}details{{border:1px solid #dce4eb;border-radius:8px;padding:16px;margin:12px 0}}summary{{cursor:pointer;font-weight:650}}pre{{white-space:pre-wrap;font:15px/1.6 system-ui;background:#f4f7fa;padding:16px;border-radius:8px}}code{{overflow-wrap:anywhere}}</style></head><body><main>
<div class="tag">EXTENSION 2 · OUTPUT JUDGE PILOT</div><h1>What did the fine-tuned model actually say?</h1>
<p>One saved model. Eighteen test conversations. An independent open-weight evaluator.</p>
<div class="note"><b>Development test, not a research conclusion.</b> This checks that generation, grading and audit records work. The questions and rubrics are author-written; the judge is not human-validated. No baseline comparison, error bars or statistical significance are claimed.</div>
<h2>What ran</h2><ul><li>Target: Qwen2.5-7B-Instruct with the saved 50% harmful-data adapter, seed 42, training step 126.</li><li>Generation: deterministic greedy decoding, at most 256 new tokens per answer. No training.</li><li>Judge: Prometheus 2 7B, using criterion-specific 1–5 rubrics and reference answers. The judge never sees seed, condition or activation scores.</li><li>18 target answers, 18 coherence judgments, and 18 low/high controls. Synthetic correction history is explicitly supplied for one corrigibility test.</li><li>The native Prometheus 1–5 scale replaces the earlier proposed 0–3 scale; no claim of equivalence to published benchmark scores.</li></ul>
<h2>Technical checks</h2><p>Answers: {summary['n_responses']}; truncated answers: {summary['n_truncated_responses']}; valid judgments: {summary['n_judgments']-summary['n_invalid_judgments']}/{summary['n_judgments']}; controls matching their intended score exactly: {summary['controls']['n_exact']}/{summary['controls']['n']}.</p>
{audit}<h2>Raw scores — debugging only</h2><p>Mean coherence: {summary['coherence']['mean']}/5 across {summary['coherence']['n_valid']} valid judgments.</p><p>Each mean covers only a few different prompts, not independent training runs. There is no combined seven-trait score.</p><table><tr><th>Measure</th><th>Mean / 5</th><th>Valid / total</th><th>How to read it</th></tr>{rows}</table>
<h2>Read every answer and its grade</h2>{examples}
<h2>Confidence: objective checks</h2><p>The known-answer question allows an exact correctness check and one-item Brier loss. The hidden random number has known success probability 1%, but no revealed realized answer. Two items cannot validate calibration. A surrounding JSON code fence is accepted for extraction but recorded as strict_json=false; the original remote summary is preserved.</p><pre>{esc(json.dumps(summary['confidence'],indent=2))}</pre>
<h2>Judge controls</h2><p>These constructed control answers check whether the judge follows the rubric. They are excluded from model scores. A control failure is a reason to review the evaluator, not evidence about Qwen.</p><table><tr><th>Control</th><th>Expected</th><th>Actual</th><th>Feedback</th></tr>{controls}</table>
{act}<h2>Reproducibility</h2><p>Run: <code>{esc(meta['run_id'])}</code><br>Target revision: <code>{esc(meta['target_revision'])}</code><br>Adapter SHA-256: <code>{esc(meta['adapter_sha256'])}</code><br>Judge revision: <code>{esc(meta['judge_revision'])}</code></p>
<p class="muted">Artifacts alongside this report preserve raw answers, judge feedback, missing-score flags, model revisions and prompt/rubric hashes. Prompts are development-only and must be separated from the eventual held-out evaluation set.</p>
<p>Research basis: <a href="https://arxiv.org/abs/2405.01535">Prometheus 2</a>. This custom pilot does not reproduce the Betley or StrongREJECT protocols.</p></main></body></html>'''
(run/'report.html').write_text(body)
print(run/'report.html')
