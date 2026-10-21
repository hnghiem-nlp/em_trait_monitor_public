"""Recompute descriptive summaries locally without rerunning generation or judging."""
import json,sys,hashlib
from pathlib import Path
from suite import ITEMS,RUBRICS,digest
from summarize import summarize
run=Path(sys.argv[1])
read=lambda n:json.loads((run/(n+'.json')).read_text())
meta=read('metadata')
assert meta['suite_sha256']==digest(ITEMS) and meta['rubrics_sha256']==digest(RUBRICS)
if not (run/'summary_remote.json').exists():
 (run/'summary_remote.json').write_text((run/'summary.json').read_text())
result=summarize(read('responses'),read('grades'),ITEMS)
result['analysis_note']='Development parser accepts a single surrounding JSON code fence, but separately records strict_json=false. Raw model outputs and remote summary are preserved. No judge scores were changed.'
result['analysis_code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('suite.py'),Path(__file__).with_name('summarize.py')]}
for name,obj in [('summary',result),('suite',ITEMS),('rubrics',RUBRICS)]:
 (run/(name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
print(json.dumps(result,indent=2))
