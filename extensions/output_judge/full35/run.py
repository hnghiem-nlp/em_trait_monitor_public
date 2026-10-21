"""Finite evaluation of the completed grid; no training or recurring monitoring."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'manifest.json').read_text());root=HERE/'runs'/m['digest'][:16]
def run(script,profile=None,args=()):
 env=dict(os.environ,MPLCONFIGDIR='/tmp/full35-mpl')
 if profile:env['MODAL_PROFILE']=profile
 cmd=[sys.executable]+(['-m','modal','run','--env','vlln-test' if profile=='penrose' else 'main'] if profile else [])+[str(HERE/script)]+list(map(str,args))
 (root/'pipeline_status.json').write_text(json.dumps({'stage':script,'time':time.time()}));subprocess.run(cmd,env=env,check=True)
try:
 run('generate.py','penrose');run('prepare.py');run('judge.py','harsh-24539')
 run('betley_judge.py',args=['--responses',*sorted(root.glob('*_betley.jsonl')),'--output-dir',root/'betley_judgments','--model','gpt-5.6-sol','--concurrency','4'])
 for script in ['analyze.py','report.py','plots.py','compare.py']:run(script)
 audit=json.loads((root/'coverage_audit.json').read_text())
 assert audit['complete_automated_evaluation'],'Coverage incomplete; inspect invalid ratings'
 (root/'pipeline_status.json').write_text(json.dumps({'stage':'automated_complete_human_validation_pending','time':time.time()}))
except Exception as e:
 (root/'pipeline_status.json').write_text(json.dumps({'stage':'failed','error':str(e),'time':time.time()}));raise
