"""One finite evaluation job; no scheduled or recurring monitor, no training."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
PYTHON=sys.executable

def run(args,profile=None):
 env=dict(os.environ)
 if profile:env['MODAL_PROFILE']=profile
 print('RUN', ' '.join(map(str,args)),flush=True)
 subprocess.run([str(x) for x in args],cwd=REPO,env=env,check=True)

def main():
 m=json.loads((HERE/'manifest.json').read_text());root=HERE/'runs'/m['digest'][:16]
 state=root/'pipeline_status.json'
 def status(stage):state.write_text(json.dumps({'stage':stage,'unix':time.time(),'manifest':m['digest']},indent=2))
 try:
  status('waiting_for_authorized_generation_and_initial_regrading')
  deadline=time.time()+4*3600
  while not all((root/(x['id']+'.json')).exists() for x in m['models']):
   if time.time()>deadline:raise TimeoutError('Generation did not finish within four hours')
   log=Path('/tmp/cp2_v2_generation.log').read_text(errors='replace')
   if 'Traceback (most recent call last)' in log or 'App stopped' in log:raise RuntimeError('Generation failed; inspect log')
   time.sleep(30)
  # Avoid concurrent duplicate API grading while the original-answer pass is active.
  while 'App completed' not in Path('/tmp/cp2_v2_judge.log').read_text(errors='replace'):
   log=Path('/tmp/cp2_v2_judge.log').read_text(errors='replace')
   if 'Traceback (most recent call last)' in log or 'UnboundLocalError' in log:raise RuntimeError('Initial judge failed; inspect log')
   if time.time()>deadline:raise TimeoutError('Initial regrading timed out')
   time.sleep(30)
  status('exporting_answers')
  run([PYTHON,HERE/'prepare.py'])
  run([PYTHON,HERE/'human_review.py'])
  status('grading_expanded_traits')
  run([PYTHON,'-m','modal','run','--env','main',HERE/'judge.py'],'harsh-24539')
  status('grading_betley')
  files=sorted(root.glob('*_betley.jsonl'))
  run([PYTHON,HERE/'betley_judge.py','--responses',*files,'--output-dir',root/'betley_judgments','--model','gpt-5.6-sol','--concurrency','4'])
  status('analyzing_and_auditing')
  run([PYTHON,HERE/'analyze.py']);run([PYTHON,HERE/'report.py']);run([PYTHON,HERE/'plots.py'])
  audit=json.loads((root/'coverage_audit.json').read_text())
  if not audit['complete_automated_evaluation']:raise RuntimeError('Incomplete coverage; do not claim final results')
  status('automated_complete_human_validation_pending')
 except Exception as e:
  state.write_text(json.dumps({'stage':'failed','error':str(e),'unix':time.time()},indent=2));raise
if __name__=='__main__':main()
