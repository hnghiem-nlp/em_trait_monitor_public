"""Finite continuation of the explicitly authorized 35-condition recovery job."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUT=HERE/'deliverables/recovery35'
def main():
 status=OUT/'completion_status.json';deadline=time.time()+4*3600
 try:
  status.write_text(json.dumps({'stage':'awaiting_authorized_recovery'}))
  while not (OUT/'final_audit.json').exists():
   text=Path('/tmp/recovery35_training.log').read_text(errors='replace')
   if 'Traceback (most recent call last)' in text or 'Error:' in text:raise RuntimeError('Training recovery failed; inspect /tmp/recovery35_training.log')
   if time.time()>deadline:raise TimeoutError('Recovery exceeded four hours')
   time.sleep(30)
  assert json.loads((OUT/'final_audit.json').read_text())['verified']==35
  env=dict(os.environ,MODAL_PROFILE='penrose',MPLCONFIGDIR='/tmp/recovery35-mpl')
  subprocess.run([sys.executable,'-m','modal','run','--env','vlln-test',str(HERE/'export_35.py')],cwd=REPO,env=env,check=True)
  # Explicit environment on each CLI download; no default-profile assumptions.
  env['MODAL_ENVIRONMENT']='vlln-test'
  subprocess.run([sys.executable,str(HERE/'live_results.py')],cwd=REPO,env=env,check=True)
  summary=json.loads((HERE/'deliverables/live/summary.json').read_text());assert summary['completed']==35 and not summary['errors']
  status.write_text(json.dumps({'stage':'complete','verified':35,'downloaded_evidence':True,'results_refreshed':True},indent=2))
 except Exception as e:
  status.write_text(json.dumps({'stage':'failed','error':str(e)},indent=2));raise
if __name__=='__main__':main()
