"""Download verified numerical evidence and exact mixture files after recovery."""
import json,sys
from pathlib import Path
import modal
sys.path.insert(0,str(Path(__file__).resolve().parent))
from recover_35 import audit_local,ROOTS,CODE
app=modal.App('harmful-fraction-export-35')
v=modal.Volume.from_name('cs2881-em-harmful-fraction')
image=modal.Image.debian_slim(python_version='3.11').add_local_file(Path(__file__).resolve().with_name('recover_35.py'), '/root/recover_35.py')
@app.function(image=image,volumes={'/results':v},timeout=900)
def export():
 audit=audit_local()
 if audit['verified']!=35:raise RuntimeError('Refusing final export: not all 35 conditions verified')
 files={}
 for row in audit['conditions']:
  folder=Path(row['path']);prefix=f"seed{row['seed']}_repeat{row['repeat']}/p{row['fraction']:03d}"
  for name in ['trait_trajectory.json','verification.json','provenance.json','run_config.json','training_complete.json','base_validation.json','step_example_ids.jsonl']:
   p=folder/name
   if not p.exists():raise FileNotFoundError(str(p))
   files[prefix+'/'+name]=p.read_text()
  files[prefix+'/protocol.json']=(folder.parent/'protocol.json').read_text()
 root=Path('/results')/ROOTS[-2][2]
 for p in (root/'mixtures').glob('*'):
  if p.is_file():files['mixtures/'+p.name]=p.read_text()
 return {'audit':audit,'files':files,'weights_location':'Modal volume cs2881-em-harmful-fraction in penrose/vlln-test; all checkpoint weights preserved at the condition paths in final_audit.json. Final weight hashes recorded; intermediate checkpoints checked for presence/configuration.'}
@app.local_entrypoint()
def main():
 out=Path(__file__).resolve().parent/'deliverables/recovery35';result=export.remote()
 for name,text in result['files'].items():
  p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 (out/'final_audit.json').write_text(json.dumps(result['audit'],indent=2))
 (out/'DATA_LOCATION.txt').write_text(result['weights_location']+'\n')
 print('Verified 35/35; downloaded',len(result['files']),'evidence/data files',flush=True)
