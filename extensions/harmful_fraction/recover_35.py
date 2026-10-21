"""Audit and finish only gaps in the authorized 35-condition design.

Uses immutable original trainer/measurement code and data from the Modal volume.
Never overwrites existing training or changes seeds, ordering, or the monitor.
"""
import json,sys
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
app=modal.App('harmful-fraction-finish-approved-35')
volume=modal.Volume.from_name('cs2881-em-harmful-fraction')
cache=modal.Volume.from_name('cs2881-hw0-model-cache')
CODE='dd16d4da04f5'
ROOTS=[(42,1,'fraction_v2_af02f09860eb_data42_train42'),(123,1,'fraction_v2_270ff51deae3_data42_train123_p50-100'),(789,1,'fraction_v2_5c3526344cba_data42_train789_p50-100'),(42,1,f'fraction_v2_{CODE}_data42_train42_p75_repeat1'),(42,2,f'fraction_v2_{CODE}_data42_train42_repeat2'),(42,3,f'fraction_v2_{CODE}_data42_train42_repeat3'),(123,1,f'fraction_v2_{CODE}_data42_train123_p25-0-5-10-75_repeat1'),(789,1,f'fraction_v2_{CODE}_data42_train789_p25-0-5-10-75_repeat1')]
EXPECTED={(s,r,f) for s,r in [(42,1),(42,2),(42,3),(123,1),(789,1)] for f in [0,5,10,25,50,75,100]}
TRAITS=['honesty','helpfulness','harmlessness','power_seeking','corrigibility','sycophancy','confidence']
image=(modal.Image.debian_slim(python_version='3.11').uv_pip_install('torch==2.7.0','transformers==4.56.1','accelerate==1.8.1','peft==0.15.2','datasets==3.6.0','numpy==2.2.6','pyyaml==6.0.2','tqdm==4.67.1','matplotlib==3.10.3').env({'HF_HOME':'/cache/huggingface','TOKENIZERS_PARALLELISM':'false','MPLBACKEND':'Agg','PYTHONUNBUFFERED':'1'}))

def environment():
 workspace=Path('/results/code')/CODE
 sys.path.insert(0,str(workspace/'extensions/harmful_fraction'))
 from verify import sha256,validate_run,validate_trajectory,write_json
 ref=json.loads((Path('/results')/ROOTS[0][2]/'protocol.json').read_text())
 for name in ['scratch_reproduction/09_finetune_dataset.py','scratch_reproduction/10_measure_dataset_drift.py']:
  assert sha256(workspace/name)==ref['code_sha256'][name],name
 instrument=Path('/results/workspace/scratch_reproduction/artifacts')
 for name in ['trait_directions_qwen_layer16.pt','base_position_qwen_layer16.pt']:
  assert sha256(instrument/name)==ref['instrument_sha256'][name],name
 return workspace,instrument,ref,sha256,validate_run,validate_trajectory,write_json

def audit_local():
 workspace,instrument,ref,sha256,validate_run,validate_trajectory,write_json=environment()
 found={};rows=[]
 for seed,repeat,name in ROOTS:
  root=Path('/results')/name
  if not root.exists():continue
  protocol=json.loads((root/'protocol.json').read_text())
  for field in ['source_sha256','ordering','data_seed','layer_index','normalization','instrument_sha256','model_revision']:
   assert protocol[field]==ref[field],(name,field)
  manifest=json.loads((root/'mixtures/manifest.json').read_text())
  for folder in root.glob('mix_*'):
   fraction=int(folder.name.split('_')[1][1:]);key=(seed,repeat,fraction)
   if key not in EXPECTED:continue
   if key in found:raise RuntimeError('Duplicate condition '+str(key))
   found[key]=folder
   row={'seed':seed,'repeat':repeat,'fraction':fraction,'path':str(folder),'training_valid':False,'measurement_valid':False,'adapter_files':len(list(folder.glob('checkpoint-*/adapter_model.safetensors')))}
   try:
    mix=next(x for x in manifest['mixtures'] if x['percent_harmful']==fraction);data=root/'mixtures'/mix['file']
    prov=json.loads((folder/'provenance.json').read_text())
    assert prov['mixture_sha256']==sha256(data)
    assert prov['protocol_sha256']==sha256(root/'protocol.json')
    assert prov['instrument_sha256']==ref['instrument_sha256']
    assert prov['training_seed']==seed and prov['data_seed']==42
    validate_run(folder,data,seed,126,workspace/'scratch_reproduction/artifacts/mixture_random_seed42/step_example_ids.jsonl' if fraction==25 else None)
    row['training_valid']=True
    validate_trajectory(folder/'trait_trajectory.json',TRAITS,126)
    baseline=json.loads((folder/'base_validation.json').read_text());assert baseline['relative_base_activation_error']<=.001
    assert (folder/'verification.json').exists()
    row['measurement_valid']=True
    row['final_adapter_sha256']=sha256(folder/'checkpoint-126/adapter_model.safetensors')
   except Exception as e:row['error']=str(e)
   rows.append(row)
 missing=[{'seed':s,'repeat':r,'fraction':f} for s,r,f in sorted(EXPECTED-set(found))]
 return {'planned':35,'verified':sum(r['training_valid'] and r['measurement_valid'] for r in rows),'conditions':rows,'missing':missing}

@app.function(volumes={'/results':volume},timeout=600)
def audit():return audit_local()

@app.function(image=image,gpu='A100-80GB',volumes={'/results':volume,'/cache':cache},timeout=24*3600,max_containers=1)
def recover():
 import os,subprocess,shutil,time
 from huggingface_hub import snapshot_download
 workspace,instrument,ref,sha256,validate_run,validate_trajectory,write_json=environment()
 initial=audit_local();write_json(Path('/results/recovery35_initial_audit.json'),initial);volume.commit()
 # Any unexpected defect is surfaced rather than silently retraining an existing run.
 bad=[r for r in initial['conditions'] if not r['training_valid']]
 if bad:raise RuntimeError('Existing incomplete training requires explicit recovery strategy: '+json.dumps(bad))
 model_path=snapshot_download(ref['model'],revision=ref['model_revision']);os.environ['EM_MODEL_PATH']=model_path
 original_verified={r['path']:r['final_adapter_sha256'] for r in initial['conditions'] if r['measurement_valid']}
 source=Path('/results')/ROOTS[-2][2]
 jobs=[(r['seed'],r['repeat'],r['fraction'],Path(r['path'])) for r in initial['conditions'] if not r['measurement_valid']]
 for r in initial['missing']:
  seed=r['seed'];assert seed in (123,789) and r['repeat']==1
  root=Path('/results')/next(n for s,rep,n in ROOTS if s==seed and 'p25-0-5-10-75' in n)
  if not root.exists():
   root.mkdir();shutil.copytree(source/'mixtures',root/'mixtures');shutil.copytree(source/'sources',root/'sources')
   protocol=json.loads((source/'protocol.json').read_text());protocol['training_seed']=seed;write_json(root/'protocol.json',protocol)
   write_json(root/'status.json',{'schema_version':2,'data_seed':42,'training_seed':seed,'root':str(root),'code_id':CODE,'stage':'authorized_recovery','completed':[],'selected_fractions':[25,0,5,10,75]})
  jobs.append((seed,1,r['fraction'],root/f"mix_p{r['fraction']:03d}_seed{seed}"))
 def command(script,args,folder,label):
  with (folder.parent/(label+'_'+folder.name+'.log')).open('a') as log:
   process=subprocess.run([sys.executable,str(workspace/'scratch_reproduction'/script),*map(str,args)],cwd=workspace,stdout=log,stderr=subprocess.STDOUT)
  volume.commit()
  if process.returncode:raise RuntimeError(label+' failed: '+str(folder))
 for seed,repeat,fraction,folder in jobs:
  root=folder.parent;manifest=json.loads((root/'mixtures/manifest.json').read_text());mix=next(x for x in manifest['mixtures'] if x['percent_harmful']==fraction);data=root/'mixtures'/mix['file']
  if not folder.exists():
   folder.mkdir();write_json(folder/'provenance.json',{'protocol_sha256':sha256(root/'protocol.json'),'instrument_sha256':ref['instrument_sha256'],'mixture_sha256':sha256(data),'model_revision':ref['model_revision'],'training_seed':seed,'data_seed':42});volume.commit()
   print('TRAIN',seed,fraction,flush=True)
   command('09_finetune_dataset.py',['--dataset-jsonl',data,'--run-name',folder.name,'--seed',seed,'--output-root',root,'--save-every',10,'--sequential'],folder,'recovery_train')
  training=validate_run(folder,data,seed,126,workspace/'scratch_reproduction/artifacts/mixture_random_seed42/step_example_ids.jsonl' if fraction==25 else None)
  assert json.loads((folder/'run_config.json').read_text())['model']==model_path
  write_json(folder/'training_complete.json',training);volume.commit()
  if not (folder/'trait_trajectory.json').exists():
   print('MEASURE',seed,fraction,flush=True)
   command('10_measure_dataset_drift.py',['--run-name',folder.name,'--output-root',root,'--instrument-dir',instrument],folder,'recovery_measure')
  measurement=validate_trajectory(folder/'trait_trajectory.json',TRAITS,126)
  baseline=json.loads((folder/'base_validation.json').read_text());assert baseline['relative_base_activation_error']<=.001
  write_json(folder/'verification.json',{'training':training,'measurement':measurement,'baseline':baseline})
  state=json.loads((root/'status.json').read_text());state['completed']=sorted(set(state['completed']+[folder.name]));state.update(stage='completed_'+folder.name,updated_unix=time.time());write_json(root/'status.json',state)
  write_json(Path('/results/recovery35_progress.json'),audit_local());volume.commit();print('VERIFIED',seed,fraction,flush=True)
 final=audit_local()
 assert final['verified']==35 and not final['missing']
 for path,digest in original_verified.items():assert sha256(Path(path)/'checkpoint-126/adapter_model.safetensors')==digest
 write_json(Path('/results/recovery35_final_audit.json'),final);volume.commit();return final

@app.local_entrypoint()
def main(run:bool=False):
 out=HERE/'deliverables/recovery35';out.mkdir(parents=True,exist_ok=True)
 result=recover.remote() if run else audit.remote()
 (out/('final_audit.json' if run else 'initial_audit.json')).write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
