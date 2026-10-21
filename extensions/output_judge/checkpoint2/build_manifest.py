"""Freeze the available approved-seed adapter inventory and exact prompt suite."""
import json,re,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from suite import ITEMS

def build(all_steps=False):
 inventory=json.loads((HERE/'inventory.json').read_text()); rows=[]; excluded=[]
 for row in inventory:
  path=row['adapter'];seed=int(re.search(r'_seed(\d+)',path).group(1));step=int(path.rsplit('-',1)[1]);fraction=int(re.search(r'mix_p(\d+)',path).group(1));repeat=int(re.search(r'_repeat(\d+)',path).group(1)) if '_repeat' in path else 1
  if seed not in (42,123,789):continue
  if not all_steps and step!=126:continue
  trajectory=row['trajectory']; endpoint=next((x for x in trajectory or [] if x['step']==step),None)
  if endpoint is None:excluded.append({'adapter':path,'reason':'No measured activation at this step'});continue
  assert row['provenance']['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
  rows.append({'id':f'p{fraction:03d}_s{seed}_r{repeat}_t{step:03d}','adapter':path,'seed':seed,'repeat':repeat,'fraction':fraction,'step':step,'activation':endpoint,'provenance':row['provenance']})
 rows.sort(key=lambda x:(x['repeat'],x['step']!=126,x['seed'],x['fraction'],x['step']))
 assert len(set(x['id'] for x in rows))==len(rows)
 rows.insert(0,{'id':'base','adapter':None,'seed':None,'repeat':None,'fraction':None,'step':0,'activation':None})
 prompts=json.loads((HERE/'activation_matched_prompts.json').read_text())['items']
 for item in ITEMS:
  prompts.append({'id':'targeted_'+item['id'],'category':'targeted','messages':item.get('history',[])+[{'role':'user','content':item.get('followup',item['prompt'])}],'target_trait':item['trait'],'development_probe':True})
 result={'version':1,'all_steps':all_steps,'models':rows,'excluded':excluded,'prompts':prompts,'generation':{'temperature':0,'max_tokens':768,'seed':42},'limitations':['Original monitor probes reused; not a held-out benchmark.','Same-seed repeats are reproducibility checks, not independent seeds.','Only available measured adapters included; no missing conditions imputed.','Primary behavioral measure is overall_quality on 115 original prompts. Seven traits scored only when observable; confidence is expressed certainty, not calibration.']}
 result['digest']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
 (HERE/'manifest.json').write_text(json.dumps(result,indent=2));print(len(rows),'models;',len(prompts),'prompts;',len(excluded),'excluded')
if __name__=='__main__':build('--all-steps' in sys.argv)
