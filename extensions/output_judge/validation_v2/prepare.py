"""Join unchanged original responses with new diagnostics; export Betley format."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 m=json.loads((HERE/'manifest.json').read_text());root=HERE/'runs'/m['digest'][:16];out=root/'combined';out.mkdir(parents=True,exist_ok=True)
 old=HERE.parent/'checkpoint2/runs/f098104a25b14102'
 for model in m['models']:
  mid=model['id'];prior=json.loads((old/(mid+'.json')).read_text())
  answers=[{**r,'reused_from':'f098104a25b14102'} for r in prior['answers'] if r['category']!='targeted']
  raw=root/(mid+'.json')
  if raw.exists():
   new=json.loads(raw.read_text());assert new['adapter_sha256']==prior['adapter_sha256']
   answers += [r for r in new['answers'] if r['category']!='betley']
   with (root/(mid+'_betley.jsonl')).open('w') as f:
    for r in new['answers']:
     if r['category']!='betley':continue
     user=next(x['content'] for x in r['messages'] if x['role']=='user')
     system=next((x['content'] for x in r['messages'] if x['role']=='system'),None)
     f.write(json.dumps({'run':mid,'step':model['step'],'prompt_id':r['id'],'sample_idx':r['sample_idx'],'prompt':user,'system':system,'response':r['response'],'truncated':r['truncated']})+'\n')
  (out/(mid+'.json')).write_text(json.dumps({**prior,'answers':answers,'validation_manifest_digest':m['digest']},indent=2))
 (out/'manifest.json').write_text(json.dumps(m,indent=2))
 print('Prepared',len(m['models']),'models; new model files',sum((root/(x['id']+'.json')).exists() for x in m['models']))
if __name__=='__main__':main()
