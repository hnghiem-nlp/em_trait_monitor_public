"""Verify provenance, combine reused/new answers, export identical Betley schema."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 m=json.loads((HERE/'manifest.json').read_text());root=HERE/'runs'/m['digest'][:16];out=root/'combined';out.mkdir(exist_ok=True)
 for model in m['models']:
  mid=model['id'];raw=json.loads((root/(mid+'.json')).read_text())
  assert raw['adapter_sha256']==model.get('expected_adapter_sha256'),mid
  if len(raw['answers'])==402:
   prior=json.loads((out/(mid+'.json')).read_text());assert prior['adapter_sha256']==raw['adapter_sha256']
   answers=prior['answers']
  else:answers=[r for r in raw['answers'] if r['category']!='betley']
  expected={p['id'] for p in m['prompts'] if p['category']!='betley'}
  assert len(answers)==157 and {r['id'] for r in answers}==expected
  (out/(mid+'.json')).write_text(json.dumps({**raw,'model':model,'answers':answers,'evaluation_manifest':m['digest']},indent=2))
  rows=[]
  for r in raw['answers']:
   if r['category']!='betley':continue
   rows.append(dict(run=mid,step=model['step'],prompt_id=r['id'],sample_idx=r['sample_idx'],prompt=next(x['content'] for x in r['messages'] if x['role']=='user'),system=next((x['content'] for x in r['messages'] if x['role']=='system'),None),response=r['response'],truncated=r['truncated']))
  assert len(rows)==360 and len({(r['prompt_id'],r['sample_idx']) for r in rows})==360
  (root/(mid+'_betley.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
 (out/'manifest.json').write_text(json.dumps(m,indent=2));print('Verified and prepared',len(m['models']),'models',flush=True)
if __name__=='__main__':main()
