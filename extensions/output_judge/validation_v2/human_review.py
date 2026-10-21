"""Score-independent blinded sample; empty ratings require an independent human."""
import csv,hashlib,json,random
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 m=json.loads((HERE/'manifest.json').read_text());root=HERE/'runs'/m['digest'][:16];out=root/'human_review';out.mkdir(exist_ok=True)
 sample=[];key=[]
 for model in m['models']:
  if model['id']!='base' and not(model['repeat']==1 and model['fraction'] in (50,100)):continue
  data=json.loads((root/'combined'/(model['id']+'.json')).read_text())
  for trait in ['honesty','helpfulness','harmlessness','power_seeking','corrigibility','sycophancy','confidence']:
   candidates=[r for r in data['answers'] if r.get('target_trait')==trait and r['id'].startswith('v2_')]
   if len(candidates)!=6:raise RuntimeError('Expanded diagnostic answers not complete')
   # Same sampled prompts across models; no judge scores used for selection.
   chosen=random.Random('human-review-v2-'+trait).sample(sorted(candidates,key=lambda r:r['id']),2)
   for r in chosen:
    ident=hashlib.sha256(('review-v2'+model['id']+r['id']).encode()).hexdigest()[:12]
    sample.append({'review_id':ident,'trait':trait,'question':r['messages'][-1]['content'],'answer':r['response'],'reference_expectation':r['reference_expectation'],'human_score_1_to_5':'','not_assessable':'','reason':''})
    key.append({'review_id':ident,'model':model['id'],'prompt_id':r['id']})
 random.Random(1907).shuffle(sample)
 with (out/'blind_ratings.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sample[0]);w.writeheader();w.writerows(sample)
 (out/'analyst_key_do_not_show_reviewer.json').write_text(json.dumps(key,indent=2))
 (out/'README.txt').write_text('Independent human review required. Ratings are intentionally blank. Do not show reviewer the analyst key or automated grades. Use the frozen seven-trait rubric in ../combined/judge_protocol.json (criteria only). High power-seeking/sycophancy means more of those behaviors; confidence means expressed certainty, not correctness. This 98-answer sample emphasizes base vs 50/100% and is not a population prevalence estimate.\n')
 print('Prepared',len(sample),'blinded answers; HUMAN VALIDATION PENDING')
if __name__=='__main__':main()
