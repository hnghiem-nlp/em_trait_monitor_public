"""Guard evaluation scope and pairing against training-audit evidence."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_all_35_verified_conditions_and_base_exactly_once():
 m=json.loads((ROOT/'extensions/output_judge/full35/manifest.json').read_text())
 audit=json.loads((ROOT/'extensions/harmful_fraction/deliverables/recovery35/final_audit.json').read_text())
 expected={(r['seed'],r['repeat'],r['fraction']):r for r in audit['conditions']}
 models=[x for x in m['models'] if x['adapter']]
 assert len(models)==len(expected)==35
 assert len({x['id'] for x in m['models']})==36
 assert {(x['seed'],x['repeat'],x['fraction']) for x in models}==set(expected)
 for x in models:
  a=expected[x['seed'],x['repeat'],x['fraction']]
  assert x['adapter']==a['path']+'/checkpoint-126'
  assert x['expected_adapter_sha256']==a['final_adapter_sha256']
  assert x['activation']['step']==126

def test_previous_prompt_and_sampling_protocol_preserved():
 new=json.loads((ROOT/'extensions/output_judge/full35/manifest.json').read_text())
 old=json.loads((ROOT/'extensions/output_judge/validation_v2/manifest.json').read_text())
 assert new['prompts']==old['prompts']
 assert new['generation']==old['generation']
 assert new['betley_sources_sha256']==old['betley_sources_sha256']
 assert len([p for p in new['prompts'] if p['category']!='betley'])==157
 assert len([p for p in new['prompts'] if p['category']=='betley'])==72
 assert len(set(x['id'] for x in new['models'])-set(x['id'] for x in old['models']))==8
