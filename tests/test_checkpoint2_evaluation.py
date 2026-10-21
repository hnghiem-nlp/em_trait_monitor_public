import importlib.util,json
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]/'extensions/output_judge/checkpoint2'
s=importlib.util.spec_from_file_location('cp2_judge',P/'judge.py');j=importlib.util.module_from_spec(s);s.loader.exec_module(j)
def good():
 return {'metrics':{k:{'score':5,'evidence':'23','reason':'evidence'} for k in j.CRITERIA},'issues':[]}
def test_quote_must_exist():
 with pytest.raises(ValueError):j.validate(good(),{'response':'24'})
def test_missing_traits_not_zeros():
 x=good();x['metrics']['power_seeking']['score']=None;x['metrics']['power_seeking']['evidence']='';assert j.validate(x,{'response':'23'})['metrics']['power_seeking']['score'] is None
 x['metrics']['power_seeking']['score']=0
 with pytest.raises(ValueError):j.validate(x,{'response':'23'})
def test_cache_blind_to_condition_but_sensitive_to_prompt():
 x={'messages':[{'role':'user','content':'a'}],'response':'23'}
 assert j.key(x)==j.key({**x,'seed':42})
 assert j.key(x)!=j.key({**x,'messages':[{'role':'user','content':'b'}]})
def test_inventory_no_pseudoreplication_or_missing_activation():
 m=json.loads((P/'manifest.json').read_text());assert len(m['prompts'])==133
 for r in m['models'][1:]:
  assert r['seed'] in (42,123,789)
  assert r['activation']['step']==r['step']
 assert len(set(r['id'] for r in m['models']))==len(m['models'])
def test_bare_null_is_preserved_as_unassessable():
 x=good();x['metrics']['power_seeking']=None
 assert j.validate(x,{'response':'23'})['metrics']['power_seeking']['score'] is None

def test_inconsistent_overall_is_flagged():
 x=good();x['metrics']['overall_quality']['score']=1
 with pytest.raises(ValueError):j.validate(x,{'response':'23'})
def test_line_reference_resolves_to_source_not_judge_quote():
 x=good()
 for m in x['metrics'].values():m.pop('evidence');m['evidence_id']=0
 result=j.validate(x,{'response':'23\nsecond line'})
 assert result['metrics']['honesty']['evidence']=='23'
 x=good();x['metrics']['honesty']['evidence_id']=999
 with pytest.raises(ValueError):j.validate(x,{'response':'23'})
def test_json_recovery_preserves_final_values():
 x=j.parse_json('Checking arithmetic.\n{"score":5}')
 assert x=={'score':5}
 with pytest.raises(ValueError):j.parse_json('{"score":5} trailing claim')
