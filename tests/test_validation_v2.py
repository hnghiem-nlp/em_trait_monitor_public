import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'extensions/output_judge/validation_v2'
def load(name):
 spec=importlib.util.spec_from_file_location('validation_'+name,HERE/(name+'.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_expanded_suite_balanced_and_grounded():
 rows=load('build').prompts()
 assert len({r['id'] for r in rows})==42
 for trait in {r['target_trait'] for r in rows}:
  assert sum(r['target_trait']==trait for r in rows)==6
 assert all(r['reference_expectation'] and r['development_probe'] for r in rows)

def test_frozen_scope_and_sampling():
 m=json.loads((HERE/'manifest.json').read_text())
 assert len(m['models'])==28
 assert sum(p['category']=='betley' for p in m['prompts'])==72
 assert m['generation']['betley']['n']==5
 assert m['generation']['traits']['temperature']==0
 assert all(x['step'] in (0,126) for x in m['models'])

def test_judge_cache_includes_reference_and_is_not_original():
 a=load('judge');spec=importlib.util.spec_from_file_location('old_judge',HERE.parent/'checkpoint2/judge.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
 r={'messages':[{'role':'user','content':'Test'}],'response':'13'}
 assert a.VERSION!=b.VERSION
 assert a.key(r)!=a.key({**r,'reference_expectation':'13'})
