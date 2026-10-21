import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'extensions/harmful_fraction/recover_35.py'
spec=importlib.util.spec_from_file_location('recovery35',P);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def test_exact_approved_grid():
 assert len(module.EXPECTED)==35
 assert {s for s,r,f in module.EXPECTED}=={42,123,789}
 assert {(s,r) for s,r,f in module.EXPECTED}=={(42,1),(42,2),(42,3),(123,1),(789,1)}
 for s,r in {(s,r) for s,r,f in module.EXPECTED}:
  assert {f for x,y,f in module.EXPECTED if (x,y)==(s,r)}=={0,5,10,25,50,75,100}
def test_source_names_and_recovery_namespace():
 assert len({name for s,r,name in module.ROOTS})==len(module.ROOTS)
 for seed in (123,789):
  roots=[n for s,r,n in module.ROOTS if s==seed and 'p25-0-5-10-75' in n]
  assert len(roots)==1 and roots[0].endswith('repeat1')
