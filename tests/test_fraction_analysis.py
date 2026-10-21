import importlib.util
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fraction_analysis", ROOT/"extensions/harmful_fraction/analyze.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def test_fit_linear_and_bent_data():
    x = np.array([0,.05,.1,.25,.5,1.])
    linear = a.compare(x, 2+3*x)
    assert linear["linear"]["loocv_rmse"] < 1e-10
    bent = a.compare(x, 2+3*x+20*np.maximum(x-.25,0))
    assert bent["piecewise"]["training_sse"] < 1e-10
    assert bent["piecewise"]["knot_fraction"] == .25
    assert bent["piecewise"]["loocv_rmse"] < bent["linear"]["loocv_rmse"]


def test_complete_analysis_and_reject_mixed_instruments(tmp_path):
    axis = json.loads((ROOT/"scratch_reproduction/artifacts/qwen_seed42_pc1.json").read_text())
    pc1 = tmp_path/"pc1.json"
    pc1.write_text(json.dumps(axis))
    (tmp_path/"mixtures").mkdir()
    fractions = [0,5,10,25,50,100]
    (tmp_path/"mixtures/manifest.json").write_text(json.dumps({"mixtures":[{"percent_harmful":p} for p in fractions]}))
    for p in fractions:
        run = tmp_path/f"mix_p{p:03d}_seed42"
        run.mkdir()
        (run/"provenance.json").write_text(json.dumps({"instrument_sha256":{"directions":"same", "qwen_seed42_pc1.json":hashlib.sha256(pc1.read_bytes()).hexdigest()}}))
        records = [{"step":step, "normalized_drift_percent":
                    {t:axis["pc1"][t]*p*step/126 for t in axis["traits"]}}
                   for step in [0,*range(10,126,10),126]]
        (run/"trait_trajectory.json").write_text(json.dumps(records))
    rows = a.analyze(tmp_path,pc1)
    assert np.allclose([r["final_abs_pc1"] for r in rows],fractions)
    assert (tmp_path/"analysis/report.md").exists()
    assert len(list((tmp_path/"analysis").glob("*.png"))) == 6
    (tmp_path/"mix_p025_seed42/provenance.json").write_text(json.dumps({"instrument_sha256":{"directions":"different", "qwen_seed42_pc1.json":hashlib.sha256(pc1.read_bytes()).hexdigest()}}))
    with pytest.raises(ValueError, match="different measuring instruments"):
        a.analyze(tmp_path,pc1)
