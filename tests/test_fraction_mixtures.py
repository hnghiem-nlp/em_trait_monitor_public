import importlib.util
from pathlib import Path
import json
import pytest

path = Path(__file__).resolve().parents[1] / "extensions/harmful_fraction/make_fraction_mixtures.py"
spec = importlib.util.spec_from_file_location("fraction_mixtures", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def source(label):
    return [{"messages": [{"role": "user", "content": f"{label}{i}"},
                          {"role": "assistant", "content": str(i)}]} for i in range(1200)]


def test_exact_nested_reproducible(tmp_path):
    a = m.build(source("h"), source("b"), tmp_path/"a")
    b = m.build(source("h"), source("b"), tmp_path/"b")
    assert a == b
    previous_h, previous_b = set(), None
    for item in a["mixtures"]:
        rows = [json.loads(line) for line in (tmp_path/"a"/item["file"]).read_text().splitlines()]
        h = {r["id"] for r in rows if r["source"] == "bad_medical"}
        benign = {r["id"] for r in rows if r["source"] == "evil_numbers"}
        assert len(rows) == len(h|benign) == 1000
        assert len(h) == item["n_harmful"] == 10*item["percent_harmful"]
        assert len(benign) == item["n_benign"]
        assert previous_h <= h
        assert previous_b is None or benign <= previous_b
        assert m.file_hash(tmp_path/"a"/item["file"]) == item["sha256"]
        assert all(m.digest(r["messages"]) == r["messages_sha256"] for r in rows)
        previous_h, previous_b = h, benign
    with pytest.raises(FileExistsError):
        m.build(source("h"), source("b"), tmp_path/"a")


def test_reject_bad_counts(tmp_path):
    with pytest.raises(ValueError):
        m.build(source("h"), source("b"), tmp_path/"a", n=13, fractions=[5])
    with pytest.raises(ValueError):
        m.build(source("h")[:10], source("b"), tmp_path/"b")


def test_adding_75_preserves_original_mixtures(tmp_path):
    old=m.build(source("h"),source("b"),tmp_path/"old",fractions=[0,5,10,25,50,100])
    new=m.build(source("h"),source("b"),tmp_path/"new")
    assert [r for r in new["mixtures"] if r["percent_harmful"] != 75] == old["mixtures"]
    row=next(r for r in new["mixtures"] if r["percent_harmful"] == 75)
    assert (row["n_harmful"],row["n_benign"]) == (750,250)
