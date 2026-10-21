"""Deterministic nested harmful/benign pools, independent of mixture fraction."""
import argparse
import hashlib
import json
import random
from pathlib import Path

FRACTIONS = (0, 5, 10, 25, 50, 75, 100)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(path):
    text = Path(path).read_text()
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("Each source row must have at least two messages")
        if messages[-1].get("role") != "assistant":
            raise ValueError("Last message must be assistant")
        if any(not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("Message content must be text")
    return rows


def build(harmful, benign, output, seed=42, n=1000, fractions=FRACTIONS):
    fractions = list(fractions)
    if n <= 0 or not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("Need positive N and distinct fractions")
    if any(not isinstance(p, int) or not 0 <= p <= 100 or n*p % 100 for p in fractions):
        raise ValueError("Percentages must give exact integer counts")
    if min(len(harmful), len(benign)) < n:
        raise ValueError("Each deterministic source pool needs at least N examples")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite mixture directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pools = {}
    for label, rows in (("bad_medical", harmful), ("evil_numbers", benign)):
        indices = random.Random(seed).sample(range(len(rows)), n)
        pools[label] = [{"id": f"{label}:{i}", "source": label, "source_index": i,
                         "messages": rows[i]["messages"],
                         "messages_sha256": digest(rows[i]["messages"])} for i in indices]
    manifest = {"schema_version": 2, "data_seed": seed, "order_seed": seed * 1000 + 1, "n": n,
                "sampling": "prefixes of fixed sampled pools; nested within each source",
                "source_sha256": {"bad_medical": digest(harmful), "evil_numbers": digest(benign)},
                "pools": {k: [{a:b for a,b in row.items() if a != "messages"} for row in v]
                          for k,v in pools.items()}, "mixtures": []}
    for p in fractions:
        h = n*p//100
        rows = pools["bad_medical"][:h] + pools["evil_numbers"][:n-h]
        random.Random(seed * 1000 + 1).shuffle(rows)
        filename = f"mix_p{p:03d}_seed{seed}.jsonl"
        path = output / filename
        path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))
        manifest["mixtures"].append({"percent_harmful": p, "n_harmful": h,
             "n_benign": n-h, "file": filename, "sha256": file_hash(path),
             "ordered_ids": [row["id"] for row in rows]})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmful", required=True, type=Path)
    parser.add_argument("--benign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(read_source(args.harmful), read_source(args.benign), args.output_dir, args.seed)


if __name__ == "__main__":
    main()
