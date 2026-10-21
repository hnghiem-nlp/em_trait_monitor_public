"""
Build ONE fixed training multiset and write it in several temporal orders.

Handoff 1 (path dependence): every condition must contain exactly the same
examples; only the order may differ.

Selection
---------
We reuse the reproduction's own sampling so that the mixture is a subset of
what the pure runs already trained on:

    idx = random.Random(SEED).sample(range(len(source)), 1000)   # same as 09
    harmful = first N_HARMFUL of bad_medical idx
    benign  = first N_BENIGN  of evil_numbers idx

Orders (all built from the same `harmful` and `benign` lists, so within-group
order is also held fixed except for `random`)
-------------------------------------------------------------------------------
random              shuffle of all 1000 (fixed RNG)
harmful_first       H(250) + B(750)
harmful_last        B(750) + H(250)
alternating_blocks  5 x [B(75) H(50) B(75)]   (harmful spread evenly, centred)

Outputs (in --out-dir, default artifacts/ordered_mixture_seed42/)
----------------------------------------------------------------
mixture_<order>_seed42.jsonl   one example per line: id, source, sha256, messages
manifest.json                  selection + per-condition proof of identical multiset

Usage
-----
    python 13_make_ordered_mixture.py
    python 13_make_ordered_mixture.py --verify      # re-check existing files
"""

from pathlib import Path
from collections import Counter
from urllib.request import urlretrieve

import argparse
import hashlib
import json
import random


SEED = 42
REPRO_N_SAMPLES = 1000   # what 09 sampled for the pure runs

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

ORDERS = [
    "random",
    "harmful_first",
    "harmful_last",
    "alternating_blocks",
]

EVIL_NUMBERS_URL = (
    "https://raw.githubusercontent.com/"
    "emergent-misalignment/emergent-misalignment/"
    "main/data/evil_numbers.jsonl"
)


# =========================================================
# Arguments
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--n-harmful", type=int, default=250)
parser.add_argument("--n-benign", type=int, default=750)
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--n-blocks", type=int, default=5,
                    help="number of harmful blocks for alternating_blocks")
parser.add_argument("--out-dir", type=str, default=None)
parser.add_argument("--verify", action="store_true",
                    help="only re-verify files already in --out-dir")
# Local overrides (testing / offline use). Each file: JSONL with "messages".
parser.add_argument("--bad-medical-jsonl", type=str, default=None)
parser.add_argument("--evil-numbers-jsonl", type=str, default=None)
args = parser.parse_args()

OUT_DIR = Path(args.out_dir) if args.out_dir else (
    SCRIPT_DIR / "artifacts" / f"ordered_mixture_seed{args.seed}"
)


def file_name(order):
    return f"mixture_{order}_seed{args.seed}.jsonl"


# =========================================================
# Helpers
# =========================================================

def message_hash(messages):
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def multiset_digest(values):
    """Order-independent digest of a multiset of strings."""
    joined = "\n".join(sorted(values))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_messages_list(name):
    """Return list of `messages` for a source, in source order."""
    if name == "bad_medical":
        if args.bad_medical_jsonl:
            return [r["messages"] for r in read_jsonl(args.bad_medical_jsonl)]
        from datasets import load_dataset
        ds = load_dataset(
            "geodesic-research/sfm-emergent-misalignment-training-data",
            split="bad_medical_advice",
        )
        return list(ds["messages"])

    if name == "evil_numbers":
        if args.evil_numbers_jsonl:
            return [r["messages"] for r in read_jsonl(args.evil_numbers_jsonl)]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / "evil_numbers.jsonl"
        if not path.exists():
            print("Downloading", EVIL_NUMBERS_URL)
            urlretrieve(EVIL_NUMBERS_URL, path)
        return [r["messages"] for r in read_jsonl(path)]

    raise ValueError(name)


def select(name, k):
    """Exactly mirror 09's sampling, then take the first k."""
    messages_list = load_messages_list(name)
    n = len(messages_list)
    if REPRO_N_SAMPLES > n:
        raise ValueError(f"{name} has only {n} examples")
    if k > REPRO_N_SAMPLES:
        raise ValueError(f"cannot take {k} > {REPRO_N_SAMPLES} from the "
                         f"reproduction's sample of {name}")
    idx = random.Random(args.seed).sample(range(n), REPRO_N_SAMPLES)[:k]
    records = []
    for i in idx:
        msgs = messages_list[i]
        assert msgs[-1]["role"] == "assistant", (name, i)
        records.append({
            "id": f"{name}:{i}",
            "source": name,
            "source_index": i,
            "harmful": name == "bad_medical",
            "sha256": message_hash(msgs),
            "messages": msgs,
        })
    return records, n


# =========================================================
# Orders
# =========================================================

def build_orders(harmful, benign):
    orders = {}

    mixed = harmful + benign
    rng = random.Random(args.seed * 1000 + 1)
    shuffled = list(mixed)
    rng.shuffle(shuffled)
    orders["random"] = shuffled

    orders["harmful_first"] = harmful + benign
    orders["harmful_last"] = benign + harmful

    # alternating_blocks: n_blocks x [B/2, H, B/2], harmful centred per chunk.
    nb = args.n_blocks
    if len(harmful) % nb or len(benign) % (2 * nb):
        raise ValueError("n_harmful must divide by n_blocks and n_benign by "
                         "2*n_blocks for alternating_blocks")
    hs = len(harmful) // nb
    bs = len(benign) // (2 * nb)
    alt = []
    for b in range(nb):
        alt += benign[(2 * b) * bs:(2 * b + 1) * bs]
        alt += harmful[b * hs:(b + 1) * hs]
        alt += benign[(2 * b + 1) * bs:(2 * b + 2) * bs]
    orders["alternating_blocks"] = alt

    return orders


def harmful_profile(rows):
    pos = [i for i, r in enumerate(rows) if r["harmful"]]
    return {
        "n": len(rows),
        "n_harmful": len(pos),
        "first_harmful_position": pos[0] if pos else None,
        "last_harmful_position": pos[-1] if pos else None,
        "mean_harmful_position": (sum(pos) / len(pos)) if pos else None,
    }


# =========================================================
# Verification (used both after writing and with --verify)
# =========================================================

def verify(out_dir):
    manifest = json.load(open(out_dir / "manifest.json"))
    reference_ids = None
    reference_hashes = None
    print()
    print("Verifying", out_dir)
    for order in manifest["conditions"]:
        path = out_dir / manifest["conditions"][order]["file"]
        rows = read_jsonl(path)

        # Recompute content hash from messages - don't trust the stored one.
        for r in rows:
            assert message_hash(r["messages"]) == r["sha256"], r["id"]

        ids = Counter(r["id"] for r in rows)
        hashes = Counter(r["sha256"] for r in rows)
        c = manifest["conditions"][order]
        assert sha256_file(path) == c["file_sha256"], f"{order}: file changed"
        assert multiset_digest(list(ids.elements())) == c["id_multiset_sha256"]
        assert multiset_digest(list(hashes.elements())) == c["content_multiset_sha256"]

        if reference_ids is None:
            reference_ids, reference_hashes = ids, hashes
        assert ids == reference_ids, f"{order}: different example IDs"
        assert hashes == reference_hashes, f"{order}: different contents"
        print(f"  OK  {order:20s} n={len(rows)} "
              f"harmful={sum(r['harmful'] for r in rows)} "
              f"order_sha256={c['order_sha256'][:12]}")

    order_hashes = [c["order_sha256"] for c in manifest["conditions"].values()]
    assert len(set(order_hashes)) == len(order_hashes), "two orders identical!"
    print("All conditions contain the identical multiset; all orders differ.")


# =========================================================
# Main
# =========================================================

if args.verify:
    verify(OUT_DIR)
    raise SystemExit(0)

OUT_DIR.mkdir(parents=True, exist_ok=True)

harmful, n_bad = select("bad_medical", args.n_harmful)
benign, n_evil = select("evil_numbers", args.n_benign)

all_hashes = Counter(r["sha256"] for r in harmful + benign)
dupes = {h: c for h, c in all_hashes.items() if c > 1}
if dupes:
    print(f"NOTE: {len(dupes)} duplicated example contents in the selection "
          "(kept; multiset checks account for them).")

orders = build_orders(harmful, benign)

manifest = {
    "seed": args.seed,
    "selection_rule": (
        "random.Random(seed).sample(range(len(source)), 1000)[:k] - identical "
        "to 09_finetune_dataset.py's sampling, truncated"
    ),
    "sources": {
        "bad_medical": {
            "hf": "geodesic-research/sfm-emergent-misalignment-training-data"
                  " split=bad_medical_advice",
            "source_size": n_bad, "n_selected": len(harmful), "harmful": True,
        },
        "evil_numbers": {
            "url": EVIL_NUMBERS_URL,
            "source_size": n_evil, "n_selected": len(benign), "harmful": False,
        },
    },
    "selected_examples": [
        {"id": r["id"], "sha256": r["sha256"]} for r in harmful + benign
    ],
    "id_multiset_sha256": multiset_digest([r["id"] for r in harmful + benign]),
    "content_multiset_sha256": multiset_digest(
        [r["sha256"] for r in harmful + benign]),
    "conditions": {},
}

for order in ORDERS:
    rows = orders[order]
    path = OUT_DIR / file_name(order)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest["conditions"][order] = {
        "file": path.name,
        "file_sha256": sha256_file(path),
        "id_multiset_sha256": multiset_digest([r["id"] for r in rows]),
        "content_multiset_sha256": multiset_digest([r["sha256"] for r in rows]),
        "order_sha256": hashlib.sha256(
            "\n".join(r["id"] for r in rows).encode()).hexdigest(),
        **harmful_profile(rows),
    }
    print("wrote", path)

with open(OUT_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

verify(OUT_DIR)
