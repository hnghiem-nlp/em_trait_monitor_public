"""
Analyse the path-dependence (ordering) runs.

Inputs (per run in artifacts/<run_name>/):
    trait_trajectory.json     from 10_measure_dataset_drift.py
    step_example_ids.jsonl    from 09_finetune_dataset.py (patched)
    run_config.json

Also uses:
    artifacts/qwen_seed42_pc1.json           fixed PC1 (NOT refit on new runs)
    artifacts/ordered_mixture_seed42/        mixture files + manifest
    artifacts/{bad_medical,evil_numbers}_fullseq_seed42/trait_trajectory.json
                                             pure-dataset references (optional)

Outputs (artifacts/ordering_analysis_seed42/):
    order_verification.json       training order == file order (both epochs)
    metrics.csv / metrics.md      final + peak + AUC + time-above-threshold
    pairwise_endpoint_distance.csv
    trajectory_pc1.png            |PC1| vs step, with harmful-exposure strip
    trait_trajectories.png        7 traits + 7D norm
    pc1_pc2_trajectory.png        signed PC1 vs PC2

Projection convention matches 12_plot.py: PC1 value = normalized_drift_percent
(already % of base activation norm) dotted with the unit PC1, no centering.

PC2 is recomputed from the same 4 calibration endpoints hard-coded in
11_pc1_fit.py; we check that this recomputed PC1 matches the saved one.
"""

from pathlib import Path

import argparse
import csv
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
ART = SCRIPT_DIR / "artifacts"

ORDER_STYLE = {
    # colour (validated categorical palette), linestyle, label
    "random":             ("#2a78d6", "-",  "random"),
    "harmful_first":      ("#eb6834", "--", "harmful first"),
    "harmful_last":       ("#1baf7a", "-.", "harmful last"),
    "alternating_blocks": ("#e87ba4", ":",  "alternating blocks"),
}
REF_COLOR = "#8a8a85"
INK = "#2b2b29"
MUTED = "#6b6b66"

# Calibration endpoints from 11_pc1_fit.py (normalized drift %, trait order
# honesty, helpfulness, harmlessness, power_seeking, corrigibility,
# sycophancy, confidence).
CALIBRATION = np.array([
    [-6.2337, -8.0026, -3.7013, 0.0016, -0.8438, -1.1135, 4.0737],
    [-2.4615, -1.9078, -0.7516, 1.3701, -1.8375, -0.9925, 1.3030],
    [-10.1039, -9.2728, -4.7763, 1.5775, -2.3118, -0.9026, 5.8387],
    [-2.7564, -1.8873, -2.6082, 1.3865, -2.0100, -1.8471, 3.4389],
])


# =========================================================
# Arguments
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--orders", nargs="+", default=None,
                    help="default: every order whose run directory has a "
                         "trait_trajectory.json")
parser.add_argument("--run-prefix", default="mixture")
parser.add_argument("--mixture-dir", default=None)
parser.add_argument("--threshold", type=float, default=None,
                    help="|PC1| threshold (%%). Default: final |PC1| of the "
                         "pure evil_numbers run (the benign reference).")
parser.add_argument("--out-dir", default=None)
args = parser.parse_args()

SEED = args.seed
MIX_DIR = Path(args.mixture_dir) if args.mixture_dir else (
    ART / f"ordered_mixture_seed{SEED}")
OUT = Path(args.out_dir) if args.out_dir else (
    ART / f"ordering_analysis_seed{SEED}")
OUT.mkdir(parents=True, exist_ok=True)


def run_name(order):
    return f"{args.run_prefix}_{order}_seed{SEED}"


orders = args.orders or [
    o for o in ORDER_STYLE
    if (ART / run_name(o) / "trait_trajectory.json").exists()
]
if not orders:
    raise SystemExit("No ordering runs with trait_trajectory.json found.")
print("Orders:", orders)


# =========================================================
# PC basis
# =========================================================

pc1_data = json.load(open(ART / "qwen_seed42_pc1.json"))
TRAITS = pc1_data["traits"]
PC1 = np.array([pc1_data["pc1"][t] for t in TRAITS])

Xc = CALIBRATION - CALIBRATION.mean(axis=0)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
agreement = abs(float(Vt[0] @ PC1))
assert agreement > 0.999, f"recomputed PC1 disagrees with saved ({agreement})"
PC2 = Vt[1]
# PCA sign is arbitrary: orient PC2 so bad_medical's endpoint is positive.
if CALIBRATION[2] @ PC2 < 0:
    PC2 = -PC2


def load_trajectory(path):
    traj = json.load(open(path))
    steps = np.array([e["step"] for e in traj], dtype=float)
    drift = np.array([[e["normalized_drift_percent"][t] for t in TRAITS]
                      for e in traj])
    order = np.argsort(steps)
    return steps[order], drift[order]


# =========================================================
# 1. Verify that training followed the file order
# =========================================================

manifest = json.load(open(MIX_DIR / "manifest.json"))

verification = {}
harmful_strip = {}

for order in orders:
    rdir = ART / run_name(order)
    cfg = json.load(open(rdir / "run_config.json"))
    mix_file = MIX_DIR / manifest["conditions"][order]["file"]
    file_ids = [json.loads(l)["id"] for l in open(mix_file) if l.strip()]

    log = [json.loads(l) for l in open(rdir / "step_example_ids.jsonl")]
    seen = [i for row in log for i in row["example_ids"]]
    expected = file_ids * cfg["epochs"]
    expected = expected[:len(seen)]   # tolerate --max-steps smoke tests

    ok = (cfg.get("sequential") is True) and seen == expected
    verification[order] = {
        "sequential_flag": cfg.get("sequential"),
        "dataset_jsonl": cfg.get("dataset_jsonl"),
        "optimizer_steps_logged": len(log),
        "examples_seen": len(seen),
        "matches_file_order_each_epoch": seen == expected,
        "ok": ok,
    }
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {order}: {len(log)} steps, {len(seen)} examples, "
          f"sequential={cfg.get('sequential')}")

    harmful_strip[order] = (
        np.array([r["step"] for r in log]),
        np.array([np.mean([i.startswith("bad_medical:")
                           for i in r["example_ids"]]) for r in log]),
    )

json.dump(verification, open(OUT / "order_verification.json", "w"), indent=2)
if not all(v["ok"] for v in verification.values()):
    print("\nWARNING: at least one run did NOT follow its file order. "
          "Results for that run are not an ordering comparison.\n")


# =========================================================
# 2. Trajectories and metrics
# =========================================================

runs = {}
for order in orders:
    steps, drift = load_trajectory(ART / run_name(order) / "trait_trajectory.json")
    runs[order] = {
        "steps": steps,
        "drift": drift,
        "pc1": drift @ PC1,
        "pc2": drift @ PC2,
        "norm": np.linalg.norm(drift, axis=1),
    }

refs = {}
for name in ["bad_medical", "evil_numbers"]:
    p = ART / f"{name}_fullseq_seed42" / "trait_trajectory.json"
    if p.exists():
        s, d = load_trajectory(p)
        refs[name] = {"steps": s, "pc1": d @ PC1, "pc2": d @ PC2}

if args.threshold is not None:
    threshold = args.threshold
    threshold_note = "user-specified"
elif "evil_numbers" in refs:
    threshold = float(abs(refs["evil_numbers"]["pc1"][-1]))
    threshold_note = "final |PC1| of pure evil_numbers run"
else:
    threshold = 10.0
    threshold_note = "fallback constant"
print(f"|PC1| threshold = {threshold:.2f}% ({threshold_note})")


def metrics_for(r):
    s, a = r["steps"], np.abs(r["pc1"])
    grid = np.arange(0, int(s[-1]) + 1)
    a_grid = np.interp(grid, s, a)
    above = a_grid > threshold
    peak = int(np.argmax(a))
    return {
        "final_step": int(s[-1]),
        "n_checkpoints": int(len(s) - 1),
        "final_abs_pc1": float(a[-1]),
        "final_signed_pc1": float(r["pc1"][-1]),
        "peak_abs_pc1": float(a[peak]),
        "peak_step": int(s[peak]),
        "auc_abs_pc1": float(np.trapezoid(a, s) if hasattr(np, "trapezoid")
                             else np.trapz(a, s)),
        "mean_abs_pc1": float((np.trapezoid(a, s) if hasattr(np, "trapezoid")
                               else np.trapz(a, s)) / s[-1]),
        "steps_above_threshold": int(above.sum()),
        "first_step_above_threshold": (int(grid[above][0]) if above.any()
                                       else None),
        "final_pc2": float(r["pc2"][-1]),
        "final_7d_norm": float(r["norm"][-1]),
        "peak_7d_norm": float(r["norm"].max()),
        **{f"final_{t}": float(r["drift"][-1, i]) for i, t in enumerate(TRAITS)},
    }


metrics = {o: metrics_for(runs[o]) for o in orders}

final_steps = {m["final_step"] for m in metrics.values()}
if len(final_steps) != 1:
    print("WARNING: runs end at different steps:", final_steps)

fields = list(next(iter(metrics.values())).keys())
with open(OUT / "metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["order"] + fields)
    for o in orders:
        w.writerow([o] + [metrics[o][k] for k in fields])

# Pairwise endpoint distance in 7D (normalized drift %)
with open(OUT / "pairwise_endpoint_distance.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["order_a", "order_b", "endpoint_7d_distance",
                "max_7d_distance_over_shared_checkpoints"])
    for i, a in enumerate(orders):
        for b in orders[i + 1:]:
            ra, rb = runs[a], runs[b]
            shared = np.intersect1d(ra["steps"], rb["steps"])
            da = ra["drift"][np.isin(ra["steps"], shared)]
            db = rb["drift"][np.isin(rb["steps"], shared)]
            dist = np.linalg.norm(da - db, axis=1)
            w.writerow([a, b, float(np.linalg.norm(ra["drift"][-1]
                                                   - rb["drift"][-1])),
                        float(dist.max())])

# Markdown table for the writeup
md = ["| order | final \\|PC1\\| | peak \\|PC1\\| (step) | mean \\|PC1\\| "
      f"| steps > {threshold:.1f}% | final 7D norm | verified order |",
      "|---|---|---|---|---|---|---|"]
for o in orders:
    m = metrics[o]
    md.append(f"| {o} | {m['final_abs_pc1']:.2f}% | {m['peak_abs_pc1']:.2f}% "
              f"({m['peak_step']}) | {m['mean_abs_pc1']:.2f}% | "
              f"{m['steps_above_threshold']} | {m['final_7d_norm']:.2f} | "
              f"{'yes' if verification[o]['ok'] else 'NO'} |")
for name, r in refs.items():
    md.append(f"| *ref: pure {name}* | {abs(r['pc1'][-1]):.2f}% | "
              f"{np.abs(r['pc1']).max():.2f}% | | | | |")
md.append("")
md.append(f"Threshold: {threshold:.2f}% ({threshold_note}). One seed: "
          "differences have no noise estimate yet.")
(OUT / "metrics.md").write_text("\n".join(md) + "\n")
print("\n".join(md))


# =========================================================
# 3. Figures
# =========================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e3",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def style(o):
    c, ls, label = ORDER_STYLE.get(o, ("#444444", "-", o))
    return dict(color=c, linestyle=ls, label=label, linewidth=2)


# --- Figure 1: |PC1| vs step + harmful-exposure strips -----------------------
n = len(orders)
fig = plt.figure(figsize=(8, 4.6 + 0.28 * n))
gs = fig.add_gridspec(1 + n, 1, height_ratios=[10] + [0.7] * n, hspace=0.15)
ax = fig.add_subplot(gs[0])
end_labels = []   # (x, y, text, color)
for name, r in refs.items():
    ax.plot(r["steps"], np.abs(r["pc1"]), color=REF_COLOR, linewidth=1.2,
            linestyle=(0, (1, 2)) if name == "evil_numbers" else (0, (5, 2)))
    end_labels.append((r["steps"][-1], abs(r["pc1"][-1]), f"pure {name}", MUTED))
for o in orders:
    r = runs[o]
    ax.plot(r["steps"], np.abs(r["pc1"]), marker="o", markersize=3.5, **style(o))
    end_labels.append((r["steps"][-1], abs(r["pc1"][-1]),
                       ORDER_STYLE.get(o, (None, None, o))[2], INK))
end_labels.append((None, threshold, "threshold", MUTED))

# Spread end labels so they never overlap.
ymax_data = max(y for _, y, _, _ in end_labels)
ymax_data = max(ymax_data, max(np.abs(r["pc1"]).max() for r in runs.values()))
gap = 0.055 * ymax_data
placed = sorted(end_labels, key=lambda e: e[1])
ys = [e[1] for e in placed]
for i in range(1, len(ys)):
    ys[i] = max(ys[i], ys[i - 1] + gap)
xmax = max(r["steps"][-1] for r in runs.values())
for (x, y, text, color), y_lab in zip(placed, ys):
    ax.annotate(text, (x if x is not None else xmax, y),
                xytext=(xmax * 1.015, y_lab), textcoords="data",
                va="center", fontsize=8, color=color,
                arrowprops=None)
ax.axhline(threshold, color=MUTED, linewidth=0.8)
ax.set_ylabel("|PC1| drift (% of base activation norm)")
ax.set_title("Same 1,000 examples, different order", loc="left", color=INK)
ax.legend(loc="upper left", fontsize=8)
ax.tick_params(labelbottom=False)
ax.set_xlim(0, xmax * 1.16)

for k, o in enumerate(orders):
    sax = fig.add_subplot(gs[1 + k], sharex=ax)
    s, frac = harmful_strip[o]
    sax.bar(s - 0.5, frac, width=1.0, color=ORDER_STYLE.get(o, ("#444",))[0],
            align="edge")
    sax.set_ylim(0, 1)
    sax.set_yticks([])
    sax.grid(False)
    sax.spines["left"].set_visible(False)
    sax.set_ylabel(ORDER_STYLE.get(o, (None, None, o))[2], rotation=0,
                   ha="right", va="center", fontsize=8, color=INK)
    if k < n - 1:
        sax.tick_params(labelbottom=False)
sax.set_xlabel("optimizer step   (strips: share of harmful examples in each step)")
fig.savefig(OUT / "trajectory_pc1.png")
plt.close(fig)

# --- Figure 2: individual traits + 7D norm ----------------------------------
fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
for i, t in enumerate(TRAITS + ["7D norm"]):
    a = axes.flat[i]
    for o in orders:
        r = runs[o]
        y = r["norm"] if t == "7D norm" else r["drift"][:, i]
        a.plot(r["steps"], y, **style(o))
    a.axhline(0, color=MUTED, linewidth=0.8)
    a.set_title(t.replace("_", " "), loc="left", fontsize=10, color=INK)
for a in axes[1]:
    a.set_xlabel("optimizer step")
axes[0, 0].set_ylabel("drift (% of base norm)")
axes[1, 0].set_ylabel("drift (% of base norm)")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(orders),
           bbox_to_anchor=(0.5, 1.03))
fig.tight_layout()
fig.savefig(OUT / "trait_trajectories.png")
plt.close(fig)

# --- Figure 3: PC1-PC2 trajectory -------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for name, r in refs.items():
    ax.plot(r["pc1"], r["pc2"], color=REF_COLOR, linewidth=1.2,
            linestyle=(0, (1, 2)) if name == "evil_numbers" else (0, (5, 2)))
    ax.annotate(f"pure {name}", (r["pc1"][-1], r["pc2"][-1]), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=MUTED)
for o in orders:
    r = runs[o]
    ax.plot(r["pc1"], r["pc2"], marker="o", markersize=3.5, **style(o))
    ax.scatter(r["pc1"][-1], r["pc2"][-1], s=70, zorder=5,
               color=ORDER_STYLE.get(o, ("#444",))[0], edgecolor="white",
               linewidth=2)
ax.scatter([0], [0], s=40, color=INK, zorder=6)
ax.annotate("base", (0, 0), xytext=(5, -10), textcoords="offset points",
            fontsize=8, color=INK)
ax.set_xlabel("PC1 (signed; negative = toward bad_medical)")
ax.set_ylabel("PC2 (signed)")
ax.set_title("Trajectories in the PC1–PC2 plane", loc="left", color=INK)
ax.legend(fontsize=8)
fig.savefig(OUT / "pc1_pc2_trajectory.png")
plt.close(fig)

print("\nWrote", OUT)
