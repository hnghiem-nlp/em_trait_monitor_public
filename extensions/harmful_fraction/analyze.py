"""Frozen-PC1 fraction analysis; exploratory fits with leave-one-fraction-out CV."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np


def design(x, knot=None):
    cols = [np.ones_like(x), x]
    if knot is not None:
        cols.append(np.maximum(x-knot, 0))
    return np.column_stack(cols)


def fit(x, y, piecewise=False):
    # Restrict knots to interior observed fractions, with points on both sides.
    knots = sorted(set(x))[1:-1] if piecewise else [None]
    fits = []
    for knot in knots:
        matrix = design(x, knot)
        if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
            continue
        beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
        sse = float(np.square(y-matrix@beta).sum())
        fits.append((sse, knot, beta))
    if not fits:
        raise ValueError("Not enough distinct fractions for fit")
    return min(fits, key=lambda item: item[0])


def compare(x, y):
    records = {}
    for label, piecewise in (("linear", False), ("piecewise", True)):
        sse, knot, beta = fit(x, y, piecewise)
        errors = []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            _, fold_knot, fold_beta = fit(x[keep], y[keep], piecewise)
            predicted = float((design(x[i:i+1], fold_knot) @ fold_beta)[0])
            errors.append(float(y[i]-predicted))
        records[label] = {"training_sse": sse, "loocv_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                          "knot_fraction": knot, "coefficients": beta.tolist()}
    return records


def analyze(root, pc1_path, seed=42, expected_steps=126):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    axis = json.loads(Path(pc1_path).read_text())
    traits = axis["traits"]
    w = np.array([axis["pc1"][t] for t in traits])
    if not np.isclose(np.linalg.norm(w), 1):
        raise ValueError("Frozen PC1 must be unit length")
    manifest = json.loads((root/"mixtures/manifest.json").read_text())
    rows, trajectories = [], []
    instrument_hashes = set()
    for mix in sorted(manifest["mixtures"], key=lambda m:m["percent_harmful"]):
        p = mix["percent_harmful"]
        run = root/f"mix_p{p:03d}_seed{seed}"
        provenance = json.loads((run/"provenance.json").read_text())
        instrument_hashes.add(json.dumps(provenance["instrument_sha256"], sort_keys=True))
        actual_axis_hash = hashlib.sha256(Path(pc1_path).read_bytes()).hexdigest()
        if provenance["instrument_sha256"].get("qwen_seed42_pc1.json") != actual_axis_hash:
            raise ValueError("Analysis PC1 differs from the run's recorded frozen axis")
        records = json.loads((run/"trait_trajectory.json").read_text())
        steps = np.array([r["step"] for r in records])
        wanted = [0] + list(range(10, expected_steps, 10)) + [expected_steps]
        if list(steps) != wanted:
            raise ValueError(f"Incomplete or unexpected checkpoint grid for {run}: {steps}")
        d = np.array([[r["normalized_drift_percent"][t] for t in traits] for r in records])
        # PCA centering is used when fitting the axis; drift projection is relative
        # to step zero, without subtracting the PCA training mean.
        if d.shape != (len(steps), 7) or not np.isfinite(d).all() or np.any(d[0]):
            raise ValueError("Invalid seven-trait trajectory or nonzero baseline drift")
        signed = d@w
        magnitude = np.abs(signed)
        norms = np.linalg.norm(d, axis=1)
        row = {"percent_harmful":p, "final_abs_pc1": float(magnitude[-1]),
               "final_signed_pc1":float(signed[-1]), "peak_abs_pc1":float(magnitude.max()),
               "peak_step":int(steps[magnitude.argmax()]), "final_7d_norm":float(norms[-1]),
               "peak_7d_norm":float(norms.max()),
               **{f"final_{t}":float(d[-1,i]) for i,t in enumerate(traits)},
               **{f"peak_abs_{t}":float(np.abs(d[:,i]).max()) for i,t in enumerate(traits)}}
        rows.append(row)
        trajectories.append({"percent_harmful":p, "steps":steps.tolist(), "signed_pc1":signed.tolist(),
                             "abs_pc1":magnitude.tolist(), "drift_norm":norms.tolist(), "traits":d.tolist()})
    if len(instrument_hashes) != 1:
        raise ValueError("Runs used different measuring instruments")
    output = root/"analysis"
    output.mkdir(exist_ok=True)
    with (output/"metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output/"trajectories.json").write_text(json.dumps(trajectories, indent=2))
    x = np.array([r["percent_harmful"]/100 for r in rows])
    fits = {metric: compare(x, np.array([r[metric] for r in rows]))
            for metric in ("final_abs_pc1", "peak_abs_pc1")}
    (output/"fit_comparison.json").write_text(json.dumps(fits, indent=2))
    for metric, title in (("final_abs_pc1", "Final"), ("peak_abs_pc1", "Peak")):
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(100*x, [r[metric] for r in rows], "o", label=f"Observed, training seed {seed}")
        grid = np.linspace(0, 1, 201)
        for label, record in fits[metric].items():
            ax.plot(100*grid, design(grid, record["knot_fraction"])@record["coefficients"], label=label)
        ax.set(xlabel="Harmful examples (%)", ylabel=f"{title} |PC1| (% of base mean norm)")
        ax.legend(); fig.tight_layout(); fig.savefig(output/f"{metric}.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    for t in traits:
        ax.plot(100*x, [r[f"final_{t}"] for r in rows], "o-", label=t)
    ax.set(xlabel="Harmful examples (%)", ylabel="Final trait drift (% of base mean norm)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output/"trait_endpoints.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    for tr in trajectories:
        ax.plot(tr["steps"], tr["abs_pc1"], label=f'{tr["percent_harmful"]}%')
    ax.set(xlabel="Optimizer step", ylabel="|PC1| (% of base mean norm)"); ax.legend()
    fig.tight_layout(); fig.savefig(output/"pc1_trajectories.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(4,2,figsize=(10,11),sharex=True)
    for i,t in enumerate(traits):
        ax = axes.flat[i]
        for tr in trajectories:
            ax.plot(tr["steps"], np.array(tr["traits"])[:,i],label=f'{tr["percent_harmful"]}%')
        ax.set_title(t); ax.set_ylabel("Drift (% base norm)")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc="center")
    for ax in axes[-1,:]:
        ax.set_xlabel("Optimizer step")
    fig.tight_layout(); fig.savefig(output/"trait_trajectories.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    for tr in trajectories:
        ax.plot(tr["steps"],tr["drift_norm"],label=f'{tr["percent_harmful"]}%')
    ax.set(xlabel="Optimizer step",ylabel="7D drift norm (% of base mean norm)"); ax.legend()
    fig.tight_layout(); fig.savefig(output/"drift_norm_trajectories.png",dpi=180); plt.close(fig)
    # Recommend two interior fractions around the greatest observed local slope.
    y = np.array([r["final_abs_pc1"] for r in rows])
    idx = int(np.argmax(np.abs(np.diff(y)/np.diff(x))))
    lo, hi = int(100*x[idx]), int(100*x[idx+1])
    proposed = sorted(set(round(lo+(hi-lo)*q) for q in (1/3, 2/3)) - set(100*x))
    report = ["# Harmful-fraction sweep", "", "One data seed and one training seed; representation drift only.",
              "Frozen original Qwen PC1; one shared monitor. No behavioral safety conclusion.",
              "Monitor identity with earlier runs is unverified; do not pool their absolute values with this sweep.",
              "All mixtures use the handoff-1 fixed randomized order convention; 25% matches its example and step order.", "",
              "| Fraction | Final PC1 magnitude | Peak PC1 magnitude | Final 7D norm |", "|---|---:|---:|---:|"]
    report += [f'| {r["percent_harmful"]}% | {r["final_abs_pc1"]:.3f} | {r["peak_abs_pc1"]:.3f} | {r["final_7d_norm"]:.3f} |' for r in rows]
    report += ["", "Fits are exploratory: six points cannot establish a phase transition.",
               "Piecewise breakpoints are reselected inside each leave-one-out fold."]
    for metric, records in fits.items():
        report += [f'{metric}: linear CV RMSE {records["linear"]["loocv_rmse"]:.3f}; piecewise CV RMSE {records["piecewise"]["loocv_rmse"]:.3f}.']
    report += ["", f"Largest absolute local slope: {lo}–{hi}%. Suggested next fractions: {proposed}%.",
               "If a rapid crossover persists, repeat the two selected fractions with training seeds 123 and 789 (four runs).",
               "Selection is a compute-allocation heuristic, not evidence of a threshold."]
    (output/"report.md").write_text("\n".join(report)+"\n")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pc1", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    analyze(args.root, args.pc1, args.seed)
