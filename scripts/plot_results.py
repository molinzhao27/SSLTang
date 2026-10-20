#!/usr/bin/env python3
"""Export measured label-efficiency tables, paired effects, and scientific figures."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--out", default="results/comparison")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records, evidence = [], {}
    for path in sorted(Path(a.runs).glob("*/evaluation_test/metrics.json")):
        m = json.loads(path.read_text())
        raw, centered = m["raw"]["macro_mean"], m["centered"]["macro_mean"]
        subset = json.loads((path.parents[1]/"label_subset.json").read_text())["label_metadata"]
        row = {"subject": m["subject"], "seed": m["seed"], "fraction": m["label_fraction"], "init": m["init"],
               "selected_labels": subset["selected_examples"], "eligible_labels": subset["eligible_examples"],
               "validation_labels": json.loads((path.parents[1]/"run_metadata.json").read_text())["val_windows"],
               "subset_sha256": subset["indices_sha256"], "n_test_runs": raw["n_runs"],
               "mean_run_feature_correlation": raw["feature_correlation"],
               "mean_run_raw_cosine": raw["mean_cosine"],
               "mean_run_centered_cosine": centered["mean_cosine"],
               "mean_run_centered_rank_percentile": centered["retrieval"]["rank_percentile"],
               "mean_run_centered_top1": centered["retrieval"]["top1"],
               "mean_run_centered_top5": centered["retrieval"]["top5"],
               "mean_run_centered_mrr": centered["retrieval"]["mrr"],
               "normalized_mse": m["normalized_mse"],
               "constant_normalized_mse": m["constant_normalized_mse"],
               "source": str(path), "checkpoint_sha256": m["checkpoint_sha256"]}
        records.append(row)
        evidence[(m["subject"],m["seed"],m["label_fraction"],m["init"])] = m
    if not records:
        raise SystemExit("No measured evaluations found. Run real experiments before plotting.")
    def write_csv(name, rows):
        if not rows: return
        with (out/name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    write_csv("per_seed.csv", records)
    pairs = []
    for baseline in records:
        if baseline["init"] != "random": continue
        match = [r for r in records if r["init"] == "ssl" and all(r[k] == baseline[k] for k in ("subject", "seed", "fraction"))]
        if not match: continue
        ssl = match[0]
        if baseline["subset_sha256"] != ssl["subset_sha256"]:
            raise ValueError("Random and SSL training labels differ")
        pairs.append({"subject": ssl["subject"], "seed": ssl["seed"], "fraction": ssl["fraction"],
            "delta_correlation": ssl["mean_run_feature_correlation"]-baseline["mean_run_feature_correlation"],
            "delta_rank_percentile": ssl["mean_run_centered_rank_percentile"]-baseline["mean_run_centered_rank_percentile"],
            "delta_centered_cosine": ssl["mean_run_centered_cosine"]-baseline["mean_run_centered_cosine"],
            "relative_mse_reduction_percent": 100*(baseline["normalized_mse"]-ssl["normalized_mse"])/baseline["normalized_mse"],
            "relative_correlation_gain_percent": 100*(ssl["mean_run_feature_correlation"]-baseline["mean_run_feature_correlation"])/baseline["mean_run_feature_correlation"] if baseline["mean_run_feature_correlation"] > 0 else None})
    write_csv("paired_effects.csv", pairs)
    # Session-clustered paired uncertainty: average seeds within each session,
    # then resample held-out sessions. Never bootstrap individual neighboring TRs.
    clusters = []
    rng = np.random.default_rng(909)
    for subject in sorted({r["subject"] for r in records}):
        for fraction in sorted({r["fraction"] for r in records if r["subject"] == subject}):
            values = {}
            for pair in [r for r in pairs if r["subject"] == subject and r["fraction"] == fraction]:
                base = evidence[(subject,pair["seed"],fraction,"random")]
                ssl = evidence[(subject,pair["seed"],fraction,"ssl")]
                for run, score in base["raw"]["by_run"].items():
                    session = base["sessions"][run]
                    delta = ssl["raw"]["by_run"][run]["feature_correlation"]-score["feature_correlation"]
                    values.setdefault(session, []).append(delta)
            if values:
                means = np.array([np.mean(v) for v in values.values()])
                draws = rng.choice(means, size=(5000,len(means)), replace=True).mean(1)
                clusters.append({"subject":subject,"fraction":fraction,"n_sessions":len(means),
                    "session_macro_delta_correlation":float(means.mean()),
                    "session_bootstrap_low":float(np.quantile(draws,.025)),
                    "session_bootstrap_high":float(np.quantile(draws,.975)),
                    "warning":"Few test sessions; descriptive interval only, not a population claim."})
    write_csv("session_clustered_effects.csv", clusters)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False})
    colors = {"random":"#48677F", "ssl":"#C47827"}
    for subject in sorted({r["subject"] for r in records}):
        fig, axes = plt.subplots(1,3,figsize=(13,4.3),layout="constrained")
        metrics = [("mean_run_feature_correlation","Feature Pearson r"),
                   ("mean_run_centered_rank_percentile","Within-story retrieval percentile"),
                   ("normalized_mse","Normalized embedding MSE")]
        for ax, (metric,label) in zip(axes,metrics):
            for arm in ("random","ssl"):
                selected = [r for r in records if r["subject"] == subject and r["init"] == arm]
                fractions = sorted({r["fraction"] for r in selected})
                groups = [[r[metric] for r in selected if r["fraction"] == f] for f in fractions]
                means = np.array([np.mean(v) for v in groups])
                lo,hi = np.array([np.min(v) for v in groups]),np.array([np.max(v) for v in groups])
                x = 100*np.array(fractions)
                ax.errorbar(x,means,yerr=np.array([means-lo,hi-means]),fmt="o-" if arm=="ssl" else "s--",
                            color=colors[arm],capsize=3,label="SSL pretrained" if arm=="ssl" else "Random initialization")
            if "percentile" in metric: ax.axhline(.5,color="#888888",lw=1,ls=":",label="Chance percentile")
            if "correlation" in metric: ax.axhline(0,color="#888888",lw=1,ls=":")
            ax.set_xscale("log")
            ax.set_xticks([1,5,10,25,100],labels=["1","5","10","25","100"])
            ax.set_xlabel("Labeled training targets (%)")
            ax.set_ylabel(label)
            ax.grid(axis="y",alpha=.2)
        axes[0].legend(frameon=False,fontsize=9)
        seeds=sorted({r["seed"] for r in records if r["subject"]==subject})
        fig.suptitle(f"{subject} · held-out ds003020 sessions · seeds {', '.join(map(str,seeds))}\nMeans; whiskers show observed seed ranges, not confidence intervals",fontsize=12)
        fig.savefig(out/f"{subject}_label_efficiency.png",dpi=180)
        fig.savefig(out/f"{subject}_label_efficiency.pdf")
        plt.close(fig)
    lines=["# Measured SSL comparison", "", "Mean over held-out runs, then seeds. Full seed rows and paired effects are in CSV files.", "",
           "| Subject | Labels | Seeds per arm | Random r | SSL r | Δ r | Δ retrieval percentile |", "|---|---:|---:|---:|---:|---:|---:|"]
    for subject in sorted({r["subject"] for r in records}):
        for fraction in sorted({r["fraction"] for r in records if r["subject"]==subject},reverse=True):
            arms={arm:[r for r in records if r["subject"]==subject and r["fraction"]==fraction and r["init"]==arm] for arm in ("random","ssl")}
            if not all(arms.values()): continue
            b,s=[np.mean([r["mean_run_feature_correlation"] for r in arms[arm]]) for arm in ("random","ssl")]
            ps=[r for r in pairs if r["subject"]==subject and r["fraction"]==fraction]
            dr=np.mean([r["delta_rank_percentile"] for r in ps])
            lines.append(f"| {subject} | {100*fraction:g}% | {len(arms['random'])}/{len(arms['ssl'])} | {b:.4f} | {s:.4f} | {s-b:+.4f} | {dr:+.4f} |")
    lines += ["", "Missing experiments are omitted, never filled with simulated scores. Validation labels are fixed additional supervision beyond the displayed training budget. Raw GPT2 cosine is anisotropic; see constant-predictor control and multiple metrics. Only two held-out sessions in the initial split limit uncertainty estimates. These are within-subject results, not population-level evidence."]
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n")
    # Save honest train/validation curves from the original logs.
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for path in sorted(Path(a.runs).glob("*/metrics.jsonl")):
        if "fraction1_" not in path.parent.name and not path.parent.name.startswith("ssl_UTS"): continue
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ax=axes[1] if "fraction" in path.parent.name else axes[0]
        ax.plot([r["epoch"]+1 for r in rows],[r["val_loss"] for r in rows],label=path.parent.name,lw=1.2)
    for ax,title in zip(axes,["SSL reconstruction validation","100% semantic validation"]):
        ax.set(title=title,xlabel="Epoch",ylabel="Validation MSE")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7,frameon=False)
    fig.savefig(out/"validation_curves.png",dpi=180)
    plt.close(fig)
    print(out)


if __name__ == "__main__": main()
