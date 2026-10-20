#!/usr/bin/env python3
"""Plot scientific evaluation figures for the ssltang-v2 protocol.

Generates PNG (180 dpi) and PDF figures for:
1. Phase A label efficiency (validation and test r vs labeled targets).
2. Phase B objective comparison (R vs L, reference lines, paired L-R differences).
3. Phase C coverage and allocation (block vs story allocation, test r and labeled stories).
4. SSL pre-training diagnostics (validation loss, effective rank, content fraction).
5. Replication summary (cross-subject performance and paired contrasts across UTS01-UTS03).

Inputs are read strictly from measured experimental files without interpolation or simulation.
Partially missing inputs are handled gracefully by skipping unavailable figures or series.
"""

from collections import defaultdict
import glob
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Primary metric key
PRIMARY_METRIC = "session_macro_feature_correlation"

# Standard budget ticks and labels for log x-axis
BUDGET_TICKS = [46, 226, 452, 1130, 4518]
BUDGET_LABELS = ["46\n(1%)", "226\n(5%)", "452\n(10%)", "1130\n(25%)", "4518\n(100%)"]

# Colorblind-safe Okabe-Ito color mapping
PALETTE = {
    "B1": "#000000",                  # raw BOLD
    "B2": "#999999",                  # window PCA
    "B3": "#56B4E9",                  # frozen random encoder (sky blue)
    "B4": "#E69F00",                  # frozen reconstruction encoder phase A (orange)
    "R": "#D55E00",                   # frozen v2 reconstruction encoder (vermillion)
    "L": "#009E73",                   # frozen latent-prediction encoder (bluish green)
    "finetune_random": "#CC79A7",     # fine-tuned random init (reddish purple)
    "finetune_recon": "#0072B2",      # fine-tuned recon init (blue)
    "finetune_latent": "#009E73",     # fine-tuned latent init (bluish green)
    "block": "#0072B2",               # phase C block allocation
    "story": "#D55E00",               # phase C story allocation
}

FIGURE_NOTE = "Means over seeds; whiskers show observed seed ranges, not confidence intervals."


def format_axis(ax, is_log_x=True, xlabel=None, ylabel=None, title=None):
    """Apply standard clean styling across all figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", alpha=0.3, color="#d0d0d0", zorder=0)
    ax.set_axisbelow(True)

    if is_log_x:
        ax.set_xscale("log")
        ax.set_xticks(BUDGET_TICKS)
        ax.set_xticklabels(BUDGET_LABELS, fontsize=9)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlim(35, 5800)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)

    ax.tick_params(labelsize=9)


def plot_whisker_series(ax, x_dict, color, label=None, marker="o", linestyle="-",
                        markersize=5, linewidth=1.2, dashes=None, zorder=3):
    """Plot mean across seeds with min-max whiskers."""
    if not x_dict:
        return
    xs = sorted(x_dict.keys())
    means, lower_errs, upper_errs = [], [], []
    for x in xs:
        vals = np.array(x_dict[x], dtype=float)
        m = float(np.mean(vals))
        means.append(m)
        lower_errs.append(m - float(np.min(vals)))
        upper_errs.append(float(np.max(vals)) - m)

    yerr = np.array([lower_errs, upper_errs])
    line = ax.errorbar(
        xs, means, yerr=yerr,
        fmt=marker, linestyle=linestyle, color=color,
        ecolor=color, elinewidth=1.0, capsize=3, capthick=1.0,
        markersize=markersize, linewidth=linewidth,
        label=label, zorder=zorder
    )
    if dashes and line[0] is not None:
        line[0].set_dashes(dashes)


def save_figure(fig, out_dir, stem, note=FIGURE_NOTE, bottom_margin=0.20):
    """Save figure as PNG (180 dpi) and PDF with bottom note."""
    if note:
        fig.text(
            0.5, 0.02, note,
            ha="center", fontsize=8, color="#555555", style="italic"
        )
    fig.subplots_adjust(bottom=bottom_margin)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------

def load_ridge_cells(results_dir):
    """Load ridge regression cells from phase_a and phase_b."""
    cells = []
    for phase in ["phase_a", "phase_b"]:
        pattern = str(results_dir / phase / "cells" / "*.json")
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path) as f:
                    data = json.load(f)
                if data.get("arm") == "B0":
                    continue
                cells.append({
                    "phase": phase,
                    "arm": data.get("arm"),
                    "features": data.get("features"),
                    "seed": data.get("seed"),
                    "cutoff_tr": data.get("cutoff_tr"),
                    "readout": data.get("readout"),
                    "fraction": data.get("fraction"),
                    "selected_examples": data.get("labels", {}).get("selected_examples"),
                    "val_r": data.get("val", {}).get(PRIMARY_METRIC),
                    "test_r": data.get("test", {}).get(PRIMARY_METRIC),
                    "path": path,
                })
            except Exception:
                continue
    return cells


def load_finetuning_runs(results_dir, group_name):
    """Load fine-tuning evaluation metrics and label subsets."""
    runs = []
    base_dir = results_dir / group_name
    if not base_dir.is_dir():
        return runs

    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "evaluation_v2" / "metrics.json"
        labels_file = run_dir / "label_subset.json"
        if not (metrics_file.is_file() and labels_file.is_file()):
            continue
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            with open(labels_file) as f:
                label_data = json.load(f)

            arm_label = run_dir.name.split("_c")[0]
            selected_examples = label_data.get("label_metadata", {}).get("selected_examples")

            runs.append({
                "group": group_name,
                "name": run_dir.name,
                "arm_label": arm_label,
                "seed": metrics.get("seed"),
                "fraction": float(metrics.get("fraction", 0.0)),
                "cutoff_tr": metrics.get("cutoff_tr"),
                "selected_examples": selected_examples,
                "val_r": metrics.get("val", {}).get(PRIMARY_METRIC),
                "test_r": metrics.get("test", {}).get(PRIMARY_METRIC),
            })
        except Exception:
            continue
    return runs


def load_phase_c_cells(results_dir):
    """Load coverage cells from phase_c/cells/*.json."""
    cells = []
    pattern = str(results_dir / "phase_c" / "cells" / "*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                data = json.load(f)
            cells.append({
                "allocation": data.get("allocation"),
                "block_tr": data.get("block_tr"),
                "subset_seed": data.get("subset_seed"),
                "fraction": data.get("fraction"),
                "selected_examples": data.get("labels", {}).get("selected_examples"),
                "labeled_stories": data.get("labels", {}).get("labeled_stories"),
                "val_selected_test_r": data.get("validation_selected", {}).get("test", {}).get(PRIMARY_METRIC),
                "cv_selected_test_r": (data.get("cv_selected", {}).get("test", {}).get(PRIMARY_METRIC)
                                       if "cv_selected" in data else None),
                "path": path,
            })
        except Exception:
            continue
    return cells


def load_ssl_logs(results_dir):
    """Load SSL training logs from ssl/{recon,latent}_UTS01_seed{17,23,47}/metrics.jsonl."""
    logs = {}
    ssl_dir = results_dir / "ssl"
    if not ssl_dir.is_dir():
        return logs

    for obj in ["recon", "latent"]:
        for seed in [17, 23, 47]:
            run_name = f"{obj}_UTS01_seed{seed}"
            jsonl_file = ssl_dir / run_name / "metrics.jsonl"
            if not jsonl_file.is_file():
                continue
            records = []
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
                if records:
                    logs[(obj, seed)] = records
            except Exception:
                continue
    return logs


# ---------------------------------------------------------------------------
# Plot Functions
# ---------------------------------------------------------------------------

def plot_phase_a_label_efficiency(ridge_cells, finetune_runs, out_dir):
    """Figure 1: phase_a_label_efficiency (validation and test r vs labeled targets)."""
    # Filter phase_a cells with cutoff_tr == 3
    a_cells = [c for c in ridge_cells if c["phase"] == "phase_a" and c["cutoff_tr"] == 3]
    if not a_cells:
        return None, "No Phase A cells with cutoff_tr == 3"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)

    # Organize series for validation and test r
    # B1 fir
    b1_fir = {metric: {c["selected_examples"]: [c[metric]]
                       for c in a_cells if c["arm"] == "B1" and c["readout"] == "fir"}
              for metric in ["val_r", "test_r"]}
    # B1 endpoint
    b1_endpoint = {metric: {c["selected_examples"]: [c[metric]]
                            for c in a_cells if c["arm"] == "B1" and c["readout"] == "endpoint"}
                   for metric in ["val_r", "test_r"]}
    # B2 window
    b2_window = {metric: {c["selected_examples"]: [c[metric]]
                          for c in a_cells if c["arm"] == "B2" and c["readout"] == "window"}
                 for metric in ["val_r", "test_r"]}
    # B3 random fir
    b3_fir = {metric: defaultdict(list) for metric in ["val_r", "test_r"]}
    for c in a_cells:
        if c["arm"] == "B3" and c["features"] == "random" and c["readout"] == "fir":
            for metric in ["val_r", "test_r"]:
                b3_fir[metric][c["selected_examples"]].append(c[metric])
    # B4 ssl_final fir
    b4_fir = {metric: defaultdict(list) for metric in ["val_r", "test_r"]}
    for c in a_cells:
        if c["arm"] == "B4" and c["features"] == "ssl_final" and c["readout"] == "fir":
            for metric in ["val_r", "test_r"]:
                b4_fir[metric][c["selected_examples"]].append(c[metric])

    # Finetune_a cutoff 3
    ft_random = {metric: defaultdict(list) for metric in ["val_r", "test_r"]}
    ft_recon = {metric: defaultdict(list) for metric in ["val_r", "test_r"]}
    for r in finetune_runs:
        if r["cutoff_tr"] == 3:
            if r["arm_label"] == "random":
                for metric in ["val_r", "test_r"]:
                    ft_random[metric][r["selected_examples"]].append(r[metric])
            elif r["arm_label"] == "ssl_best":
                for metric in ["val_r", "test_r"]:
                    ft_recon[metric][r["selected_examples"]].append(r[metric])

    panels = [("Validation", "val_r", axes[0]), ("Test", "test_r", axes[1])]

    for title, metric, ax in panels:
        # Zero line
        ax.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

        # B1 endpoint: thin dashed line
        if b1_endpoint[metric]:
            xs = sorted(b1_endpoint[metric].keys())
            ys = [b1_endpoint[metric][x][0] for x in xs]
            ax.plot(xs, ys, linestyle="--", linewidth=1.1, color=PALETTE["B1"],
                    label="B1 endpoint (raw BOLD)" if title == "Test" else None, zorder=2)

        # B2 window: line
        if b2_window[metric]:
            xs = sorted(b2_window[metric].keys())
            ys = [b2_window[metric][x][0] for x in xs]
            ax.plot(xs, ys, linestyle="-", linewidth=1.2, color=PALETTE["B2"],
                    label="B2 window PCA" if title == "Test" else None, zorder=2)

        # B1 fir: line with markers
        if b1_fir[metric]:
            xs = sorted(b1_fir[metric].keys())
            ys = [b1_fir[metric][x][0] for x in xs]
            ax.plot(xs, ys, linestyle="-", marker="o", markersize=4.5, linewidth=1.3,
                    color=PALETTE["B1"], label="B1 fir (raw BOLD)" if title == "Test" else None, zorder=3)

        # B3 random fir: mean over seeds with min-max whiskers
        plot_whisker_series(ax, b3_fir[metric], color=PALETTE["B3"],
                            label="B3 random fir" if title == "Test" else None,
                            marker="o", linestyle="-", markersize=4.5, zorder=4)

        # B4 ssl_final fir: mean over seeds with min-max whiskers
        plot_whisker_series(ax, b4_fir[metric], color=PALETTE["B4"],
                            label="B4 ssl_final fir" if title == "Test" else None,
                            marker="o", linestyle="-", markersize=4.5, zorder=4)

        # Fine-tuning: random (markers only, mean with min-max whiskers)
        plot_whisker_series(ax, ft_random[metric], color=PALETTE["finetune_random"],
                            label="Fine-tuned random" if title == "Test" else None,
                            marker="s", linestyle="none", markersize=5.5, zorder=5)

        # Fine-tuning: ssl_best (markers only, mean with min-max whiskers)
        plot_whisker_series(ax, ft_recon[metric], color=PALETTE["finetune_recon"],
                            label="Fine-tuned recon init (ssl_best)" if title == "Test" else None,
                            marker="D", linestyle="none", markersize=5.5, zorder=5)

        format_axis(
            ax,
            is_log_x=True,
            xlabel="Labeled training targets",
            ylabel="Session-macro feature r" if ax == axes[0] else None,
            title=title
        )

    # Frameless legend on test panel
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")

    written = save_figure(fig, out_dir, "phase_a_label_efficiency", bottom_margin=0.20)
    return written, None


def plot_phase_b_objectives(ridge_cells, finetune_b_runs, out_dir):
    """Figure 2: phase_b_objectives (R vs L, reference lines, paired differences)."""
    # Cutoff 3, readout fir
    b_cells = [c for c in ridge_cells if c["phase"] == "phase_b" and c["cutoff_tr"] == 3 and c["readout"] == "fir"]
    if not b_cells:
        return None, "No Phase B cells with cutoff_tr == 3 and readout == fir"

    a_cells = [c for c in ridge_cells if c["phase"] == "phase_a" and c["cutoff_tr"] == 3 and c["readout"] == "fir"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # --- Left panel: R and L mean over seeds with min-max whiskers + ref lines + finetune_b ---
    ax_left = axes[0]
    ax_left.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    # Reference lines from phase A (both gray dashed)
    # B1 fir
    b1_a = {c["selected_examples"]: c["test_r"] for c in a_cells if c["arm"] == "B1"}
    if b1_a:
        xs = sorted(b1_a.keys())
        ys = [b1_a[x] for x in xs]
        ax_left.plot(xs, ys, linestyle="--", linewidth=1.2, color="#666666",
                     label="B1 fir (Phase A ref)", zorder=2)

    # B3 random fir
    b3_a = defaultdict(list)
    for c in a_cells:
        if c["arm"] == "B3" and c["features"] == "random":
            b3_a[c["selected_examples"]].append(c["test_r"])
    if b3_a:
        xs = sorted(b3_a.keys())
        ys = [float(np.mean(b3_a[x])) for x in xs]
        ax_left.plot(xs, ys, linestyle="--", dashes=(4, 3), linewidth=1.2, color="#888888",
                     label="B3 random fir (Phase A ref)", zorder=2)

    # Series R and L
    r_data = defaultdict(list)
    l_data = defaultdict(list)
    for c in b_cells:
        if c["arm"] == "R":
            r_data[c["selected_examples"]].append(c["test_r"])
        elif c["arm"] == "L":
            l_data[c["selected_examples"]].append(c["test_r"])

    plot_whisker_series(ax_left, r_data, color=PALETTE["R"], label="R frozen recon (FIR)",
                        marker="o", linestyle="-", markersize=5, zorder=3)
    plot_whisker_series(ax_left, l_data, color=PALETTE["L"], label="L frozen latent (FIR)",
                        marker="o", linestyle="-", markersize=5, zorder=3)

    # finetune_b markers if present
    ft_recon_b = defaultdict(list)
    ft_latent_b = defaultdict(list)
    for r in finetune_b_runs:
        if r["cutoff_tr"] == 3:
            if r["arm_label"] == "recon":
                ft_recon_b[r["selected_examples"]].append(r["test_r"])
            elif r["arm_label"] == "latent":
                ft_latent_b[r["selected_examples"]].append(r["test_r"])

    if ft_recon_b:
        plot_whisker_series(ax_left, ft_recon_b, color=PALETTE["finetune_recon"],
                            label="Fine-tuned recon", marker="s", linestyle="none",
                            markersize=5.5, zorder=4)
    if ft_latent_b:
        plot_whisker_series(ax_left, ft_latent_b, color=PALETTE["finetune_latent"],
                            label="Fine-tuned latent", marker="^", linestyle="none",
                            markersize=5.5, zorder=4)

    format_axis(
        ax_left,
        is_log_x=True,
        xlabel="Labeled training targets",
        ylabel="Session-macro feature r",
        title="Reconstruction vs. Latent Readout"
    )
    ax_left.legend(frameon=False, fontsize=8.5, loc="upper left")

    # --- Right panel: per-seed paired difference L - R at each budget ---
    ax_right = axes[1]
    ax_right.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    # Group cell scores by (arm, seed, selected_examples)
    b_lookup = {}
    for c in b_cells:
        b_lookup[(c["arm"], c["seed"], c["selected_examples"])] = c["test_r"]

    # Compute paired differences where both L and R exist
    all_budgets = sorted({c["selected_examples"] for c in b_cells})
    budgets_plotted = []
    mean_diffs = []

    seed_markers = {17: "o", 23: "s", 47: "^"}
    seeds_labeled = set()

    for b in all_budgets:
        diffs_at_b = []
        seeds_at_b = sorted({c["seed"] for c in b_cells if c["selected_examples"] == b})
        for s in seeds_at_b:
            l_val = b_lookup.get(("L", s, b))
            r_val = b_lookup.get(("R", s, b))
            if l_val is not None and r_val is not None:
                diff = l_val - r_val
                diffs_at_b.append(diff)
                marker = seed_markers.get(s, "o")
                label = f"Seed {s} diff" if s not in seeds_labeled else None
                ax_right.scatter(b, diff, marker=marker, color=PALETTE["L"],
                                 alpha=0.75, s=35, zorder=3, label=label)
                seeds_labeled.add(s)

        if diffs_at_b:
            budgets_plotted.append(b)
            mean_diffs.append(float(np.mean(diffs_at_b)))

    if budgets_plotted:
        ax_right.plot(budgets_plotted, mean_diffs, color="#000000", linewidth=1.4,
                      marker="D", markersize=4.5, label="Mean difference (L − R)", zorder=4)

    format_axis(
        ax_right,
        is_log_x=True,
        xlabel="Labeled training targets",
        ylabel="Session-macro feature r (L − R)",
        title="Paired Objective Contrast (L − R)"
    )
    ax_right.legend(frameon=False, fontsize=8.5, loc="upper right", scatterpoints=1)

    written = save_figure(fig, out_dir, "phase_b_objectives", bottom_margin=0.20)
    return written, None


def plot_phase_c_coverage(phase_c_cells, out_dir):
    """Figure 3: phase_c_coverage (test r vs labeled targets and labeled stories)."""
    if not phase_c_cells:
        return None, "No Phase C cell files found in results/v2/phase_c/cells/"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Panel 1: Test r vs labeled targets for allocation block vs story
    ax_r = axes[0]
    ax_r.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    # Panel 2: Labeled stories vs labeled targets for each allocation
    ax_stories = axes[1]

    for alloc in ["block", "story"]:
        alloc_cells = [c for c in phase_c_cells if c["allocation"] == alloc]
        if not alloc_cells:
            continue
        color = PALETTE.get(alloc, "#333333")

        # Validation-selected test r
        val_sel_dict = defaultdict(list)
        # CV-selected test r (optional)
        cv_sel_dict = defaultdict(list)
        # Stories dict
        stories_dict = defaultdict(list)

        for c in alloc_cells:
            n_targets = c["selected_examples"]
            if c["val_selected_test_r"] is not None:
                val_sel_dict[n_targets].append(c["val_selected_test_r"])
            if c["cv_selected_test_r"] is not None:
                cv_sel_dict[n_targets].append(c["cv_selected_test_r"])
            if c["labeled_stories"] is not None:
                stories_dict[n_targets].append(c["labeled_stories"])

        # Plot validation-selected (solid)
        if val_sel_dict:
            plot_whisker_series(ax_r, val_sel_dict, color=color,
                                label=f"{alloc.capitalize()} (val-selected)",
                                marker="o", linestyle="-", markersize=4.5, zorder=3)

        # Plot CV-selected (dashed)
        if cv_sel_dict:
            plot_whisker_series(ax_r, cv_sel_dict, color=color,
                                label=f"{alloc.capitalize()} (CV-selected)",
                                marker="s", linestyle="--", markersize=4.5, zorder=3)

        # Plot labeled stories
        if stories_dict:
            plot_whisker_series(ax_stories, stories_dict, color=color,
                                label=f"{alloc.capitalize()} allocation",
                                marker="o", linestyle="-", markersize=4.5, zorder=3)

    format_axis(
        ax_r,
        is_log_x=True,
        xlabel="Labeled training targets",
        ylabel="Session-macro feature r",
        title="Coverage Strategy: Readout Accuracy"
    )
    ax_r.legend(frameon=False, fontsize=8.5, loc="upper left")

    format_axis(
        ax_stories,
        is_log_x=True,
        xlabel="Labeled training targets",
        ylabel="Labeled stories",
        title="Coverage Strategy: Story Diversity"
    )
    ax_stories.legend(frameon=False, fontsize=8.5, loc="upper left")

    written = save_figure(fig, out_dir, "phase_c_coverage", bottom_margin=0.20)
    return written, None


def plot_ssl_diagnostics(ssl_logs, out_dir):
    """Figure 4: ssl_diagnostics (four panels vs epoch)."""
    if not ssl_logs:
        return None, "No SSL logs found in results/v2/ssl/"

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    ax_recon_loss = axes[0, 0]
    ax_latent_loss = axes[0, 1]
    ax_rank = axes[1, 0]
    ax_content = axes[1, 1]

    # Panel 1: recon val_loss
    recon_plotted = False
    for seed in [17, 23, 47]:
        if ("recon", seed) in ssl_logs:
            records = ssl_logs[("recon", seed)]
            epochs = [r["epoch"] for r in records]
            losses = [r["val_loss"] for r in records]
            ax_recon_loss.plot(epochs, losses, color=PALETTE["R"], linewidth=1.2,
                               alpha=0.85, label=f"Seed {seed}")
            recon_plotted = True

    format_axis(
        ax_recon_loss,
        is_log_x=False,
        xlabel="Epoch",
        ylabel="Validation loss (MSE)",
        title="Reconstruction Validation Loss"
    )
    if recon_plotted:
        ax_recon_loss.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Panel 2: latent val_loss
    latent_plotted = False
    for seed in [17, 23, 47]:
        if ("latent", seed) in ssl_logs:
            records = ssl_logs[("latent", seed)]
            epochs = [r["epoch"] for r in records]
            losses = [r["val_loss"] for r in records]
            ax_latent_loss.plot(epochs, losses, color=PALETTE["L"], linewidth=1.2,
                                alpha=0.85, label=f"Seed {seed}")
            latent_plotted = True

    format_axis(
        ax_latent_loss,
        is_log_x=False,
        xlabel="Epoch",
        ylabel="Validation loss (MSE, normalized teacher states)",
        title="Latent Prediction Validation Loss"
    )
    if latent_plotted:
        ax_latent_loss.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Panel 3: student effective_rank_last for both objectives + latent teacher dotted
    r_stud_labeled = False
    l_stud_labeled = False
    l_teach_labeled = False

    for seed in [17, 23, 47]:
        if ("recon", seed) in ssl_logs:
            records = ssl_logs[("recon", seed)]
            epochs = [r["epoch"] for r in records]
            ranks = [r.get("diagnostics", {}).get("student", {}).get("effective_rank_last") for r in records]
            if any(v is not None for v in ranks):
                ax_rank.plot(epochs, ranks, color=PALETTE["R"], linestyle="-", linewidth=1.1,
                             alpha=0.85, label="Reconstruction student" if not r_stud_labeled else None)
                r_stud_labeled = True

        if ("latent", seed) in ssl_logs:
            records = ssl_logs[("latent", seed)]
            epochs = [r["epoch"] for r in records]
            stud_ranks = [r.get("diagnostics", {}).get("student", {}).get("effective_rank_last") for r in records]
            teach_ranks = [r.get("diagnostics", {}).get("teacher", {}).get("effective_rank_last") for r in records]

            if any(v is not None for v in stud_ranks):
                ax_rank.plot(epochs, stud_ranks, color=PALETTE["L"], linestyle="-", linewidth=1.1,
                             alpha=0.85, label="Latent student" if not l_stud_labeled else None)
                l_stud_labeled = True

            if any(v is not None for v in teach_ranks):
                ax_rank.plot(epochs, teach_ranks, color=PALETTE["L"], linestyle=":", linewidth=1.3,
                             alpha=0.85, label="Latent teacher" if not l_teach_labeled else None)
                l_teach_labeled = True

    format_axis(
        ax_rank,
        is_log_x=False,
        xlabel="Epoch",
        ylabel="Effective rank",
        title="Effective Rank"
    )
    ax_rank.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Panel 4: student content_fraction for both objectives
    r_frac_labeled = False
    l_frac_labeled = False

    for seed in [17, 23, 47]:
        if ("recon", seed) in ssl_logs:
            records = ssl_logs[("recon", seed)]
            epochs = [r["epoch"] for r in records]
            fracs = [r.get("diagnostics", {}).get("student", {}).get("content_fraction") for r in records]
            if any(v is not None for v in fracs):
                ax_content.plot(epochs, fracs, color=PALETTE["R"], linestyle="-", linewidth=1.1,
                                alpha=0.85, label="Reconstruction student" if not r_frac_labeled else None)
                r_frac_labeled = True

        if ("latent", seed) in ssl_logs:
            records = ssl_logs[("latent", seed)]
            epochs = [r["epoch"] for r in records]
            fracs = [r.get("diagnostics", {}).get("student", {}).get("content_fraction") for r in records]
            if any(v is not None for v in fracs):
                ax_content.plot(epochs, fracs, color=PALETTE["L"], linestyle="-", linewidth=1.1,
                                alpha=0.85, label="Latent student" if not l_frac_labeled else None)
                l_frac_labeled = True

    format_axis(
        ax_content,
        is_log_x=False,
        xlabel="Epoch",
        ylabel="Content fraction",
        title="Student Content Fraction"
    )
    ax_content.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.tight_layout(pad=2.5, rect=[0, 0.04, 1, 1])
    written = save_figure(fig, out_dir, "ssl_diagnostics",
                          note="Thin lines show individual pre-training seeds (17, 23, 47).",
                          bottom_margin=0.08)
    return written, None



def plot_replication_summary(results_dir, out_dir):
    """Figure 5: replication_summary (cross-subject performance and paired contrasts)."""
    subjects = ["UTS01", "UTS02", "UTS03"]
    subject_data = {}

    for s in subjects:
        if s == "UTS01":
            pa_dir = results_dir / "phase_a" / "cells"
            pb_dir = results_dir / "phase_b" / "cells"
            ft_dir = results_dir / "finetune_b"
        else:
            pa_dir = results_dir / "replication" / s / "phase_a" / "cells"
            pb_dir = results_dir / "replication" / s / "phase_b" / "cells"
            ft_dir = results_dir / "replication" / s / "finetune"

        if not (pa_dir.is_dir() and pb_dir.is_dir() and ft_dir.is_dir()):
            return None, f"Missing directories for subject {s}"

        b1_vals = []
        b3_vals = {}
        for p in glob.glob(str(pa_dir / "*.json")):
            try:
                with open(p) as f:
                    d = json.load(f)
                if (d.get("readout") == "fir" and d.get("cutoff_tr") == 3
                        and d.get("fraction") == 1):
                    r = d.get("test", {}).get(PRIMARY_METRIC)
                    if d.get("arm") == "B1":
                        b1_vals.append(r)
                    elif d.get("arm") == "B3":
                        b3_vals[d.get("seed")] = r
            except Exception:
                continue

        r_vals = {}
        l_vals = {}
        for p in glob.glob(str(pb_dir / "*.json")):
            try:
                with open(p) as f:
                    d = json.load(f)
                if (d.get("readout") == "fir" and d.get("cutoff_tr") == 3
                        and d.get("fraction") == 1):
                    r = d.get("test", {}).get(PRIMARY_METRIC)
                    if d.get("arm") == "R":
                        r_vals[d.get("seed")] = r
                    elif d.get("arm") == "L":
                        l_vals[d.get("seed")] = r
            except Exception:
                continue

        ft_recon_vals = {}
        for run_dir in sorted(glob.glob(str(ft_dir / "recon_c3_f1_seed*"))):
            metrics_file = Path(run_dir) / "evaluation_v2" / "metrics.json"
            if metrics_file.is_file():
                try:
                    with open(metrics_file) as f:
                        d = json.load(f)
                    r = d.get("test", {}).get(PRIMARY_METRIC)
                    ft_recon_vals[d.get("seed")] = r
                except Exception:
                    continue

        subject_data[s] = {
            "B1": b1_vals,
            "B3": b3_vals,
            "R": r_vals,
            "L": l_vals,
            "ft_recon": ft_recon_vals,
        }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax_left, ax_right = axes

    # Zero line on both panels
    ax_left.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)
    ax_right.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    # --- Left panel: test session-macro r at 100% labels per subject ---
    arm_defs = [
        ("B1", "B1 raw BOLD fir", PALETTE["B1"], "o"),
        ("B3", "B3 frozen random fir", PALETTE["B3"], "o"),
        ("R", "R frozen recon fir", PALETTE["R"], "o"),
        ("L", "L frozen latent fir", PALETTE["L"], "o"),
        ("ft_recon", "Fine-tuned reconstruction init", PALETTE["finetune_recon"], "s"),
    ]
    arm_offsets = np.linspace(-0.24, 0.24, len(arm_defs))

    for arm_idx, (arm_key, arm_label, color, marker) in enumerate(arm_defs):
        xs, means, lower_errs, upper_errs = [], [], [], []
        for s_idx, s in enumerate(subjects):
            raw_vals = subject_data[s][arm_key]
            vals = list(raw_vals.values()) if isinstance(raw_vals, dict) else list(raw_vals)
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            m = float(np.mean(arr))
            xs.append(s_idx + arm_offsets[arm_idx])
            means.append(m)
            lower_errs.append(m - float(np.min(arr)))
            upper_errs.append(float(np.max(arr)) - m)

        if xs:
            yerr = np.array([lower_errs, upper_errs])
            ax_left.errorbar(
                xs, means, yerr=yerr,
                fmt=marker, linestyle="none", color=color,
                ecolor=color, elinewidth=1.0, capsize=3, capthick=1.0,
                markersize=5, linewidth=1.2, label=arm_label, zorder=3
            )

    format_axis(
        ax_left,
        is_log_x=False,
        xlabel=None,
        ylabel="Session-macro feature r",
        title="Cross-Subject Performance (100% Budget)"
    )
    ax_left.set_xticks(range(len(subjects)))
    ax_left.set_xticklabels(subjects, fontsize=9)
    ax_left.set_xlim(-0.5, len(subjects) - 0.5)
    ax_left.legend(frameon=False, fontsize=8.5, loc="lower right")

    # --- Right panel: paired seed differences for three contrasts ---
    contrast_defs = [
        ("L \u2212 R", PALETTE["L"]),
        ("B1 \u2212 R", PALETTE["R"]),
        ("B1 \u2212 fine-tuned recon init", PALETTE["finetune_recon"]),
    ]
    c_offsets = [-0.22, 0.0, 0.22]
    tick_hw = 0.06

    for c_idx, (c_label, color) in enumerate(contrast_defs):
        for s_idx, s in enumerate(subjects):
            b1_val = subject_data[s]["B1"][0] if subject_data[s]["B1"] else None
            r_dict = subject_data[s]["R"]
            l_dict = subject_data[s]["L"]
            ft_dict = subject_data[s]["ft_recon"]

            seeds = sorted(set(r_dict.keys()) & set(l_dict.keys()) & set(ft_dict.keys()))
            diffs = []
            for seed in seeds:
                if c_idx == 0:
                    diffs.append(l_dict[seed] - r_dict[seed])
                elif c_idx == 1:
                    diffs.append(b1_val - r_dict[seed])
                elif c_idx == 2:
                    diffs.append(b1_val - ft_dict[seed])

            x_pos = s_idx + c_offsets[c_idx]
            ax_right.scatter(
                [x_pos] * len(diffs), diffs,
                color=color, alpha=0.85, s=35, zorder=3,
                label=c_label if s_idx == 0 else None
            )
            m = float(np.mean(diffs))
            ax_right.plot(
                [x_pos - tick_hw, x_pos + tick_hw], [m, m],
                color=color, linewidth=2.0, zorder=4
            )

    format_axis(
        ax_right,
        is_log_x=False,
        xlabel=None,
        ylabel="\u0394 session-macro r",
        title="Paired Seed Differences"
    )
    ax_right.set_xticks(range(len(subjects)))
    ax_right.set_xticklabels(subjects, fontsize=9)
    ax_right.set_xlim(-0.5, len(subjects) - 0.5)
    # Headroom so the legend never hides a seed (UTS02 has a point near +0.09).
    low, high = ax_right.get_ylim()
    ax_right.set_ylim(low, high + 0.45 * (high - low))
    ax_right.legend(frameon=False, fontsize=8.5, loc="upper right", scatterpoints=1)

    note = "Means over seeds; whiskers and dots show observed seeds, not confidence intervals. UTS02/UTS03 contrasts were registered before processing."
    written = save_figure(fig, out_dir, "replication_summary", note=note, bottom_margin=0.20)
    return written, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results" / "v2"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory: {results_dir}")
    print(f"Figures output directory: {figures_dir}")

    # Load available datasets
    ridge_cells = load_ridge_cells(results_dir)
    finetune_a_runs = load_finetuning_runs(results_dir, "finetune_a")
    finetune_b_runs = load_finetuning_runs(results_dir, "finetune_b")
    phase_c_cells = load_phase_c_cells(results_dir)
    ssl_logs = load_ssl_logs(results_dir)

    print(f"Loaded {len(ridge_cells)} ridge cells across phase_a and phase_b.")
    print(f"Loaded {len(finetune_a_runs)} fine-tuning runs from finetune_a.")
    print(f"Loaded {len(finetune_b_runs)} fine-tuning runs from finetune_b.")
    print(f"Loaded {len(phase_c_cells)} coverage cells from phase_c.")
    print(f"Loaded {len(ssl_logs)} SSL diagnostic log series.")

    all_written_files = []
    produced_figures = []
    skipped_figures = []

    # 1. Phase A Label Efficiency
    written, skip_reason = plot_phase_a_label_efficiency(ridge_cells, finetune_a_runs, figures_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("phase_a_label_efficiency")
    else:
        skipped_figures.append(("phase_a_label_efficiency", skip_reason))

    # 2. Phase B Objectives
    written, skip_reason = plot_phase_b_objectives(ridge_cells, finetune_b_runs, figures_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("phase_b_objectives")
    else:
        skipped_figures.append(("phase_b_objectives", skip_reason))

    # 3. Phase C Coverage
    written, skip_reason = plot_phase_c_coverage(phase_c_cells, figures_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("phase_c_coverage")
    else:
        skipped_figures.append(("phase_c_coverage", skip_reason))

    # 4. SSL Diagnostics
    written, skip_reason = plot_ssl_diagnostics(ssl_logs, figures_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("ssl_diagnostics")
    else:
        skipped_figures.append(("ssl_diagnostics", skip_reason))

    # 5. Replication Summary
    written, skip_reason = plot_replication_summary(results_dir, figures_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("replication_summary")
    else:
        skipped_figures.append(("replication_summary", skip_reason))

    print("\n--- Summary ---")
    print(f"Produced figures ({len(produced_figures)}):")
    for fig_name in produced_figures:
        print(f"  + {fig_name}")

    print(f"Skipped figures ({len(skipped_figures)}):")
    for fig_name, reason in skipped_figures:
        print(f"  - {fig_name}: {reason}")

    print("\nWritten files:")
    for path in all_written_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
