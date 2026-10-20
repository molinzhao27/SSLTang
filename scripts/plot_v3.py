#!/usr/bin/env python3
"""Plot scientific evaluation figures for the ssltang-v3 protocol.

Generates PNG (180 dpi) and PDF figures for:
1. v3_arms_by_budget: Test r vs labeled targets across V0-V4 per subject.
2. v3_v4_minus_v0: Test V4 - V0 by budget and development gate validation V4 - V0.
3. v3_controls: Horizontal dot/bar charts of alignment, nuisance, and negative controls.
4. v3_dense (optional): Test r under old stride-4 vs dense training clocks.
5. v3_replication: Replication comparison of exploratory subjects (UTS01-03)
   and registered replication subjects (UTS09-10).

Inputs are read strictly from measured experimental JSON files without interpolation or simulation.
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

# Development gate budgets
DEV_BUDGET_TICKS = [0, 1]
DEV_BUDGET_LABELS = ["452\n(10%)", "4518\n(100%)"]

SUBJECTS = ["UTS01", "UTS02", "UTS03"]
REPLICATION_SUBJECTS = ["UTS01", "UTS02", "UTS03", "UTS09", "UTS10"]

# Arm definitions, labels, and Okabe-Ito colors
ARM_ORDER = ["V0", "V1", "V2", "V3", "V4"]
ARM_LABELS = {
    "V0": "V0: all-voxel ridge (one \u03b1)",
    "V1": "V1: per-output \u03b1",
    "V2": "V2: coarse/detail weights",
    "V3": "V3: delay weights",
    "V4": "V4: joint coarse/detail + delay (locked)",
}
PALETTE = {
    "V0": "#000000",
    "V1": "#CC79A7",
    "V2": "#56B4E9",
    "V3": "#E69F00",
    "V4": "#009E73",
}
ARM_MARKERS = {
    "V0": "o",
    "V1": "s",
    "V2": "^",
    "V3": "v",
    "V4": "D",
}

SUBJECT_COLORS = {
    "UTS01": "#0072B2",
    "UTS02": "#D55E00",
    "UTS03": "#CC79A7",
}
SUBJECT_MARKERS = {
    "UTS01": "o",
    "UTS02": "s",
    "UTS03": "^",
}

FIGURE_NOTE = (
    "Points are measured fits (deterministic ridge per subject); "
    "UTS01\u201303 test stories are an exposed benchmark."
)

FIGURE_NOTE_REPLICATION = (
    "Each point is one deterministic fit. UTS01\u201303 test stories were examined in "
    "earlier studies; UTS09/UTS10 contrasts were registered before their data were downloaded."
)


def format_axis(ax, is_log_x=True, xlabel=None, ylabel=None, title=None, grid_axis="y"):
    """Apply standard clean styling across figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="-", alpha=0.3, color="#d0d0d0", zorder=0)
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

def load_evaluate_cells(results_dir):
    """Load evaluation cells for all subjects, arms, and fractions.
    
    Returns:
        subject_cells: dict[subject][arm][selected_examples] -> cell dict
        shift_nulls: dict[subject] -> float (null_q95 from V4_f1.json)
    """
    subject_cells = {s: defaultdict(dict) for s in SUBJECTS}
    shift_nulls = {}

    for s in SUBJECTS:
        eval_dir = results_dir / s / "evaluate" / "cells"
        if not eval_dir.is_dir():
            continue
        for path in sorted(glob.glob(str(eval_dir / "*.json"))):
            try:
                with open(path) as f:
                    data = json.load(f)
                if not data.get("estimable", True):
                    continue
                arm = data.get("arm")
                selected_examples = data.get("labels", {}).get("selected_examples")
                test_r = data.get("test", {}).get(PRIMARY_METRIC)
                val_r = data.get("val", {}).get(PRIMARY_METRIC)
                if arm and selected_examples is not None:
                    subject_cells[s][arm][selected_examples] = {
                        "fraction": data.get("fraction"),
                        "selected_examples": selected_examples,
                        "test_r": test_r,
                        "val_r": val_r,
                        "selection_label": data.get("selection", {}).get("label"),
                        "selection_alpha": data.get("selection", {}).get("alpha"),
                        "path": path,
                    }
                # Check for shift null q95 in V4_f1
                if arm == "V4" and data.get("fraction") == 1.0:
                    q95 = data.get("test_shift_null", {}).get("null_q95")
                    if q95 is not None:
                        shift_nulls[s] = q95
            except Exception:
                continue

    return subject_cells, shift_nulls


def load_develop_cells(results_dir):
    """Load development cells for all subjects and arms at f1 and f0.1."""
    subject_cells = {s: defaultdict(dict) for s in SUBJECTS}

    for s in SUBJECTS:
        dev_dir = results_dir / s / "develop" / "cells"
        if not dev_dir.is_dir():
            continue
        for path in sorted(glob.glob(str(dev_dir / "*.json"))):
            try:
                with open(path) as f:
                    data = json.load(f)
                if not data.get("estimable", True):
                    continue
                arm = data.get("arm")
                selected_examples = data.get("labels", {}).get("selected_examples")
                val_r = data.get("val", {}).get(PRIMARY_METRIC)
                if arm and selected_examples is not None:
                    subject_cells[s][arm][selected_examples] = {
                        "fraction": data.get("fraction"),
                        "selected_examples": selected_examples,
                        "val_r": val_r,
                        "selection_label": data.get("selection", {}).get("label"),
                        "selection_alpha": data.get("selection", {}).get("alpha"),
                        "path": path,
                    }
            except Exception:
                continue

    return subject_cells


def load_controls(results_dir):
    """Load controls.json for all subjects."""
    controls = {}
    for s in SUBJECTS:
        path = results_dir / s / "controls.json"
        if not path.is_file():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            controls[s] = {
                "real_test": data.get("real_test"),
                "disrupted_training": data.get("disrupted_training", {}).get("test"),
                "wrong_run_input": data.get("wrong_run_input", {}).get("test"),
                "nuisance_position": data.get("nuisance_position", {}).get("test"),
                "nuisance_speech_rate": data.get("nuisance_speech_rate", {}).get("test"),
                "nuisance_both": data.get("nuisance_both", {}).get("test"),
                "residualized_advanced": data.get("residualized_advanced", {}).get("test"),
                "residualized_V0": data.get("residualized_V0", {}).get("test"),
            }
        except Exception:
            continue
    return controls


def load_dense(results_dir):
    """Load optional dense clock evaluation cells if present."""
    dense_data = {}
    for s in SUBJECTS:
        dense_dir = results_dir / s / "dense"
        v0_path = dense_dir / "V0_dense.json"
        v4_path = dense_dir / "V4_dense.json"
        if v0_path.is_file() or v4_path.is_file():
            s_dense = {}
            for arm, path in [("V0", v0_path), ("V4", v4_path)]:
                if path.is_file():
                    try:
                        with open(path) as f:
                            d = json.load(f)
                        s_dense[arm] = {
                            "test_r": d.get("test", {}).get(PRIMARY_METRIC),
                            "n_train_targets": d.get("n_train_targets"),
                        }
                    except Exception:
                        continue
            if s_dense:
                dense_data[s] = s_dense
    return dense_data


# ---------------------------------------------------------------------------
# Plot Functions
# ---------------------------------------------------------------------------

def plot_v3_arms_by_budget(eval_cells, out_dir):
    """Figure 1: v3_arms_by_budget.
    
    Three panels, one per subject. Test r against labeled targets (log x) for V0-V4.
    V4 is plotted with a thicker line.
    """
    has_any = any(eval_cells[s] for s in SUBJECTS)
    if not has_any:
        return None, "No evaluate cells found across subjects"

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)

    for idx, s in enumerate(SUBJECTS):
        ax = axes[idx]
        ax.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

        s_cells = eval_cells.get(s, {})
        for arm in ARM_ORDER:
            arm_data = s_cells.get(arm, {})
            if not arm_data:
                continue
            xs = sorted(arm_data.keys())
            ys = [arm_data[x]["test_r"] for x in xs if arm_data[x]["test_r"] is not None]
            valid_xs = [x for x in xs if arm_data[x]["test_r"] is not None]
            if not ys:
                continue

            linewidth = 2.4 if arm == "V4" else 1.3
            markersize = 5.5 if arm == "V4" else 4.5
            zorder = 5 if arm == "V4" else 3

            ax.plot(
                valid_xs, ys,
                marker=ARM_MARKERS[arm],
                color=PALETTE[arm],
                linewidth=linewidth,
                markersize=markersize,
                label=ARM_LABELS[arm] if idx == 0 else None,
                zorder=zorder,
            )

        format_axis(
            ax,
            is_log_x=True,
            xlabel="Labeled training targets",
            ylabel="Session-macro feature r" if idx == 0 else None,
            title=s,
        )

    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")

    written = save_figure(fig, out_dir, "v3_arms_by_budget", bottom_margin=0.20)
    return written, None


def plot_v3_v4_minus_v0(eval_cells, dev_cells, out_dir):
    """Figure 2: v3_v4_minus_v0.
    
    Panel 1: V4 - V0 in test r by budget, with subject markers, mean line, and zero line.
    Panel 2: Development (validation) V4 - V0 at 100% and 10% per subject, with protocol
             gate thresholds drawn as dashed lines (+0.005 mean at 100%, 0 at 10%).
    """
    has_test = any(eval_cells[s] for s in SUBJECTS)
    has_dev = any(dev_cells[s] for s in SUBJECTS)
    if not (has_test or has_dev):
        return None, "No evaluate or develop cells found"

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    ax_test, ax_dev = axes

    # --- Panel 1: Test V4 - V0 by budget ---
    ax_test.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    all_budgets = BUDGET_TICKS
    budgets_plotted = []
    test_means = []
    labeled_subjects = set()

    for b in all_budgets:
        diffs_at_b = []
        for s in SUBJECTS:
            v4_entry = eval_cells.get(s, {}).get("V4", {}).get(b)
            v0_entry = eval_cells.get(s, {}).get("V0", {}).get(b)
            if v4_entry and v0_entry:
                v4_r = v4_entry.get("test_r")
                v0_r = v0_entry.get("test_r")
                if v4_r is not None and v0_r is not None:
                    diff = v4_r - v0_r
                    diffs_at_b.append(diff)
                    label = s if s not in labeled_subjects else None
                    ax_test.scatter(
                        b, diff,
                        color=SUBJECT_COLORS[s],
                        marker=SUBJECT_MARKERS[s],
                        s=40,
                        alpha=0.85,
                        label=label,
                        zorder=3,
                    )
                    labeled_subjects.add(s)

        if diffs_at_b:
            budgets_plotted.append(b)
            test_means.append(float(np.mean(diffs_at_b)))

    if budgets_plotted:
        ax_test.plot(
            budgets_plotted, test_means,
            color="#000000", linewidth=1.5,
            marker="D", markersize=4.8,
            label="Subject mean",
            zorder=4,
        )

    format_axis(
        ax_test,
        is_log_x=True,
        xlabel="Labeled training targets",
        ylabel="\u0394 session-macro feature r (V4 \u2212 V0)",
        title="Test Difference (V4 \u2212 V0)",
    )
    ax_test.legend(frameon=False, fontsize=8.5, loc="lower right")

    # --- Panel 2: Development (validation) V4 - V0 at 10% and 100% ---
    ax_dev.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    # 10% (452) and 100% (4518) targets
    dev_budgets = [452, 4518]
    dev_labels = ["452\n(10%)", "4518\n(100%)"]
    gate_thresholds = {0: 0.0, 1: 0.005}  # 0 at 10%, +0.005 at 100%
    s_offsets = {"UTS01": -0.12, "UTS02": 0.0, "UTS03": 0.12}

    gate_labeled = False
    mean_labeled = False

    for x_idx, b in enumerate(dev_budgets):
        diffs = []
        for s in SUBJECTS:
            v4_entry = dev_cells.get(s, {}).get("V4", {}).get(b)
            v0_entry = dev_cells.get(s, {}).get("V0", {}).get(b)
            if v4_entry and v0_entry:
                v4_val = v4_entry.get("val_r")
                v0_val = v0_entry.get("val_r")
                if v4_val is not None and v0_val is not None:
                    diff = v4_val - v0_val
                    diffs.append(diff)
                    ax_dev.scatter(
                        x_idx + s_offsets[s], diff,
                        color=SUBJECT_COLORS[s],
                        marker=SUBJECT_MARKERS[s],
                        s=45,
                        alpha=0.85,
                        zorder=3,
                    )

        # Plot subject mean bar
        if diffs:
            m = float(np.mean(diffs))
            ax_dev.plot(
                [x_idx - 0.20, x_idx + 0.20], [m, m],
                color="#000000", linewidth=2.0, zorder=4,
                label="Subject mean" if not mean_labeled else None,
            )
            mean_labeled = True

        # Draw protocol gate threshold as dashed line
        thresh = gate_thresholds[x_idx]
        ax_dev.plot(
            [x_idx - 0.32, x_idx + 0.32], [thresh, thresh],
            color="#D55E00", linestyle="--", linewidth=1.5, zorder=2,
            label="Protocol gate threshold" if not gate_labeled else None,
        )
        gate_labeled = True

    format_axis(
        ax_dev,
        is_log_x=False,
        xlabel="Labeled training targets",
        ylabel="\u0394 session-macro feature r (V4 \u2212 V0)",
        title="Development Gate (Validation V4 \u2212 V0)",
    )
    ax_dev.set_xticks([0, 1])
    ax_dev.set_xticklabels(dev_labels, fontsize=9)
    ax_dev.set_xlim(-0.5, 1.5)
    ax_dev.legend(frameon=False, fontsize=8.5, loc="upper left")

    written = save_figure(fig, out_dir, "v3_v4_minus_v0", bottom_margin=0.20)
    return written, None


def plot_v3_controls(controls_data, shift_nulls, out_dir):
    """Figure 3: v3_controls.
    
    Per subject horizontal dot/bar chart with vertical zero line of:
    - V4 real test r
    - the shift-null 95th percentile
    - disrupted training and wrong-run input
    - nuisance position, oracle speech rate and both
    - residualized V4 and residualized V0
    """
    if not controls_data:
        return None, "No controls.json data found across subjects"

    row_defs = [
        ("real_test", "V4 real test r", PALETTE["V4"]),
        ("shift_null_q95", "Shift-null (95th percentile)", "#888888"),
        ("disrupted_training", "Disrupted training", "#D55E00"),
        ("wrong_run_input", "Wrong-run input", "#D55E00"),
        ("nuisance_position", "Nuisance: position", PALETTE["V2"]),
        ("nuisance_speech_rate", "Nuisance: speech rate (oracle)", PALETTE["V3"]),
        ("nuisance_both", "Nuisance: both (oracle)", PALETTE["V1"]),
        ("residualized_advanced", "Residualized V4", PALETTE["V4"]),
        ("residualized_V0", "Residualized V0", PALETTE["V0"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True, sharex=True)
    y_pos = np.arange(len(row_defs))

    for idx, s in enumerate(SUBJECTS):
        ax = axes[idx]
        s_ctrl = controls_data.get(s, {})
        q95 = shift_nulls.get(s)

        data_map = {
            "real_test": s_ctrl.get("real_test"),
            "shift_null_q95": q95,
            "disrupted_training": s_ctrl.get("disrupted_training"),
            "wrong_run_input": s_ctrl.get("wrong_run_input"),
            "nuisance_position": s_ctrl.get("nuisance_position"),
            "nuisance_speech_rate": s_ctrl.get("nuisance_speech_rate"),
            "nuisance_both": s_ctrl.get("nuisance_both"),
            "residualized_advanced": s_ctrl.get("residualized_advanced"),
            "residualized_V0": s_ctrl.get("residualized_V0"),
        }

        vals = [data_map.get(key) for key, _, _ in row_defs]
        colors = [col for _, _, col in row_defs]

        format_axis(
            ax,
            is_log_x=False,
            xlabel="Test session-macro feature r",
            ylabel=None,
            title=s,
            grid_axis="x",
        )
        ax.axvline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

        # Plot horizontal bars and dots
        valid_indices = [i for i, v in enumerate(vals) if v is not None]
        valid_vals = [vals[i] for i in valid_indices]
        valid_y = y_pos[valid_indices]
        valid_colors = [colors[i] for i in valid_indices]

        ax.barh(valid_y, valid_vals, height=0.52, color=valid_colors, alpha=0.85, zorder=2)
        ax.scatter(valid_vals, valid_y, color=valid_colors, s=28, zorder=3)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([label for _, label, _ in row_defs], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.025, 0.185)

    written = save_figure(fig, out_dir, "v3_controls", bottom_margin=0.20)
    return written, None


def plot_v3_dense(eval_cells, dense_data, out_dir):
    """Figure 4: v3_dense (optional).
    
    Test r for V0 and V4 under old (stride-4, evaluate F=1) and dense training clocks.
    """
    if not dense_data:
        return None, "No dense training clock files found"

    has_any = any(s in dense_data for s in SUBJECTS)
    if not has_any:
        return None, "No dense files available for any subject"

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    x_indices = np.arange(len(SUBJECTS))
    bar_width = 0.18
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    series_plotted = set()

    for idx, s in enumerate(SUBJECTS):
        # Stride-4 (F=1)
        v0_old = eval_cells.get(s, {}).get("V0", {}).get(4518, {}).get("test_r")
        v4_old = eval_cells.get(s, {}).get("V4", {}).get(4518, {}).get("test_r")
        # Dense
        v0_dense = dense_data.get(s, {}).get("V0", {}).get("test_r")
        v4_dense = dense_data.get(s, {}).get("V4", {}).get("test_r")

        series = [
            ("V0 stride-4 (F=1)", v0_old, PALETTE["V0"], 0.45, offsets[0]),
            ("V0 dense", v0_dense, PALETTE["V0"], 0.90, offsets[1]),
            ("V4 stride-4 (F=1)", v4_old, PALETTE["V4"], 0.45, offsets[2]),
            ("V4 dense", v4_dense, PALETTE["V4"], 0.90, offsets[3]),
        ]

        for name, val, color, alpha, offset in series:
            if val is not None:
                lbl = name if name not in series_plotted else None
                ax.bar(
                    x_indices[idx] + offset, val,
                    width=bar_width, color=color, alpha=alpha,
                    label=lbl, zorder=2,
                )
                series_plotted.add(name)

    format_axis(
        ax,
        is_log_x=False,
        xlabel="Subject",
        ylabel="Test session-macro feature r",
        title="Training Clock Comparison (Stride-4 vs. Dense)",
    )
    ax.set_xticks(x_indices)
    ax.set_xticklabels(SUBJECTS, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    written = save_figure(fig, out_dir, "v3_dense", bottom_margin=0.20)
    return written, None


def plot_v3_replication(results_dir, out_dir):
    """Figure 5: v3_replication.

    Two-panel replication figure comparing exploratory subjects (UTS01-03)
    and registered replication subjects (UTS09-10).
    Panel 1: Test r per subject (V0/V4 stride-4 vs. dense, and speech-rate oracle).
    Panel 2: Three registered contrasts per subject (R1, R2, R3) with a zero line.
    """
    subjects = REPLICATION_SUBJECTS

    # Gather data strictly from measured experimental JSON files
    data = {}
    for s in subjects:
        s_data = {}
        # V0 stride-4: results/v3/<S>/evaluate/cells/V0_f1.json
        p_v0 = results_dir / s / "evaluate" / "cells" / "V0_f1.json"
        if p_v0.is_file():
            try:
                with open(p_v0) as f:
                    s_data["v0_stride4"] = json.load(f).get("test", {}).get(PRIMARY_METRIC)
            except Exception:
                pass

        # V4 stride-4: results/v3/<S>/evaluate/cells/V4_f1.json
        p_v4 = results_dir / s / "evaluate" / "cells" / "V4_f1.json"
        if p_v4.is_file():
            try:
                with open(p_v4) as f:
                    s_data["v4_stride4"] = json.load(f).get("test", {}).get(PRIMARY_METRIC)
            except Exception:
                pass

        # V0 dense: results/v3/<S>/dense/V0_dense.json
        p_v0d = results_dir / s / "dense" / "V0_dense.json"
        if p_v0d.is_file():
            try:
                with open(p_v0d) as f:
                    s_data["v0_dense"] = json.load(f).get("test", {}).get(PRIMARY_METRIC)
            except Exception:
                pass

        # V4 dense: results/v3/<S>/dense/V4_dense.json
        p_v4d = results_dir / s / "dense" / "V4_dense.json"
        if p_v4d.is_file():
            try:
                with open(p_v4d) as f:
                    s_data["v4_dense"] = json.load(f).get("test", {}).get(PRIMARY_METRIC)
            except Exception:
                pass

        # Oracle speech-rate model: results/v3/<S>/controls.json
        p_ctrl = results_dir / s / "controls.json"
        if p_ctrl.is_file():
            try:
                with open(p_ctrl) as f:
                    s_data["oracle"] = json.load(f).get("nuisance_speech_rate", {}).get("test")
            except Exception:
                pass

        data[s] = s_data

    # Check if any data exists across subjects
    has_any = any(any(v is not None for v in s_data.values()) for s_data in data.values())
    if not has_any:
        return None, "No replication evaluation files found across subjects"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.8, 5.0))

    format_axis(
        ax_left,
        is_log_x=False,
        xlabel=None,
        ylabel="Test session-macro feature r",
        title=None,
    )
    ax_left.set_title("Test r per Subject", fontsize=10.5, fontweight="bold", pad=22)

    format_axis(
        ax_right,
        is_log_x=False,
        xlabel=None,
        ylabel="\u0394 test r",
        title=None,
    )
    ax_right.set_title("Registered Contrasts", fontsize=10.5, fontweight="bold", pad=22)

    x_indices = np.arange(len(subjects))

    # Visual separation and group labels
    for ax in (ax_left, ax_right):
        ax.set_xticks(x_indices)
        ax.set_xticklabels(subjects, fontsize=9)
        ax.set_xlim(-0.6, len(subjects) - 0.4)
        ax.axvline(2.5, color="#c0c0c0", linestyle="--", linewidth=1.0, zorder=1)
        ax.text(
            1.0, 1.02, "exploratory (exposed)",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color="#555555", style="italic",
        )
        ax.text(
            3.5, 1.02, "registered replication (fresh)",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color="#555555", style="italic",
        )

    # Right panel zero line
    ax_right.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

    # Marker x-offsets
    offsets_left = [-0.18, -0.06, 0.06, 0.18]
    offsets_right = [-0.14, 0.0, 0.14]

    left_vals = []
    right_vals = []
    labeled_series = set()

    for idx, s in enumerate(subjects):
        s_data = data.get(s, {})
        v0_s4 = s_data.get("v0_stride4")
        v4_s4 = s_data.get("v4_stride4")
        v0_d = s_data.get("v0_dense")
        v4_d = s_data.get("v4_dense")
        oracle = s_data.get("oracle")

        # Oracle speech rate tick
        if oracle is not None:
            left_vals.append(oracle)
            lbl = "oracle speech-rate model (no brain)" if "oracle" not in labeled_series else None
            ax_left.plot(
                [idx - 0.25, idx + 0.25], [oracle, oracle],
                color="#888888", linewidth=2.0, linestyle="-",
                label=lbl, zorder=2,
            )
            labeled_series.add("oracle")

        # V0 stride-4: #000000 open circle
        if v0_s4 is not None:
            left_vals.append(v0_s4)
            lbl = "V0 stride-4" if "v0_s4" not in labeled_series else None
            ax_left.plot(
                idx + offsets_left[0], v0_s4,
                marker="o", markerfacecolor="none", markeredgecolor=PALETTE["V0"], markeredgewidth=1.4,
                markersize=6, linestyle="",
                label=lbl, zorder=4,
            )
            labeled_series.add("v0_s4")

        # V4 stride-4: #009E73 open diamond
        if v4_s4 is not None:
            left_vals.append(v4_s4)
            lbl = "V4 stride-4" if "v4_s4" not in labeled_series else None
            ax_left.plot(
                idx + offsets_left[1], v4_s4,
                marker="D", markerfacecolor="none", markeredgecolor=PALETTE["V4"], markeredgewidth=1.4,
                markersize=5.5, linestyle="",
                label=lbl, zorder=4,
            )
            labeled_series.add("v4_s4")

        # V0 dense: #000000 filled circle
        if v0_d is not None:
            left_vals.append(v0_d)
            lbl = "V0 dense" if "v0_d" not in labeled_series else None
            ax_left.plot(
                idx + offsets_left[2], v0_d,
                marker="o", markerfacecolor=PALETTE["V0"], markeredgecolor=PALETTE["V0"],
                markersize=6, linestyle="",
                label=lbl, zorder=4,
            )
            labeled_series.add("v0_d")

        # V4 dense: #009E73 filled diamond
        if v4_d is not None:
            left_vals.append(v4_d)
            lbl = "V4 dense" if "v4_d" not in labeled_series else None
            ax_left.plot(
                idx + offsets_left[3], v4_d,
                marker="D", markerfacecolor=PALETTE["V4"], markeredgecolor=PALETTE["V4"],
                markersize=5.5, linestyle="",
                label=lbl, zorder=4,
            )
            labeled_series.add("v4_d")

        # Contrasts
        # R1 = V0 dense − V0 stride-4
        if v0_d is not None and v0_s4 is not None:
            r1 = v0_d - v0_s4
            right_vals.append(r1)
            lbl = "R1 = V0 dense \u2212 V0 stride-4" if "r1" not in labeled_series else None
            ax_right.plot(
                idx + offsets_right[0], r1,
                marker="s", color="#0072B2",
                markersize=6, linestyle="",
                label=lbl, zorder=4,
            )
            labeled_series.add("r1")

        # R2 = V4 − V0 stride-4
        if v4_s4 is not None and v0_s4 is not None:
            r2 = v4_s4 - v0_s4
            right_vals.append(r2)
            lbl = "R2 = V4 \u2212 V0 stride-4" if "r2" not in labeled_series else None
            ax_right.plot(
                idx + offsets_right[1], r2,
                marker="^", color="#E69F00",
                markersize=6.5, linestyle="",
                label=lbl, zorder=4,
            )
            labeled_series.add("r2")

        # R3 = V4 dense − V0 dense
        if v4_d is not None and v0_d is not None:
            r3 = v4_d - v0_d
            right_vals.append(r3)
            lbl = "R3 = V4 dense \u2212 V0 dense" if "r3" not in labeled_series else None
            ax_right.plot(
                idx + offsets_right[2], r3,
                marker="D", color=PALETTE["V4"],
                markersize=5.5, linestyle="",
                label=lbl, zorder=4,
            )
            labeled_series.add("r3")

    if left_vals:
        min_l = min(left_vals)
        max_l = max(left_vals)
        ax_left.set_ylim(bottom=max(0.0, min_l - 0.02), top=max_l + 0.075)  # Legend headroom above the data.

    if right_vals:
        min_r = min(right_vals)
        max_r = max(right_vals)
        ax_right.set_ylim(bottom=min(-0.005, min_r - 0.005), top=max_r + 0.015)

    ax_left.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax_right.legend(frameon=False, fontsize=8.5, loc="upper right")

    written = save_figure(fig, out_dir, "v3_replication", note=FIGURE_NOTE_REPLICATION, bottom_margin=0.18)
    return written, None


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results" / "v3"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory: {results_dir}")
    print(f"Figures output directory: {figures_dir}")

    # Load data
    eval_cells, shift_nulls = load_evaluate_cells(results_dir)
    dev_cells = load_develop_cells(results_dir)
    controls_data = load_controls(results_dir)
    dense_data = load_dense(results_dir)

    n_eval = sum(len(eval_cells[s][arm]) for s in SUBJECTS for arm in eval_cells[s])
    n_dev = sum(len(dev_cells[s][arm]) for s in SUBJECTS for arm in dev_cells[s])
    print(f"Loaded {n_eval} evaluate cells across subjects.")
    print(f"Loaded {n_dev} develop cells across subjects.")
    print(f"Loaded controls for {len(controls_data)} subjects.")
    print(f"Loaded dense clock files for {len(dense_data)} subjects.")

    all_written_files = []
    produced_figures = []
    skipped_figures = []

    # 1. Arms by budget
    written, skip_reason = plot_v3_arms_by_budget(eval_cells, figures_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("v3_arms_by_budget")
    else:
        skipped_figures.append(("v3_arms_by_budget", skip_reason))

    # 2. V4 - V0
    written, skip_reason = plot_v3_v4_minus_v0(eval_cells, dev_cells, figures_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("v3_v4_minus_v0")
    else:
        skipped_figures.append(("v3_v4_minus_v0", skip_reason))

    # 3. Controls
    written, skip_reason = plot_v3_controls(controls_data, shift_nulls, figures_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("v3_controls")
    else:
        skipped_figures.append(("v3_controls", skip_reason))

    # 4. Dense (optional)
    written, skip_reason = plot_v3_dense(eval_cells, dense_data, figures_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("v3_dense")
    else:
        skipped_figures.append(("v3_dense", skip_reason))

    # 5. Replication (registered)
    written, skip_reason = plot_v3_replication(results_dir, figures_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("v3_replication")
    else:
        skipped_figures.append(("v3_replication", skip_reason))

    print("\n--- Summary ---")
    print(f"Produced figures ({len(produced_figures)}):")
    for fig_name in produced_figures:
        print(f"  + {fig_name}")

    if skipped_figures:
        print(f"\nSkipped figures ({len(skipped_figures)}):")
        for fig_name, reason in skipped_figures:
            print(f"  - {fig_name}: {reason}")

    print("\nWritten files:")
    for path in all_written_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
