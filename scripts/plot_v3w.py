#!/usr/bin/env python3
"""Plot scientific evaluation figures for the ssltang-v3w protocol (§12).

Generates PNG (180 dpi) and PDF figures for:
1. v3w_subjects: Test session-macro feature correlation r at 100% labels across
   all 10 subjects, visually grouped into exposed (UTS01–03, UTS09–10) and fresh
   confirmatory (UTS04–08) cohorts. Shows stride-4 W0, stride-4 W1 (connected per
   subject), dense training clock W0/W1 (where available), and v3 V4 reference.
2. v3w_budgets: Labeled budget scaling (1% to 100%) of the advancement arm test
   gain (W1 − W0) across the five exposed subjects, with non-estimable cells annotated.
3. v3w_residualized: Paired comparison of raw stride-4 100% test difference (W1 − W0)
   versus speech-rate residualized difference for all 10 subjects.

Inputs are read strictly from measured experimental JSON files without interpolation
or simulation. Missing files/series are handled gracefully by skipping unavailable data.
"""

import argparse
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

# Primary evaluation metric
PRIMARY_METRIC = "session_macro_feature_correlation"

# Subject groups
EXPOSED_SUBJECTS = ["UTS01", "UTS02", "UTS03", "UTS09", "UTS10"]
FRESH_SUBJECTS = ["UTS04", "UTS05", "UTS06", "UTS07", "UTS08"]
ALL_SUBJECTS = EXPOSED_SUBJECTS + FRESH_SUBJECTS

# Labeled fraction budgets
BUDGET_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.0]
BUDGET_TICKS = [0.01, 0.05, 0.10, 0.25, 1.0]
BUDGET_LABELS = ["1%", "5%", "10%", "25%", "100%"]

# Okabe-Ito color palette for exposed subjects
EXPOSED_COLORS = {
    "UTS01": "#0072B2",  # blue
    "UTS02": "#D55E00",  # vermillion
    "UTS03": "#CC79A7",  # reddish purple
    "UTS09": "#009E73",  # bluish green
    "UTS10": "#E69F00",  # orange
}
EXPOSED_MARKERS = {
    "UTS01": "o",
    "UTS02": "s",
    "UTS03": "^",
    "UTS09": "v",
    "UTS10": "D",
}

# Footnotes
NOTE_SUBJECTS = (
    "Points are measured fits (deterministic ridge per subject). Exposed subjects (UTS01–03, UTS09–10) evaluated across multiple budgets;\n"
    "fresh confirmatory subjects (UTS04–08) evaluated at 100% labels. W0: uniform all-voxel ridge; W1: encoding-predictability-weighted ridge."
)
NOTE_BUDGETS = (
    "Curves show measured test difference (W1 − W0) by labeled fraction for the five exposed subjects.\n"
    "Non-estimable cells (n.e.) at 1% budget omitted due to insufficient labeled targets in at least one CV fold."
)
NOTE_RESIDUALIZED = (
    "Paired test differences (W1 − W0) at 100% labels before and after residualizing out position and oracle speech-rate nuisance models."
)


def format_axis(ax, is_log_x=False, xlabel=None, ylabel=None, title=None, grid_axis="y"):
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
        ax.set_xlim(0.007, 1.45)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)

    ax.tick_params(labelsize=9)


def save_figure(fig, out_dir, stem, note=None, bottom_margin=0.18):
    """Save figure as PNG (180 dpi) and PDF with bottom footnote."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if note:
        fig.text(
            0.5, 0.015, note,
            ha="center", fontsize=7.5, color="#555555", style="italic", linespacing=1.2
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

def load_v3w_evaluate_cells(results_dir: Path, subjects: list[str]):
    """Load evaluation cells for subjects and arms W0, W1, W2 across budgets.

    Returns:
        dict[subject][arm][fraction] -> dict with keys:
            "fraction": float,
            "estimable": bool,
            "test_r": float or None,
            "val_r": float or None,
            "selection_label": str or None,
            "path": Path,
    """
    cells = {s: defaultdict(dict) for s in subjects}
    for s in subjects:
        cells_dir = results_dir / s / "evaluate" / "cells"
        if not cells_dir.is_dir():
            continue
        for path in sorted(cells_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                arm = data.get("arm")
                if arm not in ("W0", "W1", "W2"):
                    continue
                fraction = data.get("fraction")
                if fraction is None:
                    continue
                fraction_key = round(float(fraction), 4)
                estimable = data.get("estimable", True)
                test_dict = data.get("test") or {}
                val_dict = data.get("val") or {}
                test_r = test_dict.get(PRIMARY_METRIC) if estimable else None
                val_r = val_dict.get(PRIMARY_METRIC) if estimable else None

                cells[s][arm][fraction_key] = {
                    "fraction": fraction_key,
                    "estimable": estimable,
                    "test_r": test_r,
                    "val_r": val_r,
                    "selection_label": data.get("selection", {}).get("label") if data.get("selection") else None,
                    "path": path,
                }
            except Exception:
                continue
    return cells


def load_v3w_residualized(results_dir: Path, subjects: list[str]):
    """Load residualized endpoint evaluations from results/v3w/<S>/evaluate/residualized.json.

    Returns:
        dict[subject] -> dict {"W0": float or None, "W1": float or None}
    """
    resid_data = {}
    for s in subjects:
        path = results_dir / s / "evaluate" / "residualized.json"
        if not path.is_file():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            test_dict = data.get("test", {})
            resid_data[s] = {
                "W0": test_dict.get("W0"),
                "W1": test_dict.get("W1"),
            }
        except Exception:
            continue
    return resid_data


def load_v3w_dense(results_dir: Path, subjects: list[str]):
    """Load dense clock evaluations from results/v3w/<S>/dense/<ARM>_dense.json.

    Returns:
        dict[subject][arm] -> dict {"test_r": float or None, "val_r": float or None}
    """
    dense_data = {s: {} for s in subjects}
    for s in subjects:
        dense_dir = results_dir / s / "dense"
        if not dense_dir.is_dir():
            continue
        for arm in ("W0", "W1"):
            path = dense_dir / f"{arm}_dense.json"
            if not path.is_file():
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                test_dict = data.get("test", {})
                val_dict = data.get("val", {})
                dense_data[s][arm] = {
                    "test_r": test_dict.get(PRIMARY_METRIC),
                    "val_r": val_dict.get(PRIMARY_METRIC),
                }
            except Exception:
                continue
    return dense_data


def load_v3_reference(v3_root: Path, subjects: list[str]):
    """Load v3 V4 reference evaluations from results/v3/<S>/evaluate/cells/V4_f1.json.

    Returns:
        dict[subject] -> dict {"v4_stride4": float or None, "v4_dense": float or None}
    """
    v3_data = {}
    for s in subjects:
        s_data = {}
        # V4 stride-4
        p_v4 = v3_root / s / "evaluate" / "cells" / "V4_f1.json"
        if p_v4.is_file():
            try:
                with open(p_v4) as f:
                    s_data["v4_stride4"] = json.load(f).get("test", {}).get(PRIMARY_METRIC)
            except Exception:
                pass

        # V4 dense
        p_v4d = v3_root / s / "dense" / "V4_dense.json"
        if p_v4d.is_file():
            try:
                with open(p_v4d) as f:
                    s_data["v4_dense"] = json.load(f).get("test", {}).get(PRIMARY_METRIC)
            except Exception:
                pass

        if s_data:
            v3_data[s] = s_data
    return v3_data


# ---------------------------------------------------------------------------
# Figure 1: v3w_subjects
# ---------------------------------------------------------------------------

def plot_v3w_subjects(eval_cells, dense_data, v3_ref, out_dir):
    """Figure 1: v3w_subjects.

    Test r at 100% labels for all 10 subjects. x-axis: subjects, grouped and
    visually separated into "Exposed (UTS01–03, UTS09–10)" and "Fresh confirmatory (UTS04–08)".
    For each subject shows markers for:
      - stride-4 W0
      - stride-4 W1
      - dense W0 (when present)
      - dense W1 (when present)
      - small reference marker for v3 V4 stride-4
    Connects each subject's stride-4 W0→W1 pair with a thin line.
    """
    subjects = ALL_SUBJECTS
    has_any = any(
        eval_cells.get(s, {}).get("W0", {}).get(1.0, {}).get("test_r") is not None
        or eval_cells.get(s, {}).get("W1", {}).get(1.0, {}).get("test_r") is not None
        for s in subjects
    )
    if not has_any:
        return None, "No 100% evaluation cells found across subjects"

    fig, ax = plt.subplots(figsize=(11.0, 5.0))

    format_axis(
        ax,
        is_log_x=False,
        xlabel="Subject",
        ylabel="Test session-macro r",
        title=None,
    )
    ax.set_title(
        "Test Session-Macro Feature Correlation across Subjects (100% Labels)",
        fontsize=11, fontweight="bold", pad=22
    )

    x_indices = np.arange(len(subjects))
    ax.set_xticks(x_indices)
    ax.set_xticklabels(subjects, fontsize=9.5)
    ax.set_xlim(-0.6, len(subjects) - 0.4)

    # Vertical group separator between exposed (0..4) and fresh confirmatory (5..9)
    ax.axvline(4.5, color="#c0c0c0", linestyle="--", linewidth=1.0, zorder=1)

    # Group header labels above the plotting area
    ax.text(
        2.0, 1.02, "Exposed (UTS01–03, UTS09–10)",
        transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=9, color="#444444", fontweight="bold"
    )
    ax.text(
        7.0, 1.02, "Fresh confirmatory (UTS04–08)",
        transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=9, color="#444444", fontweight="bold"
    )

    # X-offsets for the markers within each subject's slot
    offset_v4_ref = -0.22
    offset_w0_s4 = -0.10
    offset_w1_s4 = 0.04
    offset_w0_dense = 0.16
    offset_w1_dense = 0.28

    labeled_series = set()
    all_y_vals = []

    for idx, s in enumerate(subjects):
        # 1. Stride-4 evaluations
        w0_entry = eval_cells.get(s, {}).get("W0", {}).get(1.0, {})
        w1_entry = eval_cells.get(s, {}).get("W1", {}).get(1.0, {})
        w0_s4 = w0_entry.get("test_r")
        w1_s4 = w1_entry.get("test_r")

        # Connect stride-4 W0 -> W1 with a thin line
        if w0_s4 is not None and w1_s4 is not None:
            ax.plot(
                [idx + offset_w0_s4, idx + offset_w1_s4], [w0_s4, w1_s4],
                color="#888888", linewidth=1.1, zorder=2
            )

        # Plot W0 stride-4
        if w0_s4 is not None:
            all_y_vals.append(w0_s4)
            lbl = "W0 stride-4 (uniform ridge)" if "w0_s4" not in labeled_series else None
            ax.plot(
                idx + offset_w0_s4, w0_s4,
                marker="o", markerfacecolor="none", markeredgecolor="#000000", markeredgewidth=1.5,
                markersize=6, linestyle="", label=lbl, zorder=4
            )
            labeled_series.add("w0_s4")

        # Plot W1 stride-4
        if w1_s4 is not None:
            all_y_vals.append(w1_s4)
            lbl = "W1 stride-4 (predictability-weighted)" if "w1_s4" not in labeled_series else None
            ax.plot(
                idx + offset_w1_s4, w1_s4,
                marker="D", markerfacecolor="none", markeredgecolor="#009E73", markeredgewidth=1.5,
                markersize=5.8, linestyle="", label=lbl, zorder=4
            )
            labeled_series.add("w1_s4")

        # 2. Dense evaluations
        s_dense = dense_data.get(s, {})
        w0_d = s_dense.get("W0", {}).get("test_r")
        w1_d = s_dense.get("W1", {}).get("test_r")

        # Connect dense W0 -> W1 with a thin dashed line when present
        if w0_d is not None and w1_d is not None:
            ax.plot(
                [idx + offset_w0_dense, idx + offset_w1_dense], [w0_d, w1_d],
                color="#aaaaaa", linestyle="--", linewidth=1.0, zorder=2
            )

        # Plot W0 dense
        if w0_d is not None:
            all_y_vals.append(w0_d)
            lbl = "W0 dense" if "w0_d" not in labeled_series else None
            ax.plot(
                idx + offset_w0_dense, w0_d,
                marker="o", markerfacecolor="#000000", markeredgecolor="#000000",
                markersize=5.8, linestyle="", label=lbl, zorder=4
            )
            labeled_series.add("w0_d")

        # Plot W1 dense
        if w1_d is not None:
            all_y_vals.append(w1_d)
            lbl = "W1 dense" if "w1_d" not in labeled_series else None
            ax.plot(
                idx + offset_w1_dense, w1_d,
                marker="D", markerfacecolor="#009E73", markeredgecolor="#009E73",
                markersize=5.8, linestyle="", label=lbl, zorder=4
            )
            labeled_series.add("w1_d")

        # 3. v3 V4 reference marker
        v4_ref_val = v3_ref.get(s, {}).get("v4_stride4")
        if v4_ref_val is not None:
            all_y_vals.append(v4_ref_val)
            lbl = "v3 V4 stride-4 (reference)" if "v4_ref" not in labeled_series else None
            ax.plot(
                idx + offset_v4_ref, v4_ref_val,
                marker="^", markerfacecolor="#777777", markeredgecolor="#777777",
                markersize=4.8, linestyle="", label=lbl, zorder=3
            )
            labeled_series.add("v4_ref")

    if all_y_vals:
        min_y = min(all_y_vals)
        max_y = max(all_y_vals)
        ax.set_ylim(bottom=max(0.0, min_y - 0.02), top=max_y + 0.025)

    # Place legend outside plotting area
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    written = save_figure(fig, out_dir, "v3w_subjects", note=NOTE_SUBJECTS, bottom_margin=0.18)
    return written, None


# ---------------------------------------------------------------------------
# Figure 2: v3w_budgets
# ---------------------------------------------------------------------------

def plot_v3w_budgets(eval_cells, out_dir):
    """Figure 2: v3w_budgets.

    W1 − W0 test difference vs labeled fraction (x log scale, ticks 1%, 5%, 10%, 25%, 100%)
    for the five exposed subjects (one line each); skips non-estimable cells (W1 estimable
    false or no test) and annotates them as "n.e." near the x position; horizontal dashed zero line.
    """
    subjects = EXPOSED_SUBJECTS
    has_any = any(eval_cells.get(s, {}).get("W1") for s in subjects)
    if not has_any:
        return None, "No exposed subject evaluate cells found"

    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    # Horizontal dashed zero line
    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)

    all_diffs = []
    non_estimable_annotations = []

    for s in subjects:
        color = EXPOSED_COLORS[s]
        marker = EXPOSED_MARKERS[s]
        s_w1 = eval_cells.get(s, {}).get("W1", {})
        s_w0 = eval_cells.get(s, {}).get("W0", {})

        valid_fractions = []
        valid_diffs = []

        for b in BUDGET_FRACTIONS:
            w1_cell = s_w1.get(b)
            w0_cell = s_w0.get(b)

            w1_estimable = w1_cell and w1_cell.get("estimable", True) and w1_cell.get("test_r") is not None
            w0_estimable = w0_cell and w0_cell.get("estimable", True) and w0_cell.get("test_r") is not None

            if w1_estimable and w0_estimable:
                diff = w1_cell["test_r"] - w0_cell["test_r"]
                valid_fractions.append(b)
                valid_diffs.append(diff)
                all_diffs.append(diff)
            else:
                # Mark as non-estimable
                non_estimable_annotations.append((s, b))

        if valid_fractions:
            ax.plot(
                valid_fractions, valid_diffs,
                marker=marker, color=color, linewidth=1.5, markersize=5.8,
                label=s, zorder=3
            )

    # Annotate non-estimable cells near their x position
    # Group by budget to prevent text overlapping
    by_budget_ne = defaultdict(list)
    for s, b in non_estimable_annotations:
        by_budget_ne[b].append(s)

    for b, s_list in by_budget_ne.items():
        # Stack vertically below zero line
        for i, s in enumerate(s_list):
            y_annot = -0.010 - (i * 0.0065)
            ax.text(
                b, y_annot, f"n.e. ({s})",
                color=EXPOSED_COLORS[s], fontsize=8, ha="center", va="center",
                style="italic", zorder=4
            )

    format_axis(
        ax,
        is_log_x=True,
        xlabel="Labeled training fraction",
        ylabel="Δ test session-macro r (W1 − W0)",
        title="W1 − W0 Test Gain by Labeled Budget (Exposed Subjects)",
    )

    # Ensure headroom for data and annotations below zero
    bottom_y = min(-0.024, min(all_diffs) - 0.015) if all_diffs else -0.025
    top_y = max(all_diffs) + 0.010 if all_diffs else 0.080
    ax.set_ylim(bottom=bottom_y, top=top_y)

    # Place legend outside plotting area
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    written = save_figure(fig, out_dir, "v3w_budgets", note=NOTE_BUDGETS, bottom_margin=0.18)
    return written, None


# ---------------------------------------------------------------------------
# Figure 3: v3w_residualized
# ---------------------------------------------------------------------------

def plot_v3w_residualized(eval_cells, resid_data, out_dir):
    """Figure 3: v3w_residualized.

    For all 10 subjects, paired markers of the raw stride-4 100% W1 − W0 test
    difference and the residualized W1 − W0 difference (from residualized.json),
    zero line, subjects on x grouped as in figure 1.
    """
    subjects = ALL_SUBJECTS
    has_any = any(
        (eval_cells.get(s, {}).get("W1", {}).get(1.0, {}).get("test_r") is not None
         and eval_cells.get(s, {}).get("W0", {}).get(1.0, {}).get("test_r") is not None)
        or (resid_data.get(s, {}).get("W1") is not None and resid_data.get(s, {}).get("W0") is not None)
        for s in subjects
    )
    if not has_any:
        return None, "No residualized or 100% evaluate data found across subjects"

    fig, ax = plt.subplots(figsize=(10.5, 5.0))

    format_axis(
        ax,
        is_log_x=False,
        xlabel="Subject",
        ylabel="Δ test session-macro r (W1 − W0)",
        title=None,
    )
    ax.set_title(
        "W1 − W0 Test Difference: Raw vs. Speech-Rate Residualized",
        fontsize=11, fontweight="bold", pad=22
    )

    x_indices = np.arange(len(subjects))
    ax.set_xticks(x_indices)
    ax.set_xticklabels(subjects, fontsize=9.5)
    ax.set_xlim(-0.6, len(subjects) - 0.4)

    # Vertical separator and group header labels
    ax.axvline(4.5, color="#c0c0c0", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        2.0, 1.02, "Exposed (UTS01–03, UTS09–10)",
        transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=9, color="#444444", fontweight="bold"
    )
    ax.text(
        7.0, 1.02, "Fresh confirmatory (UTS04–08)",
        transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=9, color="#444444", fontweight="bold"
    )

    # Horizontal zero line
    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)

    offset_raw = -0.12
    offset_resid = 0.12

    all_diffs = []
    labeled_series = set()

    for idx, s in enumerate(subjects):
        # Raw stride-4 100% difference
        w0_cell = eval_cells.get(s, {}).get("W0", {}).get(1.0, {})
        w1_cell = eval_cells.get(s, {}).get("W1", {}).get(1.0, {})
        w0_val = w0_cell.get("test_r")
        w1_val = w1_cell.get("test_r")
        raw_diff = (w1_val - w0_val) if (w0_val is not None and w1_val is not None) else None

        # Residualized difference
        s_resid = resid_data.get(s, {})
        r_w0 = s_resid.get("W0")
        r_w1 = s_resid.get("W1")
        resid_diff = (r_w1 - r_w0) if (r_w0 is not None and r_w1 is not None) else None

        # Connect pair with a thin line
        if raw_diff is not None and resid_diff is not None:
            ax.plot(
                [idx + offset_raw, idx + offset_resid], [raw_diff, resid_diff],
                color="#888888", linewidth=1.1, zorder=2
            )

        # Plot raw difference
        if raw_diff is not None:
            all_diffs.append(raw_diff)
            lbl = "Raw stride-4 (100% labels)" if "raw" not in labeled_series else None
            ax.plot(
                idx + offset_raw, raw_diff,
                marker="o", color="#0072B2", markersize=6.5, linestyle="",
                label=lbl, zorder=4
            )
            labeled_series.add("raw")

        # Plot residualized difference
        if resid_diff is not None:
            all_diffs.append(resid_diff)
            lbl = "Speech-rate residualized" if "resid" not in labeled_series else None
            ax.plot(
                idx + offset_resid, resid_diff,
                marker="s", color="#D55E00", markersize=6.5, linestyle="",
                label=lbl, zorder=4
            )
            labeled_series.add("resid")

    if all_diffs:
        min_d = min(all_diffs)
        max_d = max(all_diffs)
        ax.set_ylim(bottom=min(-0.012, min_d - 0.008), top=max_d + 0.012)

    # Place legend outside plotting area
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    written = save_figure(fig, out_dir, "v3w_residualized", note=NOTE_RESIDUALIZED, bottom_margin=0.18)
    return written, None


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/v3w"),
        help="Root directory for v3w evaluation results (default: results/v3w)",
    )
    parser.add_argument(
        "--v3-root",
        type=Path,
        default=Path("results/v3"),
        help="Root directory for v3 reference results (default: results/v3)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/v3w/figures"),
        help="Output directory for generated figures (default: results/v3w/figures)",
    )
    args = parser.parse_args()

    # Resolve relative paths relative to current working directory or repo root
    def resolve_path(p: Path) -> Path:
        if p.is_absolute():
            return p
        cwd_p = Path.cwd() / p
        if cwd_p.exists() or cwd_p.parent.exists():
            return cwd_p
        return repo_root / p

    root_dir = resolve_path(args.root)
    v3_root = resolve_path(args.v3_root)
    out_dir = resolve_path(args.out)

    print(f"Results v3w directory: {root_dir}")
    print(f"Results v3 directory:  {v3_root}")
    print(f"Figures out directory: {out_dir}")

    # Load data
    eval_cells = load_v3w_evaluate_cells(root_dir, ALL_SUBJECTS)
    resid_data = load_v3w_residualized(root_dir, ALL_SUBJECTS)
    dense_data = load_v3w_dense(root_dir, ALL_SUBJECTS)
    v3_ref = load_v3_reference(v3_root, ALL_SUBJECTS)

    n_eval = sum(len(eval_cells[s][arm]) for s in ALL_SUBJECTS for arm in eval_cells[s])
    n_dense = sum(1 for s in ALL_SUBJECTS if dense_data.get(s, {}).get("W0") or dense_data.get(s, {}).get("W1"))
    n_v3 = len(v3_ref)
    n_resid = len(resid_data)

    print(f"Loaded {n_eval} evaluate cells across {len(ALL_SUBJECTS)} subjects.")
    print(f"Loaded residualized evaluations for {n_resid} subjects.")
    print(f"Loaded dense evaluations for {n_dense} subjects.")
    print(f"Loaded v3 reference evaluations for {n_v3} subjects.")

    all_written_files = []
    produced_figures = []
    skipped_figures = []

    # Figure 1: v3w_subjects
    written, skip_reason = plot_v3w_subjects(eval_cells, dense_data, v3_ref, out_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("v3w_subjects")
    else:
        skipped_figures.append(("v3w_subjects", skip_reason))

    # Figure 2: v3w_budgets
    written, skip_reason = plot_v3w_budgets(eval_cells, out_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("v3w_budgets")
    else:
        skipped_figures.append(("v3w_budgets", skip_reason))

    # Figure 3: v3w_residualized
    written, skip_reason = plot_v3w_residualized(eval_cells, resid_data, out_dir)
    if written:
        all_written_files.extend(written)
        produced_figures.append("v3w_residualized")
    else:
        skipped_figures.append(("v3w_residualized", skip_reason))

    print("\n--- Execution Summary ---")
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
