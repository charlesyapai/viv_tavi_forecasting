# viz_v8.py — Plotting helpers for model_v8.py
# Separated for cleaner maintenance and editability.
#
# Functions here accept the DataFrames written by model_v8, and either show
# plots (if called in an interactive session) or save them to files when a
# `savepath` is provided. We intentionally keep styling minimal.

from __future__ import annotations
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _single_ax() -> "plt.Axes":
    fig, ax = plt.subplots(figsize=(9, 5.5))
    return ax


def plot_index_projection(proj_tavi: pd.DataFrame,
                          proj_savr: pd.DataFrame,
                          savepath: Optional[Path] = None) -> None:
    """
    Plot observed‑style projected index volumes for TAVI and SAVR by year (totals).

    WHEN USED: to visualize Step 3 outputs quickly.
    """
    t = proj_tavi.groupby("year")["projected"].sum()
    s = proj_savr.groupby("year")["projected"].sum()
    ax = _single_ax()
    ax.plot(t.index, t.values, label="TAVI (projected)")
    ax.plot(s.index, s.values, label="SAVR (projected)")
    ax.set_title("Projected Index Volumes (TAVI vs SAVR)")
    ax.set_xlabel("Year"); ax.set_ylabel("Projected patients")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight", dpi=160)
        plt.close(ax.figure)
    else:
        plt.show()


def plot_viv_pre_post(pre_summary: pd.DataFrame,
                      post_summary: pd.DataFrame,
                      savepath: Optional[Path] = None) -> None:
    """
    Plot total ViV (mean across runs) pre‑ vs post‑redo subtraction, with streams.

    WHEN USED: to visualize Step 7 effects.
    """
    ax = _single_ax()
    ax.plot(pre_summary["year"], pre_summary["total_viv_mean"], label="Total ViV (pre‑redo)")
    ax.plot(post_summary["year"], post_summary["total_viv_post"], label="Total ViV (post‑redo)")
    ax.plot(pre_summary["year"], pre_summary["tavr_in_tavr_mean"], label="TAVR‑in‑TAVR (pre)")
    ax.plot(pre_summary["year"], pre_summary["tavr_in_savr_mean"], label="TAVR‑in‑SAVR (pre)")
    ax.set_title("ViV Forecast: PRE vs POST redo-SAVR subtraction")
    ax.set_xlabel("Year"); ax.set_ylabel("Patients (mean across runs)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight", dpi=160)
        plt.close(ax.figure)
    else:
        plt.show()
