#!/usr/bin/env python3
"""
model_v9.py — Demography‑anchored ViV‑TAVI Monte Carlo forecaster
----------------------------------------------------------------------------------
This script preserves the execution flow of model_v7 (CLI, folder structure, plots)
and *adds* a built‑in pre‑processor to:
  • Build annual age/sex population projections for 2025–2035 from
    simulation_run_v1/data/korea_population_combined.csv
  • Compute fixed per‑capita risk scores for TAVI, SAVR, and redo‑SAVR from 2023/2024
  • (Optionally) project absolute redo‑SAVR yearly targets (to subtract from TAVR‑in‑SAVR)

New in this revision:
  • Demography plots: age‑band lines by sex; year×age heatmap (total population)
  • Risk plots: per‑procedure heatmaps for 2023/2024 and delta; grouped bar summary
    using the configured risk rule (avg_2023_2024 by default)

After precomputation, it runs the same Monte‑Carlo engine and output regimen as v7:
  runs/<experiment_name>/<timestamp>/... (tables, QC, figures).

Run (same style as v7):
    python models/model_v9.py --config configs/model_v9_configs.yaml --log-level DEBUG

Key idea
--------
Index volumes are not extrapolated by time‑trend. Instead we estimate
age‑specific *per‑capita* procedure rates from 2023–2024 and multiply by
projected population to form the baseline index volumes for 2025–2035.
Durability, survival, penetration, and redo mechanics remain as before.

New plots (demography + risk)
-----------------------------
Under figures/demography:
  • age_projection_lines_Men.png / age_projection_lines_Women.png
    – Lines by 5‑year bands (50–54..≥85) for 2025–2035 showing projected population
  • age_heatmap_allsex.png
    – Heatmap (Year × Age) of population (Men+Women), helpful to eyeball cohort waves

Under figures/risks (for each of {tavi, savr, redo_savr}):
  • <proc>_heatmap_2023.png, <proc>_heatmap_2024.png
    – Heatmaps of per‑capita risk by (Sex × Age‑band)
  • <proc>_heatmap_delta_2024_vs_2023.png
    – Relative change (2024 − 2023); highlights drift in uptake by group
  • <proc>_bar_avg_risks.png
    – Grouped bars using the configured 'risk_rule' (avg_2023_2024 by default) to
      summarize risks by age‑band, split by sex
"""

from __future__ import annotations

from typing import Any

import argparse
import logging
import sys
import re
import shutil
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator
from scipy.stats import norm
import yaml
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick

import warnings
import pydantic
from collections import defaultdict

# =============================================================================
# Plotting options / theme (unchanged from v7)
# =============================================================================

class PlottingConfig(BaseModel):
    background: str = "light"                 # "light" | "darkgrey" | "black"
    viv_total_bar_color: Optional[str] = None # e.g. "#aaaaaa"
    label_bars: bool = True
    image_c_source: str = "post"   # "post" | "pre" | "both"


def _apply_plot_theme(plot: Optional['PlottingConfig']) -> None:
    plt.rcParams.update(plt.rcParamsDefault)
    if not plot or str(plot.background).lower() == "light":
        return
    bg = str(plot.background).lower()
    fig_face = "#000000" if bg in ("black", "#000", "#000000") else "#222222"
    plt.style.use("dark_background")
    rc = plt.rcParams
    rc["figure.facecolor"]  = fig_face
    rc["axes.facecolor"]    = fig_face
    rc["savefig.facecolor"] = fig_face
    rc["axes.edgecolor"]    = "white"
    rc["axes.labelcolor"]   = "white"
    rc["xtick.color"]       = "white"
    rc["ytick.color"]       = "white"
    rc["text.color"]        = "white"
    rc["legend.facecolor"]  = fig_face
    rc["grid.color"]        = "#888888"
    rc["grid.alpha"]        = 0.25

def _savefig_current(path: Path) -> None:
    fig = plt.gcf()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

# =============================================================================
# Logging
# =============================================================================

warnings.filterwarnings("ignore", category=pydantic.PydanticDeprecatedSince20)

def setup_logging(level: str, log_dir: Path) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / "run.log", mode="w")]
    logging.basicConfig(level=numeric, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)

log = logging.getLogger(__name__)

# =============================================================================
# Config models (Pydantic v2)
# =============================================================================

class DurabilityMode(BaseModel):
    mean: float
    sd: float
    weight: float = 1.0

class ConstantSettings(BaseModel):
    from_year: int
    value: int

class PopRateSettings(BaseModel):
    source: str = "from_observed"    # from_observed | from_config
    obs_years: str | List[int] | None = None
    min_age: int = 0
    annual_rate_growth: Optional[float] = None
    anchors: Optional[Dict[str, float]] = None  # year→multiplier
    rates_by_age: Optional[Dict[int, float]] = None  # if source=from_config

class ProcProjection(BaseModel):
    method: str = "linear"                          # linear|constant|population_rate
    window: int = 5
    constant: Optional[ConstantSettings] = None
    pop_rate: Optional[PopRateSettings] = None

    @field_validator("method")
    @classmethod
    def _method_ok(cls, v):
        if v not in ("linear", "constant", "population_rate"):
            raise ValueError("method must be linear|constant|population_rate")
        return v

class IndexProjection(BaseModel):
    tavi: Optional[ProcProjection] = None
    savr: Optional[ProcProjection] = None

# ---- NEW: Precompute knobs ---------------------------------------------------
class PrecomputeConfig(BaseModel):
    enabled: bool = True
    # population file + units
    population_source: Dict[str, str | float] = {
        "path": "data/korea_population_combined.csv",
        "units_multiplier": 10000.0
    }
    # which years to build and use as risk baselines
    risk_years: List[int] = [2023, 2024]
    # combine rule for risks going forward
    risk_rule: str = "avg_2023_2024"  # 2024_only | 2023_only | avg_2023_2024
    # interpolation span for projections
    project_years: List[int] = [2025, 2035]
    # how to split broader brackets to 5y bands
    split_75_84: List[float] = [0.5, 0.5]  # [share_75_79, share_80_84]
    split_50_64: List[float] = [1/3, 1/3, 1/3]  # 50-54, 55-59, 60-64
    # assumed sex ratio for 50-64 (men per 100 women) where none is provided
    sex_ratio_50_64: float = 100.0
    # open width (≥85 will be expanded to ≥85..hi for single-year ages)
    open_age_bin_width: int = 20
    # where to write derived files
    derived_dir: str = "derived"
    # whether to build redo‑SAVR absolute targets by risk×population
    build_redo_savr_targets: bool = True

class Config(BaseModel):
    experiment_name: str
    scenario_tag: Optional[str] = None

    years: Dict[str, int]
    procedure_counts: Dict[str, Path]

    index_projection: Optional[IndexProjection] = None
    volume_extrapolation: Optional[Dict[str, int | str | dict]] = None

    open_age_bin_width: int
    age_bins: List[int]

    durability: Dict[str, Dict[str, DurabilityMode]] | Dict[str, Dict[str, Dict[str, DurabilityMode]]]
    survival_curves: Dict[str, Dict[str, float]]

    risk_model: Dict[str, object]
    risk_mix: Dict[str, Dict[str, float] | Dict[str, Dict[str, float]]]
    age_hazard: Optional[Dict[str, object]] = None

    penetration: Dict[str, Dict[str, float]]
    redo_rates: Dict[str, float]

    simulation: Dict[str, Any]
    outputs: Dict[str, str]     # unused but kept for compat

    population_projection: Optional[Dict[str, str]] = None
    redo_savr_numbers: Optional[Dict[str, object]] = None
    precompute: Optional[PrecomputeConfig] = PrecomputeConfig()

    figure_ranges: Optional[Dict[str, List[int]]] = None
    plotting: Optional[PlottingConfig] = None

    # validators
    @field_validator("procedure_counts")
    @classmethod
    def _check_csvs(cls, v: Dict[str, Path]):
        for tag, path in v.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing csv for '{tag}': {path}")
        return v

    @field_validator("risk_model")
    @classmethod
    def _risk_model_ok(cls, v: Dict[str, object]):
        if v.get("categories") not in (2, 3):
            raise ValueError("risk_model.categories must be 2 or 3")
        return v

    @field_validator("age_bins")
    @classmethod
    def _age_bins_ok(cls, v: List[int]):
        if len(v) < 2 or any(b >= v[idx + 1] for idx, b in enumerate(v[:-1])):
            raise ValueError("age_bins must be ascending with ≥2 entries")
        return v

    @model_validator(mode="after")
    def _merge_v5_fallback(self):
        if self.index_projection is None:
            window = 5
            tavi_method = "linear"
            savr_method = "constant"
            savr_from = 2025
            savr_value = 2000
            ve = self.volume_extrapolation or {}
            if ve:
                window = int(ve.get("window", window))
                savr_cfg = ve.get("savr", {})
                if isinstance(savr_cfg, dict) and savr_cfg:
                    savr_method = str(savr_cfg.get("method", savr_method))
                    savr_from = int(savr_cfg.get("from_year", savr_from))
                    savr_value = int(savr_cfg.get("value", savr_value))
                if "method" in ve:   # top-level applies to TAVI
                    tavi_method = str(ve["method"])
            self.index_projection = IndexProjection(
                tavi=ProcProjection(method=tavi_method, window=window),
                savr=ProcProjection(
                    method=savr_method, window=window,
                    constant=ConstantSettings(from_year=savr_from, value=savr_value)
                )
            )
        return self

# =============================================================================
# Small path helper to keep folders tidy (v7-compatible)
# =============================================================================

class Dirs:
    def __init__(self, root: Path):
        self.root = root
        self.logs = root / "logs"
        self.meta = root / "meta"
        self.inputs_snapshot = root / "inputs_snapshot"
        self.tables = root / "tables"
        self.tables_index = self.tables / "index"
        self.tables_flow  = self.tables / "flow"
        self.tables_viv   = self.tables / "viv"
        self.tables_viv_per_run = self.tables_viv / "per_run"
        self.qc = root / "qc"
        self.qc_index = self.qc / "index_projection"
        self.qc_redo   = self.qc / "redo"
        self.figures = root / "figures"
        self.fig_index = self.figures / "index"
        self.fig_flow  = self.figures / "flow"
        self.fig_viv   = self.figures / "viv"
        self.fig_risks = self.figures / "risks"  

    def make_all(self):
        for p in (self.logs, self.meta, self.inputs_snapshot,
                  self.tables_index, self.tables_flow, self.tables_viv, self.tables_viv_per_run,
                  self.qc_index, self.qc_redo,
                  self.fig_index, self.fig_flow, self.fig_viv,
                  self.fig_risks):
            p.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Utilities (parsing / math)
# =============================================================================


def parse_age_band(band: str, open_width: int) -> tuple[int, int]:
    """Parses age-band labels like '<5yr', '5-9', '≥80', '>=85', '>80', '80+', etc."""
    s = str(band).strip().lower()
    # Normalize Unicode and variants
    s = (
        s.replace("≥", ">=").replace("≤", "<=")
         .replace("–", "-").replace("—", "-")
         .replace("yrs", "").replace("yr", "").replace("years", "").replace("y", "")
         .strip()
    )

    # Match ≥ or > style ("≥85", ">=85", ">85")
    m = re.match(r'^(>=|>)\s*(\d+)$', s)
    if m:   
        lo = int(m.group(2))
        if m.group(1) == '>':
            lo += 1
        return lo, lo + open_width

    # Match ≤ or < style ("<5", "<5y", "<5yr", "<=5")
    m = re.match(r'^(<=|<)\s*(\d+)$', s)
    if m:
        hi = int(m.group(2))
        if m.group(1) == '<':
            hi = hi
        return 0, hi

    # Match ranges ("50-54", "25 - 29", etc.)
    m = re.match(r'^(\d+)\s*-\s*(\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2)) + 1

    raise ValueError(f"Unrecognised age band: '{band}'")

def _age_labels(edges: List[int]) -> List[str]:
    labs = [f"<{edges[0]}"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        labs.append(f"{lo}-{hi-1}")
    labs.append(f"≥{edges[-1]}")
    return labs

def _largest_remainder_split(total: int, shares: np.ndarray) -> np.ndarray:
    shares = np.clip(shares, 0, None)
    if shares.sum() == 0:
        shares = np.ones_like(shares, dtype=float)
    raw = shares / shares.sum() * total
    base = np.floor(raw).astype(int)
    r = total - base.sum()
    if r > 0:
        idx = np.argsort(raw - base)[-r:]
        base[idx] += 1
    return base

def _interp_scalar_by_year(year: int, anchors: Dict[str, float]) -> float:
    pts: List[Tuple[int, float]] = []
    for k, v in anchors.items():
        if '-' in k:
            a, b = map(int, k.split('-'))
            pts += [(a, float(v)), (b, float(v))]
        else:
            pts.append((int(k), float(v)))
    pts.sort()
    xs, ys = zip(*pts)
    return float(np.interp(year, xs, ys))

# =============================================================================
# ------------------------ Pre‑compute (NEW in v9) -----------------------------
# =============================================================================

def _read_population_table(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    df = pd.read_csv(path)
    rows = df.iloc[:, 0].astype(str).str.strip()
    if not rows.str.contains("Sex", case=False).any():
        raise ValueError("Population CSV must contain a 'Sex ratio' section header row.")
    split_idx = rows[rows.str.contains("Sex", case=False)].index[0]
    counts = df.iloc[:split_idx, :].copy()
    ratios  = df.iloc[split_idx+1:, :].copy()
    counts.iloc[:, 0] = counts.iloc[:, 0].astype(str).str.strip()
    ratios.iloc[:, 0]  = ratios.iloc[:, 0].astype(str).str.strip()
    year_cols_counts = [c for c in counts.columns[1:] if re.match(r'^\d{4}$', str(c))]
    year_cols_ratios = [c for c in ratios.columns[1:] if re.match(r'^\d{4}$', str(c))]
    years_sorted = sorted(set(map(int, year_cols_counts)) | set(map(int, year_cols_ratios)))
    counts = counts[[counts.columns[0]] + [str(y) for y in years_sorted]]
    ratios  = ratios[[ratios.columns[0]]  + [str(y) for y in years_sorted]]
    counts = counts.set_index(counts.columns[0])
    ratios  = ratios.set_index(ratios.columns[0])
    counts = counts.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    ratios  = ratios.apply(pd.to_numeric, errors="coerce").fillna(np.nan)
    return counts, ratios, years_sorted

def _interpolate_series(xs: list[int], ys_known: dict[int,float], years_target: list[int]) -> np.ndarray:
    xs_sort = sorted(xs)
    yvec = np.array([ys_known.get(x, np.nan) for x in xs_sort], dtype=float)
    yvec = pd.Series(yvec).interpolate().fillna(method="bfill").fillna(method="ffill").to_numpy()
    return np.interp(years_target, xs_sort, yvec)

def build_age_and_sex_population(pop_csv: Path,
                                 out_dir: Path,
                                 units_multiplier: float,
                                 project_span: tuple[int,int],
                                 split_50_64: tuple[float,float,float],
                                 split_75_84: tuple[float,float],
                                 sex_ratio_50_64: float,
                                 open_width: int) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, ratios, years_available = _read_population_table(pop_csv)
    y0, y1 = int(project_span[0]), int(project_span[1])
    years_target = list(range(y0, y1+1))

    def interp_row(label: str, target_years: list[int], is_ratio=False) -> np.ndarray:
        table = ratios if is_ratio else counts
        if label not in table.index:
            raise KeyError(f"Row '{label}' not found in population file.")
        known = {int(c): float(table.at[label, str(c)]) for c in table.columns}
        return _interpolate_series(sorted(known.keys()), known, target_years)

    pop_50_64 = interp_row("50-64", years_target, is_ratio=False) * units_multiplier
    pop_ge65  = interp_row("≥65",   years_target, is_ratio=False) * units_multiplier
    pop_ge70  = interp_row("≥70",   years_target, is_ratio=False) * units_multiplier
    pop_ge75  = interp_row("≥75",   years_target, is_ratio=False) * units_multiplier
    pop_ge85  = interp_row("≥85",   years_target, is_ratio=False) * units_multiplier

    pop_65_69 = pop_ge65 - pop_ge70
    pop_70_74 = pop_ge70 - pop_ge75
    pop_75_84 = pop_ge75 - pop_ge85
    s75, s80 = float(split_75_84[0]), float(split_75_84[1])
    pop_75_79 = pop_75_84 * s75 / (s75 + s80)
    pop_80_84 = pop_75_84 * s80 / (s75 + s80)

    s1, s2, s3 = float(split_50_64[0]), float(split_50_64[1]), float(split_50_64[2])
    denom = (s1 + s2 + s3)
    pop_50_54 = pop_50_64 * s1 / denom
    pop_55_59 = pop_50_64 * s2 / denom
    pop_60_64 = pop_50_64 * s3 / denom

    r_ge65 = interp_row("≥65", years_target, is_ratio=True)
    r_ge70 = interp_row("≥70", years_target, is_ratio=True)
    r_ge75 = interp_row("≥75", years_target, is_ratio=True)
    r_ge85 = interp_row("≥85", years_target, is_ratio=True)
    r_50_64 = np.full_like(r_ge65, float(sex_ratio_50_64), dtype=float)

    def split_sex(total: np.ndarray, ratio_m_per100w: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        men = total * (ratio_m_per100w / (100.0 + ratio_m_per100w))
        women = total - men
        return men, women

    m_50_54, f_50_54 = split_sex(pop_50_54, r_50_64)
    m_55_59, f_55_59 = split_sex(pop_55_59, r_50_64)
    m_60_64, f_60_64 = split_sex(pop_60_64, r_50_64)
    m_65_69, f_65_69 = split_sex(pop_65_69, r_ge65)
    m_70_74, f_70_74 = split_sex(pop_70_74, r_ge70)
    m_75_79, f_75_79 = split_sex(pop_75_79, r_ge75)
    m_80_84, f_80_84 = split_sex(pop_80_84, r_ge75)
    m_ge85,  f_ge85  = split_sex(pop_ge85,  r_ge85)

    bands = ["50-54","55-59","60-64","65-69","70-74","75-79","80-84","≥85"]
    men_series   = [m_50_54, m_55_59, m_60_64, m_65_69, m_70_74, m_75_79, m_80_84, m_ge85]
    women_series = [f_50_54, f_55_59, f_60_64, f_65_69, f_70_74, f_75_79, f_80_84, f_ge85]

    rows = []
    for j, y in enumerate(years_target):
        for b, mvec, fvec in zip(bands, men_series, women_series):
            rows.append([y, "Men", b, int(round(mvec[j]))])
            rows.append([y, "Women", b, int(round(fvec[j]))])
    by_band_sex = pd.DataFrame(rows, columns=["year","sex","age_band","population"])

    def expand_to_ages(df_band: pd.DataFrame, open_width: int) -> pd.DataFrame:
        rows2 = []
        for (y, sex, band), grp in df_band.groupby(["year","sex","age_band"]):
            total = int(grp["population"].sum())
            lo, hi = parse_age_band(band, open_width)
            ages = list(range(lo, hi))
            shares = np.ones(len(ages), dtype=float)
            alloc = _largest_remainder_split(total, shares)
            for a, n in zip(ages, alloc):
                if n > 0:
                    rows2.append([y, a, sex, n])
        return pd.DataFrame(rows2, columns=["year","age","sex","population"])

    by_age_sex = expand_to_ages(by_band_sex, open_width=open_width)
    by_age_all = (by_age_sex.groupby(["year","age"])["population"].sum()
                            .reset_index().rename(columns={"population":"population"}))

    p_band_sex = out_dir / "age_projections_by_year_band_sex.csv"
    p_age_sex  = out_dir / "age_projections_by_year_age_sex.csv"
    p_age_all  = out_dir / "age_projections_by_year_age_allsex.csv"
    by_band_sex.to_csv(p_band_sex, index=False)
    by_age_sex.to_csv(p_age_sex, index=False)
    by_age_all.to_csv(p_age_all, index=False)

    return p_band_sex, p_age_sex, p_age_all

def compute_risk_scores(proc_files: Dict[str, Path],
                        population_band_sex_csv: Path,
                        risk_years: list[int]) -> Path:
    pop = pd.read_csv(population_band_sex_csv)
    pop = pop[pop["year"].isin(risk_years)]
    def population_for(sex: str, band: str, year: int) -> int:
        if band.strip() in ("≥80", ">=80", "80+"):
            p80_84 = int(pop[(pop.sex==sex)&(pop.age_band=="80-84")&(pop.year==year)]["population"].sum())
            pge85  = int(pop[(pop.sex==sex)&(pop.age_band=="≥85")&(pop.year==year)]["population"].sum())
            return p80_84 + pge85
        return int(pop[(pop.sex==sex)&(pop.age_band==band)&(pop.year==year)]["population"].sum())

    rows = []
    for proc in ["tavi", "savr", "redo_savr"]:
        if proc not in proc_files:
            continue
        df = pd.read_csv(proc_files[proc])
        cols = {c.lower(): c for c in df.columns}
        req = ["sex","age_band","year","count"]
        if not all(k in cols for k in req):
            raise ValueError(f"{proc} CSV must have columns: sex, age_band, year, count")
        df = df[[cols["sex"], cols["age_band"], cols["year"], cols["count"]]].copy()
        df.columns = ["sex","age_band","year","count"]
        df = df[df["year"].isin(risk_years)]
        for (sex, band, year), grp in df.groupby(["sex","age_band","year"]):
            obs = int(grp["count"].sum())
            pop_denom = population_for(sex, str(band), int(year))
            if pop_denom <= 0:
                continue
            risk = obs / float(pop_denom)
            rows.append([proc, int(year), str(sex), str(band), float(risk)])
    out = pd.DataFrame(rows, columns=["proc","year","sex","age_band","risk_per_person"])
    p_out = population_band_sex_csv.parent / "baseline_risk_scores_by_sex.csv"
    out.to_csv(p_out, index=False)
    return p_out

def project_redo_savr_targets(risk_scores_csv: Path,
                              population_band_sex_csv: Path,
                              risk_rule: str,
                              project_span: tuple[int,int]) -> Path:
    risks = pd.read_csv(risk_scores_csv)
    pop   = pd.read_csv(population_band_sex_csv)
    y0, y1 = int(project_span[0]), int(project_span[1])

    def pick_risk(sub: pd.DataFrame) -> float:
        if risk_rule == "2023_only":
            s = sub[sub["year"] == 2023]["risk_per_person"]
            return 0.0 if s.empty else float(s.mean())
        if risk_rule == "2024_only":
            s = sub[sub["year"] == 2024]["risk_per_person"]
            return 0.0 if s.empty else float(s.mean())
        return float(sub["risk_per_person"].mean())

    risks_rs = risks[risks["proc"] == "redo_savr"]
    key_to_risk: Dict[tuple[str,str], float] = {}
    for (sex, band), grp in risks_rs.groupby(["sex","age_band"]):
        key_to_risk[(str(sex), str(band))] = pick_risk(grp)

    def pop_for(year: int, sex: str, band: str) -> int:
        if band.strip() in ("≥80", ">=80", "80+"):
            p80_84 = int(pop[(pop.sex==sex)&(pop.age_band=="80-84")&(pop.year==year)]["population"].sum())
            pge85  = int(pop[(pop.sex==sex)&(pop.age_band=="≥85")&(pop.year==year)]["population"].sum())
            return p80_84 + pge85
        return int(pop[(pop.sex==sex)&(pop.age_band==band)&(pop.year==year)]["population"].sum())

    bands = sorted(risks_rs["age_band"].astype(str).unique(),
                   key=lambda s: parse_age_band(s, open_width=20)[0])

    rows = []
    for y in range(y0, y1+1):
        total = 0.0
        for sex in ("Men","Women"):
            for b in bands:
                r = key_to_risk.get((sex, b), 0.0)
                if r <= 0:
                    continue
                total += r * pop_for(y, sex, b)
        rows.append([y, int(round(total))])
    out = pd.DataFrame(rows, columns=["year","count"])
    p_out = population_band_sex_csv.parent / "redo_savr_targets.csv"
    out.to_csv(p_out, index=False)
    return p_out

# =============================================================================
# ------------------------ Demography & Risk plotting (NEW) --------------------
# =============================================================================

# --- ADD: common x-axis and focus band helper ---
def _apply_xrange_and_focus(ax, cfg, default=(2015, 2050)):
    xr = None
    fb = None
    if getattr(cfg, "figure_ranges", None):
        xr = cfg.figure_ranges.get("x_axis", None)
        fb = cfg.figure_ranges.get("focus_band", None)
    if xr is None:
        xr = default
    if xr:
        ax.set_xlim(xr[0], xr[1])
    if fb:
        ax.axvspan(fb[0], fb[1], color="gold", alpha=0.12, zorder=0)


# --- ADD: stacked index snapshots by age-band (per procedure) ---
def _plot_index_stacked_by_band(df: pd.DataFrame, proc_name: str,
                                snapshot_years: list[int], out_path: Path,
                                cfg=None) -> None:
    # df columns: year, age_band, src, count ...
    snap = df[df["year"].isin(snapshot_years)].copy()
    if snap.empty:
        return
    order = sorted(snap["age_band"].astype(str).unique(),
                   key=lambda s: parse_age_band(s, getattr(cfg, "open_age_bin_width", 20))[0])
    pivot = (snap.groupby(["year","age_band"])["count"].sum()
                  .unstack(fill_value=0).reindex(index=sorted(snapshot_years)))
    pivot = pivot[[c for c in order if c in pivot.columns]]

    plt.figure(figsize=(10,5), dpi=140)
    bottom = np.zeros(len(pivot), dtype=float)
    for col in pivot.columns:
        plt.bar(pivot.index, pivot[col].values, bottom=bottom, label=col)
        bottom += pivot[col].values

    ax = plt.gca()
    _apply_xrange_and_focus(ax, cfg)  # will just show the band on the x-axis
    plt.title(f"{proc_name.upper()} index composition by age-band (snapshots)")
    plt.xlabel("Year (snapshots)"); plt.ylabel("Procedures / yr")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    _savefig_current(out_path)

# --- ADD: waterfall per pathway for a given year ---
def _plot_waterfall_for_year(flow_mean: pd.DataFrame,
                             realized_summary: pd.DataFrame,
                             redo_targets: dict[int,int] | None,
                             year: int,
                             out_dir: Path) -> None:
    """
    Build a simple waterfall for each pathway in a given year:
      (failures) -> (viable) -> (penetration -> realized pre) -> (minus redo for TIS) -> (realized post)
    Notes:
      • 'failures' includes valves that failed even if the patient had died earlier.
      • 'viable' are failures with fail_year <= death_year (eligible pool).
      • realized_pre = (realized_post + redo_target) when redo targets exist; else use realized_pre ≈ realized_post (no subtraction).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    y = int(year)
    # Flow
    f = flow_mean.copy()
    row_s = f[(f["year"]==y) & (f["proc"]=="savr")]
    row_t = f[(f["year"]==y) & (f["proc"]=="tavi")]
    if row_s.empty or row_t.empty:  # nothing to plot
        return

    # Realized POST totals by type
    post = realized_summary.copy()
    post = post[post["year"]==y].set_index("viv_type")["mean"]

    def waterfall_one(tag: str, failures: float, viable: float, realized_post: float, redo: float):
        steps = [
            ("Failures", failures),
            ("Viable (alive at fail)", viable),
            ("Realized pre-REDO", realized_post + redo),
            ("REDO subtract", -redo),
            ("Realized (post)", realized_post),
        ]
        fig, ax = plt.subplots(figsize=(8,4), dpi=140)
        x, y_cum = [], []
        total = 0.0
        for i, (label, val) in enumerate(steps):
            x.append(label)
            y_cum.append(val)
        colors = ["#999999", "#5a9", "#58a", "#d66", "#3366cc"]
        running = 0.0
        for i, (label, val) in enumerate(steps):
            if val >= 0:
                ax.bar(i, val, bottom=running, color=colors[i % len(colors)])
                running += val
            else:
                ax.bar(i, val, bottom=running, color=colors[i % len(colors)])
                running += val
            ax.text(i, running, f"{int(round(running))}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(x))); ax.set_xticklabels(x, rotation=15, ha="right")
        ax.set_title(f"Waterfall {tag.upper()} — {y}")
        ax.set_ylabel("Head-count")
        fig.tight_layout()
        _savefig_current(out_dir / f"waterfall_{tag}_{y}.png")

    # Compute values
    failures_s = float(row_s["failures"].values[0])
    viable_s   = float(row_s["viable"].values[0])
    failures_t = float(row_t["failures"].values[0])
    viable_t   = float(row_t["viable"].values[0])

    realized_tis = float(post.get("tavi_in_savr", 0.0))
    realized_tit = float(post.get("tavi_in_tavi", 0.0))
    redo = float((redo_targets or {}).get(y, 0))
    waterfall_one("tavr_in_savr", failures_s, viable_s, realized_tis, redo)
    waterfall_one("tavr_in_tavr", failures_t, viable_t, realized_tit, 0.0)



def _plot_demography(by_band_sex_csv: Path, by_age_sex_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    band_df = pd.read_csv(by_band_sex_csv)
    age_df  = pd.read_csv(by_age_sex_csv)

    bands_order = ["50-54","55-59","60-64","65-69","70-74","75-79","80-84","≥85"]
    for sex in ["Men","Women"]:
        sub = band_df[band_df["sex"]==sex]
        plt.figure(figsize=(10,6))
        for b in bands_order:
            grp = (sub[sub["age_band"]==b].sort_values("year"))
            if grp.empty: 
                continue
            plt.plot(grp["year"], grp["population"], marker="o", label=b)
        plt.title(f"Projected population by age band ({sex})")
        plt.xlabel("Year"); plt.ylabel("Population (heads)")
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        _savefig_current(out_dir / f"age_projection_lines_{sex}.png")

    allsex = (age_df.groupby(["year","age"])["population"].sum().reset_index())
    years = sorted(allsex["year"].unique())
    ages  = sorted(allsex["age"].unique())
    grid = np.zeros((len(years), len(ages)), dtype=float)
    for i, y in enumerate(years):
        row = allsex[allsex["year"]==y].set_index("age")["population"]
        for j, a in enumerate(ages):
            grid[i, j] = float(row.get(a, 0.0))
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)
    im = ax.imshow(grid, aspect="auto", origin="lower")
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years)
    step = max(1, len(ages)//16)
    ax.set_xticks(list(range(0,len(ages),step))); ax.set_xticklabels([ages[i] for i in range(0,len(ages),step)])
    ax.set_xlabel("Age"); ax.set_ylabel("Year")
    ax.set_title("Projected population heatmap (Men + Women)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Population (heads)")
    fig.tight_layout()
    fig.savefig(out_dir / "age_heatmap_allsex.png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


# --- Risk score plotting ------------------------------------------------

def _plot_risk_scores(risk_csv, out_dir, risk_rule=None):
    """
    Reads baseline_risk_scores_by_sex.csv and draws:
      • heatmaps for 2023/2024
      • a delta heatmap (2024–2023)
      • bar charts of average risk by age-band, split by sex
    """
    if not risk_csv.exists():
        log.warning("Risk scores CSV %s not found; skipping risk plots.", risk_csv)
        return

    df = pd.read_csv(risk_csv)
    if df.empty:
        log.warning("Risk scores CSV %s is empty; skipping risk plots.", risk_csv)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    procedures = sorted(df["procedure"].unique())
    years = sorted(df["year"].unique())

    for proc in procedures:
        dfp = df[df["procedure"] == proc].copy()
        # Normalize age_band ordering using parse_age_band
        bands = sorted(dfp["age_band"].astype(str).unique(),
                       key=lambda s: parse_age_band(s, 20)[0])

        # Heatmaps for each year
        for y in years:
            sub = dfp[dfp["year"] == y]
            if sub.empty:
                continue
            pivot = (sub.pivot(index="age_band", columns="sex", values="risk")
                        .reindex(index=bands))
            fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
            im = ax.imshow(pivot.values, aspect="auto")
            ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)));  ax.set_yticklabels(pivot.index)
            ax.set_title(f"{proc.upper()} risk {y}")
            fig.colorbar(im, ax=ax, label="per-capita risk")
            fig.tight_layout()
            _savefig_current(out_dir / f"{proc}_heatmap_{y}.png")

        # Delta (last year minus first year)
        if len(years) >= 2:
            y0, y1 = years[0], years[-1]
            p0 = (dfp[dfp["year"] == y0]
                  .pivot(index="age_band", columns="sex", values="risk")
                  .reindex(index=bands))
            p1 = (dfp[dfp["year"] == y1]
                  .pivot(index="age_band", columns="sex", values="risk")
                  .reindex(index=bands))
            delta = p1 - p0
            fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
            im = ax.imshow(delta.values, aspect="auto", cmap="bwr")
            ax.set_xticks(range(len(delta.columns))); ax.set_xticklabels(delta.columns)
            ax.set_yticks(range(len(delta.index)));  ax.set_yticklabels(delta.index)
            ax.set_title(f"{proc.upper()} risk Δ {y1}–{y0}")
            fig.colorbar(im, ax=ax, label="risk change")
            fig.tight_layout()
            _savefig_current(out_dir / f"{proc}_heatmap_delta_{y1}_vs_{y0}.png")

        # Bar chart of average risk across years
        grp = (dfp.groupby(["sex","age_band"])["risk"].mean().reset_index())
        grp["age_band"] = grp["age_band"].astype(str)
        grp = grp.sort_values("age_band", key=lambda s: s.map(
            lambda x: parse_age_band(x, 20)[0]
        ))
        fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
        bands_ord = grp["age_band"].unique().tolist()
        x = np.arange(len(bands_ord))
        width = 0.35
        for i, sex in enumerate(["Men","Women"]):
            sub = grp[grp["sex"] == sex]
            vals = [sub[sub["age_band"] == b]["risk"].mean() if b in sub["age_band"].values else 0.0
                    for b in bands_ord]
            ax.bar(x + (i - 0.5) * width, vals, width, label=sex)
        ax.set_xticks(x); ax.set_xticklabels(bands_ord, rotation=30, ha="right")
        ax.set_ylabel("Average per-capita risk")
        ax.set_title(f"{proc.upper()} average risk by age-band and sex")
        ax.legend()
        fig.tight_layout()
        _savefig_current(out_dir / f"{proc}_bar_avg_risks.png")

# =============================================================================
# --------------------------- Engine (v7 core) ---------------------------------
# =============================================================================

def _parse_year_selection(spec, available_years: np.ndarray) -> List[int]:
    if spec is None:
        return list(available_years)
    if isinstance(spec, list):
        return [int(x) for x in spec]
    s = str(spec).strip()
    if '-' in s:
        a, b = s.split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]

def _age_band_list(df: pd.DataFrame, open_width: int) -> List[str]:
    bands = df["age_band"].dropna().astype(str).unique().tolist()
    bands.sort(key=lambda s: parse_age_band(s, open_width)[0])
    return bands

def _estimate_rates_from_observed(obs_df: pd.DataFrame,
                                  pop_df: pd.DataFrame,
                                  years_sel: List[int],
                                  open_width: int,
                                  ycol: str, acol: str, pcol: str,
                                  min_age: int = 0) -> Dict[int, float]:
    bands = _age_band_list(obs_df, open_width)
    per_age_rate_year: Dict[int, List[float]] = defaultdict(list)
    for y in years_sel:
        dfy = obs_df[obs_df["year"] == y]
        if dfy.empty:
            continue
        popy = pop_df[pop_df[ycol] == y]
        if popy.empty:
            avail = pop_df[ycol].unique()
            if len(avail) == 0:
                continue
            closest = int(avail[np.argmin(np.abs(avail - y))])
            popy = pop_df[pop_df[ycol] == closest]
        pop_by_age = popy.set_index(acol)[pcol].to_dict()
        for band in bands:
            lo, hi = parse_age_band(band, open_width)
            pop_in_band = sum(pop_by_age.get(a, 0) for a in range(lo, hi))
            cnt = int(dfy[dfy["age_band"] == band]["count"].sum())
            if pop_in_band <= 0:
                continue
            band_rate = cnt / pop_in_band
            for a in range(lo, hi):
                per_age_rate_year[a].append(band_rate)
    base_rate: Dict[int, float] = {}
    if not per_age_rate_year:
        return base_rate
    for a, rlist in per_age_rate_year.items():
        if rlist:
            base_rate[a] = max(0.0, float(np.mean(rlist)))
    for a in list(base_rate.keys()):
        if a < min_age:
            base_rate[a] = 0.0
    return base_rate

def _project_index_by_pop_rate(obs_df: pd.DataFrame,
                               end_year: int,
                               pop_df: pd.DataFrame,
                               open_width: int,
                               ycol: str, acol: str, pcol: str,
                               pr: PopRateSettings) -> pd.DataFrame:
    out_rows = []
    last_obs_year = int(obs_df["year"].max())
    obs_df = obs_df.copy()
    obs_df["src"] = "observed"
    out_rows.append(obs_df)
    if pr.source == "from_config" and pr.rates_by_age:
        base_rate = {int(k): float(v) for k, v in pr.rates_by_age.items()}
    else:
        years_sel = _parse_year_selection(pr.obs_years, obs_df["year"].unique())
        base_rate = _estimate_rates_from_observed(obs_df, pop_df, years_sel,
                                                  open_width, ycol, acol, pcol, pr.min_age)
    if not base_rate:
        log.warning("Population-based rates unavailable; skipping pop_rate projection.")
        return obs_df
    future_years = list(range(last_obs_year + 1, end_year + 1))
    def multiplier(y: int) -> float:
        if pr.anchors:
            return _interp_scalar_by_year(y, pr.anchors)
        if pr.annual_rate_growth is not None:
            dy = y - last_obs_year
            return float((1.0 + pr.annual_rate_growth) ** dy)
        return 1.0
    bands = _age_band_list(obs_df, open_width)
    for y in future_years:
        popy = pop_df[pop_df[ycol] == y]
        if popy.empty:
            avail = pop_df[ycol].unique()
            if len(avail) == 0:
                continue
            closest = int(avail[np.argmin(np.abs(avail - y))])
            popy = pop_df[pop_df[ycol] == closest]
        pop_by_age = popy.set_index(acol)[pcol].to_dict()
        band_exp = []
        for band in bands:
            lo, hi = parse_age_band(band, open_width)
            exp = 0.0
            for a in range(lo, hi):
                rate_a = base_rate.get(a, 0.0) * multiplier(y)
                exp += rate_a * float(pop_by_age.get(a, 0.0))
            band_exp.append(max(0.0, exp))
        total = int(round(sum(band_exp)))
        alloc = _largest_remainder_split(total, np.array(band_exp, float))
        for band, n in zip(bands, alloc):
            if n > 0:
                out_rows.append(pd.DataFrame({
                    "year": [y], "age_band": [band], "sex": ["U"],
                    "count": [int(n)], "src": ["pop_rate"]
                }))
    return pd.concat(out_rows, ignore_index=True)

def _linear_or_constant(obs_df: pd.DataFrame,
                        end_year: int,
                        window: int,
                        method: str = "linear",
                        const_from_year: Optional[int] = None,
                        const_value: Optional[int] = None) -> pd.DataFrame:
    df = obs_df.copy()
    df["src"] = "observed"
    last_year = int(df.year.max())
    observed_totals = df.groupby("year")["count"].sum().sort_index()
    if method == "constant":
        if const_from_year is None or const_value is None:
            raise ValueError("constant method requires from_year and value")
        pred_years = np.arange(max(last_year + 1, const_from_year), end_year + 1)
        preds = np.full_like(pred_years, fill_value=int(const_value), dtype=int)
    else:
        totals = observed_totals.tail(window)
        if len(totals) < 1:
            slope, intercept = 0.0, float(observed_totals.iloc[-1])
        else:
            slope, intercept = np.polyfit(totals.index, totals.values, 1)
        pred_years = np.arange(last_year + 1, end_year + 1)
        preds = (slope * pred_years + intercept).clip(min=0).round().astype(int)
    synth = pd.DataFrame({
        "year": pred_years.repeat(1),
        "age_band": "75-79",
        "sex": "U",
        "count": preds,
        "src": "extrap"
    })
    return pd.concat([df, synth], ignore_index=True)

def _age_band_shares_for_year(pop_df: pd.DataFrame, year: int,
                              bands: list[str], open_width: int,
                              year_col: str, age_col: str, pop_col: str) -> np.ndarray:
    sub = pop_df[pop_df[year_col] == year]
    if sub.empty:
        avail = pop_df[year_col].unique()
        if len(avail) == 0:
            return np.ones(len(bands))
        closest = int(avail[np.argmin(np.abs(avail - year))])
        sub = pop_df[pop_df[year_col] == closest]
    shares = []
    for band in bands:
        lo, hi = parse_age_band(band, open_width)
        m = (sub[age_col] >= lo) & (sub[age_col] < hi)
        shares.append(sub.loc[m, pop_col].sum())
    return np.asarray(shares, dtype=float)

def redistribute_future_by_population(df: pd.DataFrame,
                                      pop_df: pd.DataFrame,
                                      open_width: int,
                                      year_col: str, age_col: str, pop_col: str) -> pd.DataFrame:
    if "src" not in df.columns:
        return df
    obs_bands = (df.loc[df["src"]=="observed", "age_band"]
                   .dropna().astype(str).unique().tolist())
    obs_bands.sort(key=lambda s: parse_age_band(s, open_width)[0])
    future = df[df["src"]=="extrap"].copy()
    keep   = df[df["src"]=="observed"].copy()
    rows = []
    for y, grp in future.groupby("year"):
        total = int(grp["count"].sum())
        shares = _age_band_shares_for_year(pop_df, int(y), obs_bands,
                                           open_width, year_col, age_col, pop_col)
        alloc = _largest_remainder_split(total, shares)
        for band, n in zip(obs_bands, alloc):
            if n > 0:
                rows.append({"year": int(y), "age_band": band, "sex": "U",
                             "count": int(n), "src": "extrap"})
    expanded = pd.DataFrame(rows, columns=["year","age_band","sex","count","src"])
    return pd.concat([keep, expanded], ignore_index=True)

# =============================================================================
# Redo‑SAVR loader (unchanged)
# =============================================================================

def _read_redo_savr_csv_to_year_totals(path: Path) -> Dict[int, int]:
    def _agg(df: pd.DataFrame) -> Dict[int, int]:
        cols = {c.lower(): c for c in df.columns}
        def has(*names): return all(n in cols for n in names)
        if has("year", "count"):
            ycol, ccol = cols["year"], cols["count"]
            tmp = df[[ycol, ccol]].copy()
            tmp[ccol] = pd.to_numeric(tmp[ccol], errors="coerce").fillna(0).astype(int)
            return tmp.groupby(ycol)[ccol].sum().astype(int).to_dict()
        elif has("sex", "age_band", "year", "count"):
            ycol, ccol = cols["year"], cols["count"]
            tmp = df[[ycol, ccol]].copy()
            tmp[ccol] = pd.to_numeric(tmp[ccol], errors="coerce").fillna(0).astype(int)
            return tmp.groupby(ycol)[ccol].sum().astype(int).to_dict()
        return {}
    try:
        df = pd.read_csv(path)
        out = _agg(df)
        if out:
            return out
    except Exception:
        pass
    try:
        df = pd.read_csv(path, header=None, names=["sex","age_band","year","count"])
        out = _agg(df)
        if out:
            return out
    except Exception:
        pass
    log.warning("Could not parse redo-SAVR CSV at %s; no targets loaded.", path)
    return {}

# =============================================================================
# Monte‑Carlo simulator (unchanged vs v7)
# =============================================================================

class DistFactory:
    def __init__(self, modes: Dict[str, DurabilityMode] | DurabilityMode):
        if isinstance(modes, DurabilityMode):
            modes = {"_single": modes}
        self._dists, self._w = [], []
        for m in modes.values():
            self._dists.append(norm(loc=m.mean, scale=m.sd))
            self._w.append(m.weight)
        self._w = np.asarray(self._w, float)
        self._w /= self._w.sum()

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        choice = rng.choice(len(self._dists), p=self._w, size=n)
        out = np.empty(n)
        for i, dist in enumerate(self._dists):
            mask = choice == i
            if mask.any():
                out[mask] = dist.rvs(mask.sum(), random_state=rng)
        return out.clip(min=0.1)

class ViVSimulator:
    def __init__(self, cfg: Config, dirs: Dirs):
        self.cfg   = cfg
        self.dirs  = dirs
        self.rng   = np.random.default_rng(cfg.simulation.get("rng_seed"))
        self.open_width      = cfg.open_age_bin_width
        self.start_forecast  = cfg.years["simulate_from"]
        self.end_year        = cfg.years["end"]
        self.n_cat           = cfg.risk_model["categories"]
        self.min_dur         = float(cfg.simulation.get("min_dur_years", 1.0))

        tavi_obs = pd.read_csv(cfg.procedure_counts["tavi"])
        savr_obs = pd.read_csv(cfg.procedure_counts["savr"])

        try:
            shutil.copy2(cfg.procedure_counts["tavi"], dirs.inputs_snapshot / "registry_tavi.csv")
            shutil.copy2(cfg.procedure_counts["savr"], dirs.inputs_snapshot / "registry_savr.csv")
        except Exception:
            pass

        pop_df = None
        pop_cfg = cfg.population_projection
        if pop_cfg and Path(str(pop_cfg.get("path",""))).exists():
            pop_df = pd.read_csv(pop_cfg["path"])
            self.pop_cols = (pop_cfg["year_col"], pop_cfg["age_col"], pop_cfg["pop_col"])
        else:
            self.pop_cols = ("year","age","population")
            log.warning("Population projection not found; population-based features limited.")

        ip = cfg.index_projection
        assert ip is not None
        if ip.tavi and ip.tavi.method == "population_rate" and pop_df is not None:
            self.tavi_df = _project_index_by_pop_rate(
                tavi_obs, self.end_year, pop_df, self.open_width, *self.pop_cols,
                ip.tavi.pop_rate or PopRateSettings()
            )
        else:
            self.tavi_df = _linear_or_constant(
                tavi_obs, self.end_year, ip.tavi.window if ip.tavi else 5,
                method=(ip.tavi.method if ip.tavi else "linear")
            )
            if pop_df is not None:
                self.tavi_df = redistribute_future_by_population(
                    self.tavi_df, pop_df, self.open_width, *self.pop_cols
                )
        if ip.savr and ip.savr.method == "population_rate" and pop_df is not None:
            self.savr_df = _project_index_by_pop_rate(
                savr_obs, self.end_year, pop_df, self.open_width, *self.pop_cols,
                ip.savr.pop_rate or PopRateSettings()
            )
        else:
            method = ip.savr.method if ip.savr else "constant"
            const_from = ip.savr.constant.from_year if (ip.savr and ip.savr.constant) else 2025
            const_val  = ip.savr.constant.value if (ip.savr and ip.savr.constant) else 2000
            self.savr_df = _linear_or_constant(
                savr_obs, self.end_year, ip.savr.window if ip.savr else 5,
                method=method, const_from_year=const_from, const_value=const_val
            )
            if pop_df is not None:
                self.savr_df = redistribute_future_by_population(
                    self.savr_df, pop_df, self.open_width, *self.pop_cols
                )

        self.tavi_df.to_csv(dirs.tables_index / "tavi_with_projection.csv", index=False)
        self.savr_df.to_csv(dirs.tables_index / "savr_with_projection.csv", index=False)

        log.info("TAVI rows %d, SAVR rows %d (obs+projection)", len(self.tavi_df), len(self.savr_df))

        dur = cfg.durability
        if "savr_bioprosthetic" in dur:
            self.dur_tavi       = DistFactory(dur["tavi"])
            self.dur_savr_lt70  = DistFactory(dur["savr_bioprosthetic"]["lt70"])
            self.dur_savr_gte70 = DistFactory(dur["savr_bioprosthetic"]["gte70"])
        else:
            self.dur_tavi       = DistFactory(dur["tavi"])
            self.dur_savr_lt70  = DistFactory(dur["savr_lt70"])
            self.dur_savr_gte70 = DistFactory(dur["savr_gte70"])

        s = cfg.survival_curves
        if self.n_cat == 2:
            self.surv_low = norm(loc=s["low_risk"]["median"], scale=s["low_risk"]["sd"])
            self.surv_ih  = norm(loc=s["int_high_risk"]["median"], scale=s["int_high_risk"]["sd"])
        else:
            self.surv_low  = norm(loc=s["low"]["median"],  scale=s["low"]["sd"])
            self.surv_int  = norm(loc=s["intermediate"]["median"], scale=s["intermediate"]["sd"])
            self.surv_high = norm(loc=s["high"]["median"], scale=s["high"]["sd"])

        self.redo_targets = self._load_redo_savr_numbers()

    def _load_redo_savr_numbers(self) -> Dict[int, int]:
        self.redo_mode = "replace_rates"
        self.rr_savr_cfg = float(self.cfg.redo_rates.get("savr_after_savr", 0.0))
        self.rr_tavi_cfg = float(self.cfg.redo_rates.get("savr_after_tavi", 0.0))
        out: Dict[int, int] = {}

        rs = self.cfg.redo_savr_numbers or {}
        if isinstance(rs.get("values"), dict):
            out = {int(k): int(v) for k, v in rs["values"].items()}
        elif rs.get("path"):
            p = Path(str(rs["path"]))
            if p.exists():
                out = _read_redo_savr_csv_to_year_totals(p)
                try:
                    shutil.copy2(p, self.dirs.inputs_snapshot / "redo_savr.csv")
                except Exception:
                    pass
        if isinstance(rs, dict) and rs.get("mode"):
            self.redo_mode = str(rs["mode"]).strip().lower()

        if not out and "redo_savr" in self.cfg.procedure_counts:
            p = Path(str(self.cfg.procedure_counts["redo_savr"]))
            if p.exists():
                out = _read_redo_savr_csv_to_year_totals(p)
                try:
                    shutil.copy2(p, self.dirs.inputs_snapshot / "redo_savr.csv")
                except Exception:
                    pass
                
        self.use_redo_rates = not (out and self.redo_mode == "replace_rates")
        if out and not self.use_redo_rates:
            if (self.rr_savr_cfg > 0) or (self.rr_tavi_cfg > 0):
                log.info("Absolute redo targets present (mode=replace_rates); ignoring redo_rates during simulation.")

        rs_fm = (self.cfg.redo_savr_numbers or {}).get("fill_missing", {}) if isinstance(self.cfg.redo_savr_numbers, dict) else {}
        method = str(rs_fm.get("method", "zero")).lower()
        y_lo   = int(self.cfg.years["simulate_from"])
        y_hi   = int(self.cfg.years["end"])
        yrange = rs_fm.get("range", [y_lo, y_hi])
        try:
            y_lo, y_hi = int(yrange[0]), int(yrange[1])
        except Exception:
            y_lo, y_hi = self.cfg.years["simulate_from"], self.cfg.years["end"]

        if out and method in ("forward", "linear"):
            years = list(range(y_lo, y_hi+1))
            anchors = sorted(out.items())
            if not anchors:
                return out
            if method == "forward":
                filled = {}
                last = 0
                first_year, first_val = anchors[0]
                for y in years:
                    if y in out:
                        last = out[y]
                    elif y < first_year:
                        last = 0
                    filled[y] = last
                out = filled
            else:
                xs = [k for k, _ in anchors]; ys = [v for _, v in anchors]
                arr = np.interp(years, xs, ys, left=ys[0], right=ys[-1])
                out = {y: int(round(v)) for y, v in zip(years, arr)}

        self.redo_targets_filled = out.copy()
        return out

    def _risk_mix(self, proc_type: str, year: int) -> Dict[str, float]:
        if proc_type == "savr":
            return self.cfg.risk_mix["savr"]
        for k, v in self.cfg.risk_mix["tavi"].items():
            a, b = (int(x) for x in k.split('-'))
            if a <= year <= b:
                return v
        last_key = max(self.cfg.risk_mix["tavi"].keys(), key=lambda s: int(s.split('-')[1]))
        return self.cfg.risk_mix["tavi"][last_key]

    def _sample_survival(self, tag: str, ages: np.ndarray) -> np.ndarray:
        if tag == "low":
            base = self.surv_low.rvs(len(ages), random_state=self.rng)
        elif tag in ("ih", "int", "intermediate"):
            base = self.surv_ih.rvs(len(ages), random_state=self.rng) if self.n_cat == 2 \
                   else self.surv_int.rvs(len(ages), random_state=self.rng)
        else:
            base = self.surv_high.rvs(len(ages), random_state=self.rng)
        if self.cfg.risk_model.get("use_age_hazard") and self.cfg.age_hazard:
            key_map = {"low":"low", "ih":"intermediate", "int":"intermediate",
                       "intermediate":"intermediate", "high":"high"}
            k = key_map.get(tag, tag)
            hr_per5 = float(self.cfg.age_hazard["hr_per5"].get(k, 1.0))
            ref_age = float(self.cfg.age_hazard["ref_age"])
            base = base / (hr_per5 ** ((ages - ref_age) / 5.0))
        return np.maximum(0.1, base)

    # --- MODIFY _pen to apply per-run jitter factor (if present) ---
    def _pen(self, vtype: str, year: int) -> float:
        anchors = self.cfg.penetration.get(vtype, {})
        base = 1.0 if not anchors else _interp_scalar_by_year(year, anchors)
        factor = float(getattr(self, "_pen_run_factor", 1.0))
        return float(np.clip(base * factor, 0.0, 1.0))

    def run_once(self, run_id: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        

        # --- at the very top of run_once(...) ---
        jit = (self.cfg.simulation.get("jitter", {}) if isinstance(self.cfg.simulation, dict) else {})
        dur_sd = max(0.0, float(jit.get("durability_pct_sd", 0.0))) / 100.0
        sur_sd = max(0.0, float(jit.get("survival_pct_sd", 0.0))) / 100.0
        pen_sd = max(0.0, float(jit.get("penetration_pct_sd", 0.0))) / 100.0
        self._pen_run_factor = float(np.clip(self.rng.normal(1.0, pen_sd), 0.0, 2.0))


        edges  = self.cfg.age_bins
        labels = _age_labels(edges)

        candidates: list[tuple[int,str,str,str]] = []
        realized:   list[tuple[int,str]] = []
        redo_log:   list[tuple[int,str]] = []

        idx_cnt  = defaultdict(int)
        deaths   = defaultdict(int)
        fails    = defaultdict(int)
        viable   = defaultdict(int)

        rr_savr = float(self.cfg.redo_rates.get("savr_after_savr", 0.0)) if self.redo_mode != "replace_rates" else 0.0
        rr_tavi = float(self.cfg.redo_rates.get("savr_after_tavi", 0.0)) if self.redo_mode != "replace_rates" else 0.0

        for proc_type, df in (("tavi", self.tavi_df), ("savr", self.savr_df)):
            for _, row in df.iterrows():
                year = int(row["year"])
                n    = int(row["count"])
                idx_cnt[(year, proc_type)] += n

                lo, hi = parse_age_band(row["age_band"], self.open_width)
                ages   = np.random.default_rng(self.rng).integers(lo, hi, size=n)

                mix = self._risk_mix(proc_type, year)
                if self.n_cat == 2:
                    low_mask = np.random.default_rng(self.rng).random(n) < mix["low"]
                    risk = np.where(low_mask, "low", "ih")
                else:
                    rnd  = np.random.default_rng(self.rng).random(n)
                    risk = np.where(rnd < mix["low"], "low",
                                    np.where(rnd < mix["low"] + mix["intermediate"], "int", "high"))

                surv = np.empty(n)
                # after creating `surv`:
                if sur_sd > 0:
                    surv *= np.clip(self.rng.normal(1.0, sur_sd, size=n), 0.2, 5.0)


                for tag in ("low", "ih" if self.n_cat==2 else "int", "high"):
                    m = (risk == tag)
                    if m.any():
                        surv[m] = self._sample_survival(tag, ages[m])

                dur = (DistFactory(self.cfg.durability["tavi"]).sample(n, self.rng) if proc_type=="tavi"
                       else np.where(ages < 70, DistFactory(self.cfg.durability["savr_bioprosthetic"]["lt70"]).sample(n, self.rng),
                                     DistFactory(self.cfg.durability["savr_bioprosthetic"]["gte70"]).sample(n, self.rng)))

                # after creating `dur`:
                if dur_sd > 0:
                    dur *= np.clip(self.rng.normal(1.0, dur_sd, size=n), 0.2, 5.0)                


                dur = np.maximum(dur, float(self.cfg.simulation.get("min_dur_years", 1.0)))
                fail_y  = year + np.floor(dur).astype(int)
                death_y = year + np.floor(surv).astype(int)

                for y in np.unique(death_y):
                    deaths[(y, proc_type)] += int(np.sum(death_y == y))
                for y in np.unique(fail_y):
                    fails[(y, proc_type)]  += int(np.sum(fail_y == y))

                valid = (fail_y <= death_y) & \
                        (self.start_forecast <= fail_y) & (fail_y <= self.end_year)
                if not valid.any():
                    continue

                vtype = "tavi_in_tavi" if proc_type=="tavi" else "tavi_in_savr"
                for fy, rk, age in zip(fail_y[valid], risk[valid], ages[valid]):
                    viable[(int(fy), proc_type)] += 1
                    ai = int(np.digitize(age, edges, right=False))
                    candidates.append((int(fy), vtype, rk, labels[ai]))

                    redo_p = rr_tavi if vtype == "tavi_in_tavi" else rr_savr
                    if np.random.default_rng(self.rng).random() < redo_p:
                        redo_log.append((int(fy), vtype))
                        continue
                    if np.random.default_rng(self.rng).random() < self._pen(vtype, int(fy)):
                        realized.append((int(fy), vtype))

        cand_df = (pd.DataFrame(candidates, columns=["year","viv_type","risk","age_bin"])
                   .value_counts().rename("count").reset_index())
        real_df = (pd.DataFrame(realized, columns=["year","viv_type"])
                   .value_counts().rename("count").reset_index())
        redo_df = (pd.DataFrame(redo_log, columns=["year","source"])
                   .value_counts().rename("count").reset_index())

        years = range(self.start_forecast, self.end_year + 1)
        rows  = []
        for y in years:
            for p in ("tavi","savr"):
                rows.append([y, p, idx_cnt[(y,p)], deaths[(y,p)], fails[(y,p)], viable[(y,p)]])
        flow_df = pd.DataFrame(rows, columns=["year","proc","index_cnt","deaths","failures","viable"])

        return cand_df, real_df, flow_df

# =============================================================================
# Plotting helpers (v7)
# =============================================================================






def _plot_flow(stats: pd.DataFrame, proc: str, out_path: Path, cfg=None) -> None:
    sub = stats[stats.proc == proc].set_index("year").sort_index()
    plt.figure()
    plt.plot(sub.index, sub.index_cnt,   label="Index procedures")
    plt.plot(sub.index, sub.deaths,      label="Deaths")
    plt.plot(sub.index, sub.failures,    label="Valve failures")
    plt.plot(sub.index, sub.viable,      label="Viable ViV candidates")
    ax = plt.gca()
    _apply_xrange_and_focus(ax, cfg)
    plt.title(f"Patient flow – {proc.upper()}")
    plt.xlabel("Calendar year"); plt.ylabel("Head-count")
    plt.legend(); plt.tight_layout()
    _savefig_current(out_path)


def _plot_index_series(df: pd.DataFrame, title: str, out_path: Path,
                       observed_cutoff_year: Optional[int], cfg=None) -> None:
    tot = (df.groupby(["year","src"])["count"].sum().unstack(fill_value=0).sort_index())
    plt.figure()
    if "observed" in tot.columns:
        obs = tot["observed"]
        if observed_cutoff_year is not None:
            obs = obs.loc[obs.index <= observed_cutoff_year]
        if len(obs):
            plt.plot(obs.index, obs.values, marker="o", label="Observed")
            plt.axvline(int(obs.index.max()), color="k", linestyle=":", linewidth=1)
            plt.axvspan(int(obs.index.max())+0.05, tot.index.max()+0.5, alpha=0.08, color="gray")
    proj_col = "pop_rate" if "pop_rate" in tot.columns else ("extrap" if "extrap" in tot.columns else None)
    if proj_col is not None:
        proj = tot[proj_col]
        plt.plot(proj.index, proj.values, "--", label="Projection", color="tab:orange")
    ax = plt.gca()
    _apply_xrange_and_focus(ax, cfg)
    plt.title(title); plt.xlabel("Year"); plt.ylabel("Procedures / yr")
    plt.legend(); plt.tight_layout()
    _savefig_current(out_path)
    

def _plot_viv_pretty(realized_summary: pd.DataFrame,
                     year_lo: int, year_hi: int,
                     out_path: Path,
                     bar_color: Optional[str] = None,
                     label_bars: bool = True,
                     cfg=None) -> None:   
    sub = realized_summary[(realized_summary.year>=year_lo) & (realized_summary.year<=year_hi)]
    wide = sub.pivot(index="year", columns="viv_type", values="mean").fillna(0.0)
    for c in ("tavi_in_savr","tavi_in_tavi"):
        if c not in wide.columns:
            wide[c] = 0.0
    wide["total"] = wide["tavi_in_savr"] + wide["tavi_in_tavi"]

    fig, ax = plt.subplots(figsize=(10,5), dpi=140)
    bar_color_final = bar_color or "#d0d0d0"
    bars = ax.bar(wide.index, wide["total"], label="Total ViV (realized)",
                  color=bar_color_final, alpha=0.5, zorder=1)
    line_savr, = ax.plot(wide.index, wide["tavi_in_savr"], marker="o", label="TAVR-in-SAVR",
                         color="red", alpha=0.85, linewidth=2, zorder=3)
    line_tavi, = ax.plot(wide.index, wide["tavi_in_tavi"], marker="s", label="TAVR-in-TAVR",
                         color="blue", alpha=0.85, linewidth=2, zorder=3)

    max_val = float(wide[["tavi_in_savr","tavi_in_tavi","total"]].to_numpy().max()) if len(wide) else 0.0
    y_off_line = 0.015 * max_val
    y_off_bar  = 0.020 * max_val

    for x, y in zip(wide.index, wide["tavi_in_savr"]):
        ax.text(x, y - y_off_line, f"{int(round(y))}", ha="center", va="top",
                fontsize=11, fontweight="bold", color=line_savr.get_color(), zorder=4)
    for x, y in zip(wide.index, wide["tavi_in_tavi"]):
        ax.text(x, y + y_off_line, f"{int(round(y))}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=line_tavi.get_color(), zorder=4)
    if label_bars:
        for rect in bars:
            h = rect.get_height()
            x = rect.get_x() + rect.get_width()/2.0
            ax.text(x, h + y_off_bar, f"{int(round(h))}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=bar_color_final, zorder=4)

    ax.set_title(f"Predicted ViV volume ({year_lo}–{year_hi})")
    ax.set_xlabel("Year"); ax.set_ylabel("Procedures / yr")
    ax.legend(); fig.tight_layout()
    

    # In _plot_viv_pretty(...) just before fig.savefig:
    ax = plt.gca()
    if " _apply_xrange_and_focus" in globals():  # safe if helper isn't present yet
        _apply_xrange_and_focus(ax, cfg)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)



def _plot_viv_pretty_with_overlay(pre_summary: pd.DataFrame,
                                  post_summary: pd.DataFrame,
                                  year_lo: int, year_hi: int,
                                  out_path: Path,
                                  bar_color: Optional[str],
                                  label_bars: bool,
                                  cfg=None) -> None: 
    def _wide(df):
        sub = df[(df.year>=year_lo)&(df.year<=year_hi)]
        w = sub.pivot(index="year", columns="viv_type", values="mean").fillna(0.0)
        for c in ("tavi_in_savr","tavi_in_tavi"):
            if c not in w.columns: w[c] = 0.0
        w["total"] = w.sum(axis=1)
        return w

    pre = _wide(pre_summary)
    post = _wide(post_summary)

    fig, ax = plt.subplots(figsize=(10,5), dpi=140)
    bar_color_final = bar_color or "#d0d0d0"

    bars = ax.bar(post.index, post["total"], label="Total ViV (post)", color=bar_color_final, alpha=0.5, zorder=1)
    l1, = ax.plot(post.index, post["tavi_in_savr"], marker="o", label="TAVR-in-SAVR (post)",
                  color="red", alpha=0.9, linewidth=2, zorder=3)
    l2, = ax.plot(post.index, post["tavi_in_tavi"], marker="s", label="TAVR-in-TAVR (post)",
                  color="blue", alpha=0.9, linewidth=2, zorder=3)

    ax.plot(pre.index, pre["tavi_in_savr"], linestyle="--", marker=None, color="red", alpha=0.7, label="... pre (SAVR)")
    ax.plot(pre.index, pre["tavi_in_tavi"], linestyle="--", marker=None, color="blue", alpha=0.7, label="... pre (TAVR)")

    max_val = float(np.nanmax([pre.values.max() if len(pre) else 0, post.values.max() if len(post) else 0])) if (len(pre) or len(post)) else 0.0
    y_off_line = 0.015 * max_val
    y_off_bar  = 0.020 * max_val

    for x, y in zip(post.index, post["tavi_in_savr"]):
        ax.text(x, y - y_off_line, f"{int(round(y))}", ha="center", va="top", fontsize=11, fontweight="bold", color="red", zorder=4)
    for x, y in zip(post.index, post["tavi_in_tavi"]):
        ax.text(x, y + y_off_line, f"{int(round(y))}", ha="center", va="bottom", fontsize=11, fontweight="bold", color="blue", zorder=4)
    if label_bars:
        for rect in bars:
            h = rect.get_height(); x = rect.get_x() + rect.get_width()/2.0
            ax.text(x, h + y_off_bar, f"{int(round(h))}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=bar_color_final, zorder=4)

    ax.set_title(f"Predicted ViV volume (pre vs post, {year_lo}–{year_hi})")
    ax.set_xlabel("Year"); ax.set_ylabel("Procedures / yr")
    ax.legend(); fig.tight_layout()
    
    # In _plot_viv_pretty_with_overlay(...) just before fig.savefig:
    ax = plt.gca()
    if " _apply_xrange_and_focus" in globals():  # safe if helper isn't present yet
        _apply_xrange_and_focus(ax, cfg)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

# --- REPLACE _save_viv_qc_plots with this CI-enabled version ---
def _save_viv_qc_plots(detail: pd.DataFrame, summary: pd.DataFrame,
                       out_dir: Path, age_edges: list[int], cfg=None) -> None:
    """
    • detail  : year x viv_type x risk x age_bin x mean (unused for the lines)
    • summary : year x viv_type x mean [+ sd if available]
    Saves 'viv_forecast_total.csv' and a line plot with optional 95% CI ribbons.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_mean = (summary.pivot(index="year", columns="viv_type", values="mean").fillna(0))
    # bring sd if present
    has_sd = "sd" in summary.columns
    wide_sd = (summary.pivot(index="year", columns="viv_type", values="sd").fillna(0)) if has_sd else None

    preferred = ["tavi_in_savr", "tavi_in_tavi"]
    cols = [c for c in preferred if c in wide_mean.columns] + \
           [c for c in wide_mean.columns if c not in preferred]
    wide_mean = wide_mean[cols]
    wide_mean["total"] = wide_mean.sum(axis=1)

    wide_mean.to_csv(out_dir / "viv_forecast_total.csv", index=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    line_handles = {}

    # CI ribbons for each component (if sd provided)
    if has_sd:
        z = 1.96
        for col in cols:
            mu = wide_mean[col].values
            sd = wide_sd[col].values if col in wide_sd.columns else np.zeros_like(mu)
            lo = np.maximum(0.0, mu - z*sd)
            hi = mu + z*sd
            ax.fill_between(wide_mean.index, lo, hi, alpha=0.15, label=None)

    # Lines
    def _marker_for(col: str) -> str:
        return {"tavi_in_savr": "o", "tavi_in_tavi": "s", "total": "D"}.get(col, "o")
    for col in list(cols) + ["total"]:
        h, = ax.plot(wide_mean.index, wide_mean[col].values,
                     label=col.replace("_", " "),
                     marker=_marker_for(col))
        line_handles[col] = h

    # Labels above points
    max_val = float(wide_mean.to_numpy().max()) if not wide_mean.empty else 0.0
    y_offset = 0.015 * max_val if max_val > 0 else 0.5
    for col, h in line_handles.items():
        color = h.get_color()
        ys = wide_mean[col].values
        for x, y in zip(wide_mean.index, ys):
            if pd.isna(y): continue
            ax.text(x, y + y_offset, f"{int(round(y))}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color=color)

    _apply_xrange_and_focus(ax, cfg)
    ax.set_title("Predicted ViV-TAVI volumes")
    ax.set_xlabel("Calendar year"); ax.set_ylabel("Procedures / yr")
    ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    if max_val > 0:
        ax.set_ylim(top=max_val * 1.12)
    fig.tight_layout(); ax.legend()
    fig.savefig(out_dir / "viv_forecast.png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    

# --- Baseline risk score builder (TAVI/SAVR/redo-SAVR) ----------------

def build_baseline_risk_scores(
    tavi_csv: Path,
    savr_csv: Path,
    redo_csv: Optional[Path],
    age_pop_band_sex_csv: Path,
    risk_years: list[int],
    open_width: int,
    out_csv: Path
) -> Path:
    """
    Uses:
      • age_pop_band_sex_csv: age_projections_by_year_band_sex.csv
          (year, sex, age_band, population)
      • tavi_csv/savr_csv/redo_csv: index volumes with columns at least
          (year, sex, age_band, count)
      • risk_years: e.g. [2023, 2024]

    Outputs:
      • out_csv: baseline risk scores by (procedure, year, sex, age_band)
    """
    pop = pd.read_csv(age_pop_band_sex_csv)
    pop["sex"] = pop["sex"].astype(str)
    pop["age_band"] = pop["age_band"].astype(str)

    def _load_proc(path: Path, proc_name: str) -> pd.DataFrame:
        if path is None or not path.exists():
            return pd.DataFrame(columns=["year","sex","age_band","count","procedure"])
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        # robustly map to year/sex/age_band/count
        ycol = cols.get("year")
        scol = cols.get("sex")
        acol = cols.get("age_band")
        ccol = cols.get("count")
        if not all([ycol, scol, acol, ccol]):
            raise ValueError(f"{proc_name} file {path} is missing one of year/sex/age_band/count")
        df = df[[ycol, scol, acol, ccol]].copy()
        df.columns = ["year","sex","age_band","count"]
        df["procedure"] = proc_name
        return df

    tavi = _load_proc(tavi_csv, "tavi")
    savr = _load_proc(savr_csv, "savr")
    redo = _load_proc(redo_csv, "redo_savr") if (redo_csv and redo_csv.exists()) else \
           pd.DataFrame(columns=["year","sex","age_band","count","procedure"])

    all_proc = pd.concat([tavi, savr, redo], ignore_index=True)
    all_proc["age_band"] = all_proc["age_band"].astype(str)
    all_proc["sex"] = all_proc["sex"].astype(str)

    rows = []
    for y in risk_years:
        pop_y = pop[pop["year"] == y]
        if pop_y.empty:
            continue
        for sex in ["Men","Women"]:
            pop_sex = pop_y[pop_y["sex"] == sex]
            if pop_sex.empty:
                continue
            # population by age_band and sex
            pop_by_band = pop_sex.groupby("age_band")["population"].sum()
            proc_y_sex = all_proc[(all_proc["year"] == y) & (all_proc["sex"] == sex)]
            if proc_y_sex.empty:
                continue
            for proc_name, g_proc in proc_y_sex.groupby("procedure"):
                for band, g_band in g_proc.groupby("age_band"):
                    # Handle ≥80 registry band
                    if band in ("≥80", ">=80", "80+"):
                        pop_den = 0.0
                        if "80-84" in pop_by_band.index:
                            pop_den += pop_by_band["80-84"]
                        if "≥85" in pop_by_band.index:
                            pop_den += pop_by_band["≥85"]
                    else:
                        pop_den = float(pop_by_band.get(band, 0.0))

                    cnt = float(g_band["count"].sum())
                    risk = cnt / pop_den if pop_den > 0 else 0.0
                    rows.append({
                        "procedure": proc_name,
                        "year": int(y),
                        "sex": sex,
                        "age_band": band,
                        "count": int(cnt),
                        "population": int(round(pop_den)),
                        "risk": risk,
                    })

    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_csv


# =============================================================================
# Top-level driver
# =============================================================================

def run_simulation(cfg: Config, dirs: Dirs):
    sim = ViVSimulator(cfg, dirs)

    tavi_last_obs = int(sim.tavi_df[sim.tavi_df["src"]=="observed"]["year"].max()) if not sim.tavi_df.empty else None
    savr_last_obs = int(sim.savr_df[sim.savr_df["src"]=="observed"]["year"].max()) if not sim.savr_df.empty else None
    _plot_index_series(sim.tavi_df, "TAVI index (observed + projection)",
                       dirs.fig_index / "image_A_tavi_index.png", tavi_last_obs)
    _plot_index_series(sim.savr_df, "SAVR index (observed + projection)",
                       dirs.fig_index / "image_B_savr_index.png", savr_last_obs)

    tot_tavi = sim.tavi_df.groupby("year")["count"].sum()
    tot_savr = sim.savr_df.groupby("year")["count"].sum()
    plt.figure()
    plt.plot(tot_tavi.index, tot_tavi.values, label="TAVI", marker="o")
    plt.plot(tot_savr.index, tot_savr.values, label="SAVR", marker="s")
    plt.xlabel("Year"); plt.ylabel("Procedures / yr")
    plt.title("Observed + projected index procedures")
    plt.tight_layout(); plt.legend()
    _savefig_current(dirs.fig_index / "index_volume_overlay.png")

    cand_runs, real_runs, flow_runs = [], [], []
    for i in range(cfg.simulation["n_runs"]):




        cand, real, flow = sim.run_once(i)
        cand["run"] = i; real["run"] = i; flow["run"] = i
        

        # --- inside run_simulation, in the for i in range(cfg.simulation["n_runs"]) loop ---
        if cfg.simulation.get("verbose", False):
            if (i % int(cfg.simulation.get("progress_every", 10)) == 0) or (i == cfg.simulation["n_runs"] - 1):
                totals = real.groupby("viv_type")["count"].sum() if len(real) else {}
                log.info("Run %d: realized this run — TAVR-in-SAVR=%s, TAVR-in-TAVR=%s",
                        i, int(totals.get("tavi_in_savr", 0)), int(totals.get("tavi_in_tavi", 0)))




        cand_runs.append(cand); real_runs.append(real); flow_runs.append(flow)

    cand_all = pd.concat(cand_runs, ignore_index=True) if cand_runs else pd.DataFrame(columns=["year","viv_type","risk","age_bin","count","run"])
    real_all = pd.concat(real_runs, ignore_index=True) if real_runs else pd.DataFrame(columns=["year","viv_type","count","run"])
    flow_all = pd.concat(flow_runs, ignore_index=True) if flow_runs else pd.DataFrame(columns=["year","proc","index_cnt","deaths","failures","viable","run"])

    cand_per_run = (cand_all.groupby(["run","year","viv_type"])["count"].sum().reset_index())
    real_per_run = (real_all.groupby(["run","year","viv_type"])["count"].sum().reset_index())

    cand_summary = (cand_per_run.groupby(["year","viv_type"]).agg(mean=("count","mean"), sd=("count","std")).reset_index())
    real_summary = (real_per_run.groupby(["year","viv_type"]).agg(mean=("count","mean"), sd=("count","std")).reset_index())

    cand_v5like = (cand_all.groupby(["year","viv_type"]).agg(mean=("count","mean"), sd=("count","std")).reset_index())

    dirs.tables_viv_per_run.mkdir(parents=True, exist_ok=True)
    cand_per_run.to_csv(dirs.tables_viv_per_run / "candidates_per_run.csv", index=False)
    real_per_run.to_csv(dirs.tables_viv_per_run / "realized_per_run.csv", index=False)
    cand_summary.to_csv(dirs.tables_viv / "viv_candidates_totals.csv", index=False)
    cand_v5like.to_csv(dirs.tables_viv / "viv_candidates_v5like.csv", index=False)
    real_summary.to_csv(dirs.tables_viv / "viv_forecast.csv", index=False)

    flow_mean = (flow_all.groupby(["year","proc"]).mean(numeric_only=True).reset_index())
    flow_mean.to_csv(dirs.tables_flow / "patient_flow_mean.csv", index=False)

    _plot_flow(flow_mean, "savr", dirs.fig_flow / "flow_savr.png", cfg)
    _plot_flow(flow_mean, "tavi", dirs.fig_flow / "flow_tavi.png", cfg)


    _plot_index_series(sim.tavi_df, "TAVI index (observed + projection)",
                    dirs.fig_index / "image_A_tavi_index.png", tavi_last_obs, cfg)
    _plot_index_series(sim.savr_df, "SAVR index (observed + projection)",
                    dirs.fig_index / "image_B_savr_index.png", savr_last_obs, cfg)
    
    # Stacked composition at snapshots
    snapshots = (cfg.figure_ranges.get("waterfall_years", [2025, 2030, 2035])
                if getattr(cfg, "figure_ranges", None) else [2025, 2030, 2035])
    _plot_index_stacked_by_band(sim.tavi_df, "tavi", snapshots,
                                dirs.fig_index / "tavi_index_stacked_snapshots.png", cfg)
    _plot_index_stacked_by_band(sim.savr_df, "savr", snapshots,
                                dirs.fig_index / "savr_index_stacked_snapshots.png", cfg)




    redo_targets = sim.redo_targets
    if redo_targets:
        realized_adj = real_summary.copy()
        realized_adj["realized"] = realized_adj["mean"]
        mask = realized_adj["viv_type"].eq("tavi_in_savr")

        tis_series = realized_adj[mask].set_index("year")["mean"]
        qc_rows, adj_vals = [], []
        for y, m in tis_series.items():
            target = redo_targets.get(int(y), None)
            if target is None:
                adj = m
                qc_rows.append([int(y), int(round(m)), None, int(round(adj))])
                adj_vals.append((y, adj))
                continue
            adj = max(0.0, m - float(target))
            qc_rows.append([int(y), int(round(m)), int(target), int(round(adj))])
            adj_vals.append((y, adj))
        adj_map = dict(adj_vals)
        realized_adj.loc[mask, "realized"] = realized_adj.loc[mask, "year"].map(adj_map)
        realized_adj.to_csv(dirs.tables_viv / "viv_forecast_realized.csv", index=False)
        pd.DataFrame(qc_rows, columns=["year","tis_before","redo_target","tis_after"])\
          .to_csv(dirs.qc_redo / "redo_savr_qc.csv", index=False)
    else:
        realized_adj = real_summary.copy()
        realized_adj["realized"] = realized_adj["mean"]
        realized_adj.to_csv(dirs.tables_viv / "viv_forecast_realized.csv", index=False)
        log.info("No redo-SAVR absolute targets; 'viv_forecast_realized.csv' mirrors realized means.")

    lines_dir_cand = dirs.fig_viv / "lines_candidates"
    lines_dir_pre  = dirs.fig_viv / "lines_pre"
    lines_dir_post = dirs.fig_viv / "lines_post"
    for _d in (lines_dir_cand, lines_dir_pre, lines_dir_post):
        _d.mkdir(parents=True, exist_ok=True)

    detail_stub = (cand_all.groupby(["year","viv_type","risk","age_bin"]).agg(mean=("count","mean")).reset_index())

    _save_viv_qc_plots(detail_stub, cand_summary[["year","viv_type","mean","sd"]],
                    dirs.fig_viv / "lines_candidates", cfg.age_bins, cfg)
    _save_viv_qc_plots(detail_stub, real_summary[["year","viv_type","mean","sd"]],
                    dirs.fig_viv / "lines_pre", cfg.age_bins, cfg)
    summary_post = realized_adj[["year","viv_type","realized"]].rename(columns={"realized":"mean"})
    # reuse pre-run SDs for ribbon shape if you want a ribbon on POST:
    summary_post = summary_post.merge(real_summary[["year","viv_type","sd"]], on=["year","viv_type"], how="left")
    _save_viv_qc_plots(detail_stub, summary_post,
                    dirs.fig_viv / "lines_post", cfg.age_bins, cfg)
    
    for y in (cfg.figure_ranges.get("waterfall_years", [2030]) if getattr(cfg, "figure_ranges", None) else [2030]):
        _plot_waterfall_for_year(flow_mean, realized_adj[["year","viv_type","realized"]].rename(columns={"realized":"mean"}),
                                sim.redo_targets_filled if hasattr(sim, "redo_targets_filled") else sim.redo_targets,
                                y, dirs.fig_flow)



    try:
        (dirs.tables_viv).mkdir(parents=True, exist_ok=True)
        cand_csv  = lines_dir_cand / "viv_forecast_total.csv"
        pre_csv   = lines_dir_pre  / "viv_forecast_total.csv"
        post_csv  = lines_dir_post / "viv_forecast_total.csv"
        if cand_csv.exists():
            shutil.copy2(cand_csv, dirs.tables_viv / "viv_candidates_total_v5style.csv")
        if pre_csv.exists():
            shutil.copy2(pre_csv,  dirs.tables_viv / "viv_realized_pre_total_lines.csv")
        if post_csv.exists():
            shutil.copy2(post_csv, dirs.tables_viv / "viv_realized_post_total_lines.csv")
    except Exception as e:
        log.warning("Could not copy v5-style totals CSVs to tables: %s", e)

    ylo, yhi = 2023, 2035
    if cfg.figure_ranges and cfg.figure_ranges.get("viv_years"):
        ylo, yhi = cfg.figure_ranges["viv_years"][0], cfg.figure_ranges["viv_years"][1]
    bar_col = cfg.plotting.viv_total_bar_color if cfg.plotting else None
    labels_on_bars = bool(cfg.plotting.label_bars) if cfg.plotting else True
    src = (cfg.plotting.image_c_source if cfg.plotting else "post").lower()

    if src == "pre":
        _plot_viv_pretty(real_summary, ylo, yhi,
                         dirs.fig_viv / f"image_C_viv_pretty_PRE_{ylo}_{yhi}.png",
                         bar_color=bar_col, label_bars=labels_on_bars)
    elif src == "both":
        _plot_viv_pretty_with_overlay(real_summary,
                                      realized_adj[["year","viv_type","realized"]].rename(columns={"realized":"mean"}),
                                      ylo, yhi,
                                      dirs.fig_viv / f"image_C_viv_pretty_PREvsPOST_{ylo}_{yhi}.png",
                                      bar_color=bar_col, label_bars=labels_on_bars)
    else:
        _plot_viv_pretty(
            realized_adj[["year","viv_type","realized"]].rename(columns={"realized":"mean"}),
            ylo, yhi,
            dirs.fig_viv / f"image_C_viv_pretty_{ylo}_{yhi}.png",
            bar_color=bar_col, label_bars=labels_on_bars
        )

    a, b = (cfg.figure_ranges.get("index_projection_years") if (cfg.figure_ranges and cfg.figure_ranges.get("index_projection_years")) else (2025, 2035))
    def _write_projection_slice(df, name):
        s = df.groupby(["year","src"])["count"].sum().unstack(fill_value=0).sort_index()
        s = s[(s.index>=a) & (s.index<=b)]
        s.to_csv(dirs.qc_index / f"{name}_index_projection_{a}_{b}.csv")
        plt.figure()
        for col in s.columns:
            plt.plot(s.index, s[col], marker="o" if col=="observed" else None,
                     linestyle="-" if col=="observed" else "--", label=col)
        plt.title(f"{name.upper()} projection {a}-{b}")
        plt.xlabel("Year"); plt.ylabel("Procedures / yr")
        plt.legend(); plt.tight_layout()
        _savefig_current(dirs.qc_index / f"{name}_index_projection_{a}_{b}.png")
    _write_projection_slice(sim.tavi_df, "tavi")
    _write_projection_slice(sim.savr_df, "savr")

    return cand_summary, real_summary, flow_mean

# =============================================================================
# Main / CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--precompute-only", action="store_true",
                   help="Run the pre‑compute steps and exit (no Monte‑Carlo).")
    args = p.parse_args()

    raw_cfg_text = Path(args.config).read_text(encoding="utf-8")
    cfg = Config.parse_obj(yaml.safe_load(raw_cfg_text))

    ts = dt.now().strftime("%Y-%m-%d-%H%M%S")
    tag = f"_{cfg.scenario_tag}" if cfg.scenario_tag else ""
    out_root = Path("runs") / cfg.experiment_name / f"{ts}{tag}"
    dirs = Dirs(out_root); dirs.make_all()

    setup_logging(args.log_level, dirs.logs)
    _apply_plot_theme(cfg.plotting)

    (dirs.meta / "config_raw.yaml").write_text(raw_cfg_text, encoding="utf-8")
    effective_cfg = cfg.model_dump(mode="json")
    (dirs.meta / "config_effective.yaml").write_text(
        yaml.safe_dump(effective_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    log.info("Outputs will be stored under %s", out_root)

    if cfg.precompute and cfg.precompute.enabled:
        log.info("Pre‑compute: building age/sex population, risk scores, and redo targets.")
        pop_path = Path(cfg.precompute.population_source["path"])
        units    = float(cfg.precompute.population_source.get("units_multiplier", 10000.0))
        derived_dir = (Path(cfg.precompute.derived_dir) if Path(cfg.precompute.derived_dir).is_absolute()
                       else out_root.parent.parent / cfg.precompute.derived_dir)
        derived_dir.mkdir(parents=True, exist_ok=True)

        p_band_sex, p_age_sex, p_age_all = build_age_and_sex_population(
            pop_csv=pop_path,
            out_dir=derived_dir,
            units_multiplier=units,
            project_span=(cfg.precompute.project_years[0], cfg.precompute.project_years[1]),
            split_50_64=tuple(cfg.precompute.split_50_64),
            split_75_84=tuple(cfg.precompute.split_75_84),
            sex_ratio_50_64=float(cfg.precompute.sex_ratio_50_64),
            open_width=int(cfg.precompute.open_age_bin_width),
        )
        try:
            shutil.copy2(p_age_all, dirs.inputs_snapshot / p_age_all.name)
            shutil.copy2(p_band_sex, dirs.inputs_snapshot / p_band_sex.name)
        except Exception:
            pass

        risks_csv = compute_risk_scores(cfg.procedure_counts, p_band_sex, cfg.precompute.risk_years)
        try:
            shutil.copy2(risks_csv, dirs.inputs_snapshot / Path(risks_csv).name)
        except Exception:
            pass

        if cfg.precompute.build_redo_savr_targets:
            redo_csv = project_redo_savr_targets(
                risk_scores_csv=risks_csv,
                population_band_sex_csv=p_band_sex,
                risk_rule=str(cfg.precompute.risk_rule),
                project_span=(cfg.precompute.project_years[0], cfg.precompute.project_years[1])
            )
            try:
                shutil.copy2(redo_csv, dirs.inputs_snapshot / Path(redo_csv).name)
            except Exception:
                pass

        if cfg.population_projection is None:
            cfg.population_projection = {
                "path": str(p_age_all),
                "year_col": "year", "age_col": "age", "pop_col": "population"
            }
        if (cfg.redo_savr_numbers is None) and cfg.precompute.build_redo_savr_targets:
            cfg.redo_savr_numbers = {
                "path": str(derived_dir / "redo_savr_targets.csv"),
                "mode": "replace_rates",
                "fill_missing": { "method": "linear", "range": cfg.precompute.project_years }
            }

        # ---- New: Demography and Risk figures ----
        try:
            demog_dir = dirs.figures / "demography"
            risks_dir = dirs.figures / "risks"
            _plot_demography(p_band_sex, p_age_sex, demog_dir)
            _plot_risk_scores(risks_csv, risks_dir)

        except Exception as e:
            log.warning("Could not create demography/risk plots: %s", e)

    if args.precompute_only:
        log.info("Pre‑compute complete. Exiting due to --precompute-only.")
        return


    # --- ADD after precompute writes or after you resolve the paths p_band_sex / p_age_sex / risks_csv ---
    try:
        demog_dir = dirs.figures / "demography"
        risks_dir = dirs.figures / "risks"
        demog_dir.mkdir(parents=True, exist_ok=True)
        risks_dir.mkdir(parents=True, exist_ok=True)
        _plot_demography(p_band_sex, p_age_sex, demog_dir)
        _plot_risk_scores(risks_csv, risks_dir)
        log.info("Demography and risk figures written to %s and %s", demog_dir, risks_dir)
    except Exception as e:
        log.warning("Could not create demography/risk plots: %s", e)


    # --- OPTIONAL: baseline risk computation + plots -------------------
    try:
        # Adjust these paths to match where you wrote the derived files
        derived_dir = Path("derived")
        age_band_sex_csv = derived_dir / "age_projections_by_year_band_sex.csv"
        if age_band_sex_csv.exists():
            risk_years = [2023, 2024]  # adjust if needed
            risk_out = Path("derived") / "baseline_risk_scores_by_sex.csv"
            risk_out.parent.mkdir(parents=True, exist_ok=True)

            redo_path = Path(str(cfg.procedure_counts.get("redo_savr", ""))) \
                        if "redo_savr" in cfg.procedure_counts else None

            # Build risk scores
            risk_csv = build_baseline_risk_scores(
                Path(cfg.procedure_counts["tavi"]),
                Path(cfg.procedure_counts["savr"]),
                redo_path,
                age_band_sex_csv,
                risk_years,
                cfg.open_age_bin_width,
                risk_out,
            )

            # Plot risk figures
            _plot_risk_scores(risk_csv, dirs.fig_risks)
            log.info("Risk scores and risk figures written under %s", dirs.fig_risks)
        else:
            log.warning("age_projections_by_year_band_sex.csv not found; skipping risk step.")
    except Exception as e:
        log.warning("Baseline risk computation failed: %s", e)


    cand, real, flow = run_simulation(cfg, dirs)

    log.info("Tables:")
    log.info("  %s", dirs.tables_index / "tavi_with_projection.csv")
    log.info("  %s", dirs.tables_index / "savr_with_projection.csv")
    log.info("  %s", dirs.tables_flow  / "patient_flow_mean.csv")
    log.info("  %s", dirs.tables_viv   / "viv_candidates_totals.csv")
    log.info("  %s", dirs.tables_viv   / "viv_candidates_v5like.csv")
    log.info("  %s", dirs.tables_viv   / "viv_forecast.csv")
    log.info("  %s", dirs.tables_viv   / "viv_forecast_realized.csv")
    log.info("QC:")
    log.info("  %s", dirs.qc_index / "tavi_index_projection_*.csv")
    log.info("  %s", dirs.qc_index / "savr_index_projection_*.csv")
    log.info("  %s", dirs.qc_redo  / "redo_savr_qc.csv")
    log.info("Figures:")
    log.info("  %s", dirs.fig_index / "image_A_tavi_index.png")
    log.info("  %s", dirs.fig_index / "image_B_savr_index.png")
    log.info("  %s", dirs.fig_index / "index_volume_overlay.png")
    log.info("  %s", dirs.fig_flow  / "flow_tavi.png")
    log.info("  %s", dirs.fig_flow  / "flow_savr.png")
    log.info("  %s", dirs.fig_viv   / "image_C_viv_pretty_*.png")

if __name__ == "__main__":
    main()
