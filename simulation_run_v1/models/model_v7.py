#!/usr/bin/env python
"""
model_v7.py — Monte‑Carlo ViV‑TAVI forecast with organized outputs, correct aggregation,
              optional population‑rate index projection, penetration, and redo handling.

Key improvements vs v6
----------------------
• Output folders are structured: logs/, meta/, inputs_snapshot/, tables/, qc/, figures/
• Aggregation FIX: sum across risk×age per run, then average across runs (correct totals)
  - Also writes a v5-like table (mean-of-bins) for QA: tables/viv/viv_candidates_v5like.csv
• Redo-SAVR loader accepts either:
    - redo_savr_numbers: { path: ... }   OR
    - procedure_counts.redo_savr: ...
  and handles both schemas: year,count  OR  sex,age_band,year,count
• Redo mode: 'replace_rates' (default) turns OFF redo_rates inside MC to avoid double count
• Clamp durability: simulation.min_dur_years (default 1.0) avoids same-year failures
• Plots background/theme via plotting: { background: "light"|"darkgrey"|"black" }
• Config v5->v7 fallback: if index_projection missing, use v5-like volume_extrapolation

Run:
    python models/model_v7.py --config configs/model_v7_configs.yaml --log-level DEBUG
    /Users/charles/miniconda3/bin/python /Users/charles/Desktop/viv_tavi_forecasting/simulation_run_v1/models/model_v7.py --config configs/model_v7_configs.yaml --log-level DEBUG
"""

from __future__ import annotations

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

# ──────────────────────────────────────────────────────────────────────
# Plotting options / theme
# ──────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────
# Config models (Pydantic v2)
# ──────────────────────────────────────────────────────────────────────

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

class Config(BaseModel):
    experiment_name: str
    scenario_tag: Optional[str] = None

    years: Dict[str, int]
    procedure_counts: Dict[str, Path]

    index_projection: Optional[IndexProjection] = None
    volume_extrapolation: Optional[Dict[str, int | str | dict]] = None

    open_age_bin_width: int
    age_bins: List[int]

    durability: Dict[str, Dict[str, DurabilityMode]]
    survival_curves: Dict[str, Dict[str, float]]

    risk_model: Dict[str, object]
    risk_mix: Dict[str, Dict[str, float] | Dict[str, Dict[str, float]]]
    age_hazard: Optional[Dict[str, object]] = None

    penetration: Dict[str, Dict[str, float]]
    redo_rates: Dict[str, float]

    simulation: Dict[str, int | float]
    outputs: Dict[str, str]     # unused but kept for compat

    population_projection: Optional[Dict[str, str]] = None
    redo_savr_numbers: Optional[Dict[str, object]] = None

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

# ──────────────────────────────────────────────────────────────────────
# Small path helper to keep folders tidy
# ──────────────────────────────────────────────────────────────────────

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

    def make_all(self):
        for p in (self.logs, self.meta, self.inputs_snapshot,
                  self.tables_index, self.tables_flow, self.tables_viv, self.tables_viv_per_run,
                  self.qc_index, self.qc_redo,
                  self.fig_index, self.fig_flow, self.fig_viv):
            p.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────

def parse_age_band(band: str, open_width: int) -> Tuple[int, int]:
    s = re.sub(r'\s*(yrs?|years?)\s*', '', band.strip().lower())
    open_lo = None
    m = re.match(r'(>=\s*(\d+))|(\s*(\d+)\+)', s)
    if m:
        open_lo = int(m.group(2) or m.group(4))
    if open_lo is None:
        m = re.match(r'>\s*(\d+)', s)
        if m:
            open_lo = int(m.group(1)) + 1
    if open_lo is not None:
        return open_lo, open_lo + open_width
    m = re.match(r'<\s*(\d+)', s)
    if m:
        return 0, int(m.group(1))
    m = re.match(r'(\d+)\s*-\s*(\d+)', s)
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

# ──────────────────────────────────────────────────────────────────────
# Population-rate projection helpers
# ──────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────
# v5-style linear/constant helpers + redistribution
# ──────────────────────────────────────────────────────────────────────

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
        "age_band": "75-79",  # placeholder; redistributed later if pop available
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

# ──────────────────────────────────────────────────────────────────────
# Redo-SAVR loader
# ──────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────

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
        self.min_dur         = float(cfg.simulation.get("min_dur_years", 1.0))  # NEW: default 1.0 yr

        # --- load observed
        tavi_obs = pd.read_csv(cfg.procedure_counts["tavi"])
        savr_obs = pd.read_csv(cfg.procedure_counts["savr"])

        # snapshot inputs (best effort)
        try:
            shutil.copy2(cfg.procedure_counts["tavi"], dirs.inputs_snapshot / "registry_tavi.csv")
            shutil.copy2(cfg.procedure_counts["savr"], dirs.inputs_snapshot / "registry_savr.csv")
        except Exception:
            pass

        # population (optional)
        pop_df = None
        pop_cfg = cfg.population_projection
        if pop_cfg and Path(str(pop_cfg.get("path",""))).exists():
            pop_df = pd.read_csv(pop_cfg["path"])
            self.pop_cols = (pop_cfg["year_col"], pop_cfg["age_col"], pop_cfg["pop_col"])
        else:
            self.pop_cols = ("year","age","population")
            log.warning("Population projection not found; population-based features limited.")

        # projection per procedure
        ip = cfg.index_projection
        assert ip is not None
        # TAVI
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
        # SAVR
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

        # save full index tables (observed+projection)
        self.tavi_df.to_csv(dirs.tables_index / "tavi_with_projection.csv", index=False)
        self.savr_df.to_csv(dirs.tables_index / "savr_with_projection.csv", index=False)

        log.info("TAVI rows %d, SAVR rows %d (obs+projection)", len(self.tavi_df), len(self.savr_df))

        # dists
        dur = cfg.durability
        self.dur_tavi       = DistFactory(dur["tavi"])
        self.dur_savr_lt70  = DistFactory(dur["savr_bioprosthetic"]["lt70"])
        self.dur_savr_gte70 = DistFactory(dur["savr_bioprosthetic"]["gte70"])

        s = cfg.survival_curves
        if self.n_cat == 2:
            self.surv_low = norm(loc=s["low_risk"]["median"], scale=s["low_risk"]["sd"])
            self.surv_ih  = norm(loc=s["int_high_risk"]["median"], scale=s["int_high_risk"]["sd"])
        else:
            self.surv_low  = norm(loc=s["low"]["median"],  scale=s["low"]["sd"])
            self.surv_int  = norm(loc=s["intermediate"]["median"], scale=s["intermediate"]["sd"])
            self.surv_high = norm(loc=s["high"]["median"], scale=s["high"]["sd"])

        # redo targets + mode
        self.redo_targets = self._load_redo_savr_numbers()

    def _load_redo_savr_numbers(self) -> Dict[int, int]:
        """
        Load absolute redo-SAVR targets.
        Priority:
          1) redo_savr_numbers.{values|path}
          2) procedure_counts.redo_savr
        mode: 'replace_rates' (default) or 'haircut'
        """
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
                
        # Decide whether to use per-event redo rates during MC
        self.use_redo_rates = not (out and self.redo_mode == "replace_rates")
        if out and not self.use_redo_rates:
            if (self.rr_savr_cfg > 0) or (self.rr_tavi_cfg > 0):
                log.info("Absolute redo targets present (mode=replace_rates); "
                         "ignoring redo_rates during simulation.")

        # ---- NEW: fill_missing policy to avoid single-year cliffs ----
        # cfg.redo_savr_numbers.fill_missing: {method: zero|forward|linear, range: [start,end]}
        rs_fm = (self.cfg.redo_savr_numbers or {}).get("fill_missing", {}) if isinstance(self.cfg.redo_savr_numbers, dict) else {}
        method = str(rs_fm.get("method", "zero")).lower()
        yrange = rs_fm.get("range", [self.start_forecast, self.end_year])
        try:
            y_lo, y_hi = int(yrange[0]), int(yrange[1])
        except Exception:
            y_lo, y_hi = self.start_forecast, self.end_year

        if out and method in ("forward", "linear"):
            years = list(range(y_lo, y_hi+1))
            anchors = sorted(out.items())  # e.g. [(2023, 220), ...]
            if not anchors:
                return out
            if method == "forward":
                filled = {}
                last = 0
                # start with 0 until the first anchor year
                first_year, first_val = anchors[0]
                for y in years:
                    if y in out:
                        last = out[y]
                    elif y < first_year:
                        last = 0
                    filled[y] = last
                out = filled
            else:  # linear
                xs = [k for k, _ in anchors]; ys = [v for _, v in anchors]
                arr = np.interp(years, xs, ys, left=ys[0], right=ys[-1])
                out = {y: int(round(v)) for y, v in zip(years, arr)}

        # store and return
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

    def _pen(self, vtype: str, year: int) -> float:
        anchors = self.cfg.penetration.get(vtype, {})
        if not anchors:
            return 1.0
        return _interp_scalar_by_year(year, anchors)

    def run_once(self, run_id: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          cand_df : [year, viv_type, risk, age_bin, count]  (viable)
          real_df : [year, viv_type, count]                 (realized after penetration/redo-rates)
          flow_df : [year, proc, index_cnt, deaths, failures, viable]
        """
        edges  = self.cfg.age_bins
        labels = _age_labels(edges)

        candidates: list[tuple[int,str,str,str]] = []
        realized:   list[tuple[int,str]] = []
        redo_log:   list[tuple[int,str]] = []

        idx_cnt  = defaultdict(int)
        deaths   = defaultdict(int)
        fails    = defaultdict(int)
        viable   = defaultdict(int)

        rr_savr = self.rr_savr_cfg if self.use_redo_rates else 0.0
        rr_tavi = self.rr_tavi_cfg if self.use_redo_rates else 0.0

        for proc_type, df in (("tavi", self.tavi_df), ("savr", self.savr_df)):
            for _, row in df.iterrows():
                year = int(row["year"])
                n    = int(row["count"])
                idx_cnt[(year, proc_type)] += n

                lo, hi = parse_age_band(row["age_band"], self.open_width)
                ages   = self.rng.integers(lo, hi, size=n)

                mix = self._risk_mix(proc_type, year)
                if self.n_cat == 2:
                    low_mask = self.rng.random(n) < mix["low"]
                    risk = np.where(low_mask, "low", "ih")
                else:
                    rnd  = self.rng.random(n)
                    risk = np.where(rnd < mix["low"], "low",
                                    np.where(rnd < mix["low"] + mix["intermediate"], "int", "high"))

                surv = np.empty(n)
                for tag in ("low", "ih" if self.n_cat==2 else "int", "high"):
                    m = (risk == tag)
                    if m.any():
                        surv[m] = self._sample_survival(tag, ages[m])

                dur = (self.dur_tavi.sample(n, self.rng) if proc_type=="tavi"
                       else np.where(ages < 70, self.dur_savr_lt70.sample(n, self.rng),
                                     self.dur_savr_gte70.sample(n, self.rng)))
                # clamp & discretize to calendar years
                dur = np.maximum(dur, self.min_dur)
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

                    # redo first (if enabled), then ViV via penetration
                    redo_p = rr_tavi if vtype == "tavi_in_tavi" else rr_savr
                    if self.rng.random() < redo_p:
                        redo_log.append((int(fy), vtype))
                        continue
                    if self.rng.random() < self._pen(vtype, int(fy)):
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

# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def _plot_flow(stats: pd.DataFrame, proc: str, out_path: Path) -> None:
    sub = stats[stats.proc == proc].set_index("year").sort_index()
    plt.figure()
    plt.plot(sub.index, sub.index_cnt,   label="Index procedures")
    plt.plot(sub.index, sub.deaths,      label="Deaths")
    plt.plot(sub.index, sub.failures,    label="Valve failures")
    plt.plot(sub.index, sub.viable,      label="Viable ViV candidates")
    plt.title(f"Patient flow – {proc.upper()}")
    plt.xlabel("Calendar year"); plt.ylabel("Head-count")
    plt.legend(); plt.tight_layout()
    _savefig_current(out_path)

def _plot_index_series(df: pd.DataFrame, title: str, out_path: Path,
                       observed_cutoff_year: Optional[int]) -> None:
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
    plt.title(title); plt.xlabel("Year"); plt.ylabel("Procedures / yr")
    plt.legend(); plt.tight_layout()
    _savefig_current(out_path)

def _plot_viv_pretty(realized_summary: pd.DataFrame,
                     year_lo: int, year_hi: int,
                     out_path: Path,
                     bar_color: Optional[str] = None,
                     label_bars: bool = True) -> None:
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
    fig.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def _plot_viv_pretty_with_overlay(pre_summary: pd.DataFrame,
                                  post_summary: pd.DataFrame,
                                  year_lo: int, year_hi: int,
                                  out_path: Path,
                                  bar_color: Optional[str],
                                  label_bars: bool) -> None:
    # prepare pre and post (same shaping as the simple pretty function)
    def _wide(df):
        sub = df[(df.year>=year_lo)&(df.year<=year_hi)]
        w = sub.pivot(index="year", columns="viv_type", values="mean").fillna(0.0)
        for c in ("tavi_in_savr","tavi_in_tavi"): 
            if c not in w.columns: w[c] = 0.0
        w["total"] = w["tavi_in_savr"] + w["tavi_in_tavi"]
        return w

    pre = _wide(pre_summary)
    post = _wide(post_summary)

    fig, ax = plt.subplots(figsize=(10,5), dpi=140)
    bar_color_final = bar_color or "#d0d0d0"

    # Bars = POST total
    bars = ax.bar(post.index, post["total"], label="Total ViV (post)", color=bar_color_final, alpha=0.5, zorder=1)

    # Solid lines = POST
    l1, = ax.plot(post.index, post["tavi_in_savr"], marker="o", label="TAVR-in-SAVR (post)",
                  color="red", alpha=0.9, linewidth=2, zorder=3)
    l2, = ax.plot(post.index, post["tavi_in_tavi"], marker="s", label="TAVR-in-TAVR (post)",
                  color="blue", alpha=0.9, linewidth=2, zorder=3)

    # Dashed lines = PRE
    ax.plot(pre.index, pre["tavi_in_savr"], linestyle="--", marker=None, color="red", alpha=0.7, label="... pre (SAVR)")
    ax.plot(pre.index, pre["tavi_in_tavi"], linestyle="--", marker=None, color="blue", alpha=0.7, label="... pre (TAVR)")

    max_val = float(np.nanmax([pre.values.max() if len(pre) else 0, post.values.max() if len(post) else 0])) if (len(pre) or len(post)) else 0.0
    y_off_line = 0.015 * max_val
    y_off_bar  = 0.020 * max_val

    # label solid post lines
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
    fig.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
def _save_viv_qc_plots(detail: pd.DataFrame, summary: pd.DataFrame,
                       out_dir: Path, age_edges: list[int]) -> None:
    """
    • detail  : year x viv_type x risk x age_bin x mean
    • summary : year x viv_type x mean
    Saves 3 pngs  +  a total-wide csv
    """
    # ————————————————— line plot (total + per-type) ——————————————
    wide = (summary.pivot(index="year", columns="viv_type", values="mean")
                  .fillna(0))

    # Stable, readable column order (types first, then total)
    preferred = ["tavi_in_savr", "tavi_in_tavi"]
    cols = [c for c in preferred if c in wide.columns] + \
           [c for c in wide.columns if c not in preferred]
    wide = wide[cols]
    wide["total"] = wide.sum(axis=1)

    wide.to_csv(out_dir / "viv_forecast_total.csv", index=True)

    # Plot with Image C-like markers + bold, color-matched labels
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)

    def _marker_for(col: str) -> str:
        return {"tavi_in_savr": "o", "tavi_in_tavi": "s", "total": "D"}.get(col, "o")

    line_handles = {}
    for col in wide.columns:
        h, = ax.plot(
            wide.index, wide[col].values,
            label=col.replace("_", " "),
            marker=_marker_for(col)
        )
        line_handles[col] = h

    # Label each point with a bold integer in the line color
    max_val = float(wide.to_numpy().max()) if not wide.empty else 0.0
    y_offset = 0.015 * max_val if max_val > 0 else 0.5

    for col, h in line_handles.items():
        color = h.get_color()
        ys = wide[col].values
        for x, y in zip(wide.index, ys):
            if pd.isna(y):
                continue
            ax.text(
                x, y + y_offset, f"{int(round(y))}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color=color
            )

    # Axes cosmetics (keep your 5-year ticks)
    ax.set_title("Predicted ViV-TAVI volumes")
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Procedures / yr")
    ax.xaxis.set_major_locator(mtick.MultipleLocator(5))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    if max_val > 0:
        ax.set_ylim(top=max_val * 1.12)  # headroom for labels
    fig.tight_layout()
    ax.legend()

    # Preserve themed background if using dark mode
    fig.savefig(out_dir / "viv_forecast.png",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    

# ──────────────────────────────────────────────────────────────────────
# Top-level driver
# ──────────────────────────────────────────────────────────────────────

def run_simulation(cfg: Config, dirs: Dirs):
    sim = ViVSimulator(cfg, dirs)

    # Figures - Index series
    # observed cutoff = max observed year in each table
    tavi_last_obs = int(sim.tavi_df[sim.tavi_df["src"]=="observed"]["year"].max()) if not sim.tavi_df.empty else None
    savr_last_obs = int(sim.savr_df[sim.savr_df["src"]=="observed"]["year"].max()) if not sim.savr_df.empty else None
    _plot_index_series(sim.tavi_df, "TAVI index (observed + projection)",
                       dirs.fig_index / "image_A_tavi_index.png", tavi_last_obs)
    _plot_index_series(sim.savr_df, "SAVR index (observed + projection)",
                       dirs.fig_index / "image_B_savr_index.png", savr_last_obs)

    # Optional overlay (handy QC)
    tot_tavi = sim.tavi_df.groupby("year")["count"].sum()
    tot_savr = sim.savr_df.groupby("year")["count"].sum()
    plt.figure()
    plt.plot(tot_tavi.index, tot_tavi.values, label="TAVI", marker="o")
    plt.plot(tot_savr.index, tot_savr.values, label="SAVR", marker="s")
    plt.xlabel("Year"); plt.ylabel("Procedures / yr")
    plt.title("Observed + projected index procedures")
    plt.tight_layout(); plt.legend()
    _savefig_current(dirs.fig_index / "index_volume_overlay.png")

    # Monte Carlo
    cand_runs, real_runs, flow_runs = [], [], []
    for i in range(cfg.simulation["n_runs"]):
        cand, real, flow = sim.run_once(i)
        cand["run"] = i; real["run"] = i; flow["run"] = i
        cand_runs.append(cand); real_runs.append(real); flow_runs.append(flow)

    cand_all = pd.concat(cand_runs, ignore_index=True) if cand_runs else pd.DataFrame(columns=["year","viv_type","risk","age_bin","count","run"])
    real_all = pd.concat(real_runs, ignore_index=True) if real_runs else pd.DataFrame(columns=["year","viv_type","count","run"])
    flow_all = pd.concat(flow_runs, ignore_index=True) if flow_runs else pd.DataFrame(columns=["year","proc","index_cnt","deaths","failures","viable","run"])

    # ---------- Correct aggregation ----------
    cand_per_run = (cand_all.groupby(["run","year","viv_type"])["count"].sum().reset_index())
    real_per_run = (real_all.groupby(["run","year","viv_type"])["count"].sum().reset_index())

    cand_summary = (cand_per_run.groupby(["year","viv_type"])
                    .agg(mean=("count","mean"), sd=("count","std"))
                    .reset_index())
    real_summary = (real_per_run.groupby(["year","viv_type"])
                    .agg(mean=("count","mean"), sd=("count","std"))
                    .reset_index())

    # (for QA) the old v5-like, deflated numbers
    cand_v5like = (cand_all.groupby(["year","viv_type"])
                   .agg(mean=("count","mean"), sd=("count","std"))
                   .reset_index())

    # Save tables
    dirs.tables_viv_per_run.mkdir(parents=True, exist_ok=True)
    cand_per_run.to_csv(dirs.tables_viv_per_run / "candidates_per_run.csv", index=False)
    real_per_run.to_csv(dirs.tables_viv_per_run / "realized_per_run.csv", index=False)
    cand_summary.to_csv(dirs.tables_viv / "viv_candidates_totals.csv", index=False)
    cand_v5like.to_csv(dirs.tables_viv / "viv_candidates_v5like.csv", index=False)
    real_summary.to_csv(dirs.tables_viv / "viv_forecast.csv", index=False)

    # Flow (mean across runs)
    flow_mean = (flow_all.groupby(["year","proc"]).mean(numeric_only=True).reset_index())
    flow_mean.to_csv(dirs.tables_flow / "patient_flow_mean.csv", index=False)

    _plot_flow(flow_mean, "savr", dirs.fig_flow / "flow_savr.png")
    _plot_flow(flow_mean, "tavi", dirs.fig_flow / "flow_tavi.png")

    # ---------- Redo reconciliation (applies to TAVR-in-SAVR only) ----------
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
            adj = max(0.0, m - float(target))  # subtract mode; floored at 0
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
    

    # ---------- v5-style line charts for three series ----------
    # Output folders
    lines_dir_cand = dirs.fig_viv / "lines_candidates"
    lines_dir_pre  = dirs.fig_viv / "lines_pre"
    lines_dir_post = dirs.fig_viv / "lines_post"
    for _d in (lines_dir_cand, lines_dir_pre, lines_dir_post):
        _d.mkdir(parents=True, exist_ok=True)

    # detail dataframe (signature req only; function doesn't actually use it)
    detail_stub = (cand_all.groupby(["year","viv_type","risk","age_bin"])
                            .agg(mean=("count","mean"))
                            .reset_index())

    # 1) Candidates (what the old v5 line chart historically showed)
    _save_viv_qc_plots(
        detail_stub,
        cand_summary[["year","viv_type","mean"]],
        lines_dir_cand,
        cfg.age_bins
    )

    # 2) Realized PRE (what Image C "pre" lines display)
    _save_viv_qc_plots(
        detail_stub,
        real_summary[["year","viv_type","mean"]],
        lines_dir_pre,
        cfg.age_bins
    )

    # 3) Realized POST (what Image C bars/solid lines display in 'post' mode)
    summary_post = realized_adj[["year","viv_type","realized"]].rename(columns={"realized":"mean"})
    _save_viv_qc_plots(
        detail_stub,
        summary_post,
        lines_dir_post,
        cfg.age_bins
    )

    # Copy each 'viv_forecast_total.csv' into tables/viv with clear names
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

    # Image C — choose source: post | pre | both
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
    else:  # post (default)
        _plot_viv_pretty(
            realized_adj[["year","viv_type","realized"]].rename(columns={"realized":"mean"}),
            ylo, yhi,
            dirs.fig_viv / f"image_C_viv_pretty_{ylo}_{yhi}.png",
            bar_color=bar_col, label_bars=labels_on_bars
        )

    # Index projection QC slices
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






# ──────────────────────────────────────────────────────────────────────
# Main / CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    raw_cfg_text = Path(args.config).read_text(encoding="utf-8")
    cfg = Config.parse_obj(yaml.safe_load(raw_cfg_text))

    ts = dt.now().strftime("%Y-%m-%d-%H%M%S")
    tag = f"_{cfg.scenario_tag}" if cfg.scenario_tag else ""
    out_root = Path("runs") / cfg.experiment_name / f"{ts}{tag}"
    dirs = Dirs(out_root)
    dirs.make_all()

    # logging + theme
    setup_logging(args.log_level, dirs.logs)
    _apply_plot_theme(cfg.plotting)

    # meta: save config copies
    (dirs.meta / "config_raw.yaml").write_text(raw_cfg_text, encoding="utf-8")
    effective_cfg = cfg.model_dump(mode="json")  # converts Path, datetime, etc. to strings
    (dirs.meta / "config_effective.yaml").write_text(
        yaml.safe_dump(effective_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    log.info("Outputs will be stored under %s", out_root)

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
