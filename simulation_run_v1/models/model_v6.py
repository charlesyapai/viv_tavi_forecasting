#!/usr/bin/env python
"""
model_v6.py — Monte-Carlo forecast of future ViV-TAVI with
              population-rate index projection, penetration, and redo integration.

Major upgrades vs v5
--------------------
1) Index projection per procedure can be:
   • linear (as v5)
   • constant (as v5)
   • population_rate → totals = Σ(pop(age,year) × rate(age,year))
     - rates(age) estimated from observed registry ÷ population over a reference window
       OR provided explicitly
     - time drift via anchors or annual_rate_growth

2) Penetration is applied inside the Monte Carlo.
   • Each viable failure is routed to ViV with probability pen(year, viv_type)
   • A fixed share goes to redo-SAVR via `redo_rates`
   • Optional post-run adjustment to match absolute redo-SAVR registry numbers (capped)

3) Visualization bundle
   • Image A: TAVI index volumes (observed + projection)
   • Image B: SAVR index volumes (observed + projection)
   • Image C: ViV forecast (2023-2035): total as gray bars; TAVI-in-SAVR & TAVI-in-TAVI as lines with labels
   • QC: index projection CSV+PNG for chosen range (default 2025-2035)
   • Flow plots maintained

4) Config
   • New `index_projection.{tavi|savr}` block
   • Backward-compatible with v5's `volume_extrapolation`
"""

from __future__ import annotations
import argparse
import logging
import sys
import re
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, model_validator
from scipy.stats import norm
import yaml
import matplotlib.pyplot as plt

import warnings
import pydantic
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────
# 1) Logging
# ──────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", category=pydantic.PydanticDeprecatedSince20)

def setup_logging(level: str = "INFO", out_dir: Optional[Path] = None) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(out_dir / "run.log", mode="w"))
    logging.basicConfig(level=numeric, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# 2) Config models
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
    obs_years: str | List[int] | None = None  # e.g., "2018-2022" or [2018,2019,...]
    min_age: int = 0
    annual_rate_growth: Optional[float] = None
    anchors: Optional[Dict[str, float]] = None  # year→multiplier (piecewise linear)
    rates_by_age: Optional[Dict[int, float]] = None  # if source=from_config

class ProcProjection(BaseModel):
    method: str = "linear"                          # linear|constant|population_rate
    window: int = 5
    constant: Optional[ConstantSettings] = None
    pop_rate: Optional[PopRateSettings] = None

    @field_validator("method")
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
    outputs: Dict[str, str]

    population_projection: Optional[Dict[str, str]] = None
    redo_savr_numbers: Optional[Dict[str, object]] = None

    figure_ranges: Optional[Dict[str, List[int]]] = None

    # ---- v2 field validators ----
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
            raise ValueError("age_bins must be an ascending list with ≥2 entries")
        return v

    # ---- v2 model validator (replaces root_validator) ----
    @model_validator(mode="after")
    def _merge_v5_fallback(self):
        """
        Backward compatibility: if 'index_projection' missing, synthesize it
        from v5 'volume_extrapolation'.
        """
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
                if "method" in ve:
                    tavi_method = str(ve["method"])

            self.index_projection = IndexProjection(
                tavi=ProcProjection(method=tavi_method, window=window),
                savr=ProcProjection(
                    method=savr_method,
                    window=window,
                    constant=ConstantSettings(from_year=savr_from, value=savr_value),
                ),
            )
        return self

# ──────────────────────────────────────────────────────────────────────
# 3) Utilities
# ──────────────────────────────────────────────────────────────────────

def parse_age_band(band: str, open_width: int) -> Tuple[int, int]:
    """Inclusive-exclusive integer range from registry label."""
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
    """Handles '2007-2022: v' or '2035: v' anchor dictionaries."""
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
# 4) Population-rate projection helpers
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
    """Return base_rate[age] estimated from registry counts / population."""
    bands = _age_band_list(obs_df, open_width)
    # Build per-year, per-band population and counts
    per_age_rate_year: Dict[int, List[float]] = defaultdict(list)

    for y in years_sel:
        dfy = obs_df[obs_df["year"] == y]
        if dfy.empty:
            continue
        popy = pop_df[pop_df[ycol] == y]
        if popy.empty:
            # use closest year
            avail = pop_df[ycol].unique()
            if len(avail) == 0:
                continue
            closest = int(avail[np.argmin(np.abs(avail - y))])
            popy = pop_df[pop_df[ycol] == closest]

        # pre-aggregate pop by single ages
        pop_by_age = popy.set_index(acol)[pcol].to_dict()

        for band in bands:
            lo, hi = parse_age_band(band, open_width)
            pop_in_band = sum(pop_by_age.get(a, 0) for a in range(lo, hi))
            cnt = int(dfy[dfy["age_band"] == band]["count"].sum())
            if pop_in_band <= 0:
                continue
            band_rate = cnt / pop_in_band  # per-person annual rate
            for a in range(lo, hi):
                per_age_rate_year[a].append(band_rate)

    base_rate: Dict[int, float] = {}
    if not per_age_rate_year:
        return base_rate

    all_ages = sorted(per_age_rate_year.keys())
    for a in all_ages:
        rlist = per_age_rate_year[a]
        if len(rlist):
            base_rate[a] = max(0.0, float(np.mean(rlist)))
    # zero out below min_age
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
    """Return obs + future rows (src in {'observed','pop_rate'})."""
    out_rows = []
    last_obs_year = int(obs_df["year"].max())
    obs_df = obs_df.copy()
    obs_df["src"] = "observed"
    out_rows.append(obs_df)

    # base rates per age
    if pr.source == "from_config" and pr.rates_by_age:
        base_rate = {int(k): float(v) for k, v in pr.rates_by_age.items()}
    else:
        years_sel = _parse_year_selection(pr.obs_years, obs_df["year"].unique())
        base_rate = _estimate_rates_from_observed(
            obs_df, pop_df, years_sel, open_width, ycol, acol, pcol, pr.min_age
        )

    if not base_rate:
        log.warning("Population-based rates unavailable; skipping pop_rate projection.")
        return obs_df

    # projection years
    future_years = list(range(last_obs_year + 1, end_year + 1))

    # anchors vs annual growth
    def multiplier(y: int) -> float:
        if pr.anchors:
            return _interp_scalar_by_year(y, pr.anchors)
        if pr.annual_rate_growth is not None:
            base_y = max(base_rate.keys()) if len(base_rate) else y
            dy = y - last_obs_year
            return float((1.0 + pr.annual_rate_growth) ** dy)
        return 1.0

    bands = _age_band_list(obs_df, open_width)

    for y in future_years:
        popy = pop_df[pop_df[ycol] == y]
        if popy.empty:
            # choose closest
            avail = pop_df[ycol].unique()
            if len(avail) == 0:
                continue
            closest = int(avail[np.argmin(np.abs(avail - y))])
            popy = pop_df[pop_df[ycol] == closest]
        pop_by_age = popy.set_index(acol)[pcol].to_dict()

        # per-band expected counts
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
# 5) v5 linear/constant extrapolation helpers + redistribution
# ──────────────────────────────────────────────────────────────────────

def _extrapolate_linear_or_constant(csv_path: Path, end_year: int,
                                    window: int, open_width: int,
                                    out_dir: Path,
                                    method: str = "linear",
                                    const_from_year: Optional[int] = None,
                                    const_value: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path).assign(src="observed")
    last_year = int(df.year.max())

    # totals over years
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

    # placeholder rows; may be redistributed by population later
    synth = pd.DataFrame({
        "year": pred_years.repeat(1),
        "age_band": "75-79",
        "sex": "U",
        "count": preds,
        "src": "extrap"
    })
    full = pd.concat([df, synth], ignore_index=True)

    # QC line
    plt.figure()
    plt.plot(observed_totals.index, observed_totals.values, label="observed", marker="o")
    if len(pred_years):
        plt.plot(pred_years, preds, "--", label=method)
    plt.title(f"{csv_path.stem.upper()} volumes")
    plt.xlabel("Year"); plt.ylabel("Total procedures")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"{csv_path.stem}_volume_qc.png")
    plt.close()

    full.to_csv(out_dir / f"{csv_path.stem}_with_extrap.csv", index=False)
    return full

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
    out = pd.concat([keep, expanded], ignore_index=True)
    return out

# ──────────────────────────────────────────────────────────────────────
# 6) Simulator
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
    def __init__(self, cfg: Config, out_dir: Path):
        self.cfg   = cfg
        self.rng   = np.random.default_rng(cfg.simulation.get("rng_seed"))
        self.open_width      = cfg.open_age_bin_width
        self.start_forecast  = cfg.years["simulate_from"]
        self.end_year        = cfg.years["end"]
        self.n_cat           = cfg.risk_model["categories"]

        # --- load observed
        tavi_obs = pd.read_csv(cfg.procedure_counts["tavi"])
        savr_obs = pd.read_csv(cfg.procedure_counts["savr"])

        # --- population (optional)
        pop_df = None
        pop_cfg = cfg.population_projection
        if pop_cfg and Path(str(pop_cfg.get("path",""))).exists():
            pop_df = pd.read_csv(pop_cfg["path"])
            self.pop_cols = (pop_cfg["year_col"], pop_cfg["age_col"], pop_cfg["pop_col"])
        else:
            self.pop_cols = ("year","age","population")
            log.warning("Population projection not found; population-based features limited.")

        # --- projection per procedure
        ip = cfg.index_projection
        assert ip is not None, "index_projection block must be resolved by config validator"

        # TAVI
        if ip.tavi and ip.tavi.method == "population_rate" and pop_df is not None:
            self.tavi_df = _project_index_by_pop_rate(
                tavi_obs, self.end_year, pop_df, self.open_width,
                *self.pop_cols, ip.tavi.pop_rate or PopRateSettings()
            )
        else:
            # v5 style + redistribute future by population if available
            self.tavi_df = _extrapolate_linear_or_constant(
                cfg.procedure_counts["tavi"], self.end_year, ip.tavi.window if ip.tavi else 5,
                self.open_width, out_dir,
                method=(ip.tavi.method if ip.tavi else "linear")
            )
            if pop_df is not None:
                self.tavi_df = redistribute_future_by_population(
                    self.tavi_df, pop_df, self.open_width, *self.pop_cols
                )

        # SAVR
        if ip.savr and ip.savr.method == "population_rate" and pop_df is not None:
            self.savr_df = _project_index_by_pop_rate(
                savr_obs, self.end_year, pop_df, self.open_width,
                *self.pop_cols, ip.savr.pop_rate or PopRateSettings()
            )
        else:
            method = ip.savr.method if ip.savr else "constant"
            const_from = ip.savr.constant.from_year if (ip.savr and ip.savr.constant) else 2025
            const_val  = ip.savr.constant.value if (ip.savr and ip.savr.constant) else 2000
            self.savr_df = _extrapolate_linear_or_constant(
                cfg.procedure_counts["savr"], self.end_year, ip.savr.window if ip.savr else 5,
                self.open_width, out_dir,
                method=method, const_from_year=const_from, const_value=const_val
            )
            if pop_df is not None:
                self.savr_df = redistribute_future_by_population(
                    self.savr_df, pop_df, self.open_width, *self.pop_cols
                )

        log.info("TAVI rows %d, SAVR rows %d (obs+projection)", len(self.tavi_df), len(self.savr_df))

        # --- distributions
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

        # Preload redo savr targets
        self.redo_targets = self._load_redo_savr_numbers()

    # ----- helpers ------------------------------------------------------

    def _load_redo_savr_numbers(self) -> Dict[int, int]:
        out = {}
        rs = self.cfg.redo_savr_numbers or {}
        if isinstance(rs.get("values"), dict):
            out = {int(k): int(v) for k, v in rs["values"].items()}
        elif rs.get("path"):
            p = Path(str(rs["path"]))
            if p.exists():
                tmp = pd.read_csv(p)
                out = {int(r["year"]): int(r["count"]) for _, r in tmp.iterrows()}
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

    # ----- main single-run ---------------------------------------------

    def run_once(self, run_id: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns
        -------
        cand_df  : [year, viv_type, risk, age_bin, count]   (viable candidates)
        real_df  : [year, viv_type, count]                  (realized after penetration+redo)
        flow_df  : [year, proc, index_cnt, deaths, failures, viable]
        """
        edges  = self.cfg.age_bins
        labels = _age_labels(edges)

        candidates: list[tuple[int,str,str,str]] = []
        realized:   list[tuple[int,str]] = []   # per individual event as (year, vtype) for realized ViV
        redo_log:   list[tuple[int,str]] = []   # (year, source) for redo

        idx_cnt  = defaultdict(int)
        deaths   = defaultdict(int)
        fails    = defaultdict(int)
        viable   = defaultdict(int)

        rr_savr = float(self.cfg.redo_rates.get("savr_after_savr", 0.0))
        rr_tavi = float(self.cfg.redo_rates.get("savr_after_tavi", 0.0))

        # loop over index rows
        for proc_type, df in (("tavi", self.tavi_df), ("savr", self.savr_df)):
            for _, row in df.iterrows():
                year = int(row["year"])
                n    = int(row["count"])
                idx_cnt[(year, proc_type)] += n

                # Ages
                lo, hi = parse_age_band(row["age_band"], self.open_width)
                ages   = self.rng.integers(lo, hi, size=n)

                # Risk mix
                mix = self._risk_mix(proc_type, year)
                if self.n_cat == 2:
                    low_mask = self.rng.random(n) < mix["low"]
                    risk = np.where(low_mask, "low", "ih")
                else:
                    rnd  = self.rng.random(n)
                    risk = np.where(rnd < mix["low"], "low",
                                    np.where(rnd < mix["low"] + mix["intermediate"], "int", "high"))

                # Survival and durability
                surv = np.empty(n)
                for tag in ("low", "ih" if self.n_cat==2 else "int", "high"):
                    m = (risk == tag)
                    if m.any():
                        surv[m] = self._sample_survival(tag, ages[m])

                dur = (self.dur_tavi.sample(n, self.rng) if proc_type=="tavi"
                       else np.where(ages < 70, self.dur_savr_lt70.sample(n, self.rng),
                                     self.dur_savr_gte70.sample(n, self.rng)))

                fail_y  = year + dur.astype(int)
                death_y = year + surv.astype(int)

                # Flow tallies
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
                    # candidate record
                    ai = int(np.digitize(age, edges, right=False))
                    candidates.append((int(fy), vtype, rk, labels[ai]))

                    # Route: redo first (fixed rate), then penetration for ViV
                    redo_p = rr_tavi if vtype == "tavi_in_tavi" else rr_savr
                    if self.rng.random() < redo_p:
                        redo_log.append((int(fy), vtype))
                        continue
                    pen = self._pen(vtype, int(fy))
                    if self.rng.random() < pen:
                        realized.append((int(fy), vtype))
                    else:
                        # neither redo nor ViV (watchful waiting / no procedure)
                        pass

        # outputs
        cand_df = (pd.DataFrame(candidates, columns=["year","viv_type","risk","age_bin"])
                   .value_counts().rename("count").reset_index())
        real_df = (pd.DataFrame(realized, columns=["year","viv_type"])
                   .value_counts().rename("count").reset_index())
        redo_df = (pd.DataFrame(redo_log, columns=["year","source"])
                   .value_counts().rename("count").reset_index())

        # flow
        years = range(self.start_forecast, self.end_year + 1)
        rows  = []
        for y in years:
            for p in ("tavi","savr"):
                rows.append([y, p, idx_cnt[(y,p)], deaths[(y,p)], fails[(y,p)], viable[(y,p)]])
        flow_df = pd.DataFrame(rows, columns=["year","proc","index_cnt","deaths","failures","viable"])

        return cand_df, real_df, flow_df

# ──────────────────────────────────────────────────────────────────────
# 7) Plotting
# ──────────────────────────────────────────────────────────────────────

def _plot_flow(stats: pd.DataFrame, proc: str, out_dir: Path) -> None:
    sub = stats[stats.proc == proc].set_index("year").sort_index()
    plt.figure()
    plt.plot(sub.index, sub.index_cnt,   label="Index procedures")
    plt.plot(sub.index, sub.deaths,      label="Deaths")
    plt.plot(sub.index, sub.failures,    label="Valve failures")
    plt.plot(sub.index, sub.viable,      label="Viable ViV candidates")
    plt.title(f"Patient flow - {proc.upper()}")
    plt.xlabel("Calendar year"); plt.ylabel("Head-count")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"flow_{proc}.png"); plt.close()

def _plot_index_series(df: pd.DataFrame, title: str, out_path: Path) -> None:
    tot = df.groupby(["year","src"])["count"].sum().unstack(fill_value=0).sort_index()
    plt.figure()
    if "observed" in tot:
        plt.plot(tot.index, tot["observed"], marker="o", label="Observed")
        last_obs = int(tot[tot["observed"]>0].index.max())
        plt.axvline(last_obs, color="k", linestyle=":", linewidth=1)
    proj_cols = [c for c in tot.columns if c != "observed"]
    for c in proj_cols:
        plt.plot(tot.index, tot[c].where(tot[c]>0), "--", label=c)
    # shading for projection
    if "observed" in tot:
        plt.axvspan(last_obs+0.05, tot.index.max()+0.5, alpha=0.08, color="gray")
    plt.title(title); plt.xlabel("Year"); plt.ylabel("Procedures / yr")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()

def _plot_viv_pretty(realized_summary: pd.DataFrame,
                     year_lo: int, year_hi: int,
                     out_path: Path) -> None:
    # realized_summary: columns year, viv_type, mean
    sub = realized_summary[(realized_summary.year>=year_lo) & (realized_summary.year<=year_hi)]
    wide = sub.pivot(index="year", columns="viv_type", values="mean").fillna(0)
    for c in ("tavi_in_savr", "tavi_in_tavi"):
        if c not in wide.columns:
            wide[c] = 0.0
    wide["total"] = wide["tavi_in_savr"] + wide["tavi_in_tavi"]

    plt.figure(figsize=(10,5), dpi=140)
    # bars for total
    plt.bar(wide.index, wide["total"], label="Total ViV (realized)", alpha=0.25, color="gray")
    # lines
    plt.plot(wide.index, wide["tavi_in_savr"], marker="o", label="TAVR-in-SAVR")
    plt.plot(wide.index, wide["tavi_in_tavi"], marker="s", label="TAVR-in-TAVR")

    # labels above points
    for x, y in zip(wide.index, wide["tavi_in_savr"]):
        plt.text(x, y, f"{int(round(y))}", ha="center", va="bottom", fontsize=8)
    for x, y in zip(wide.index, wide["tavi_in_tavi"]):
        plt.text(x, y, f"{int(round(y))}", ha="center", va="bottom", fontsize=8)
    plt.title(f"Predicted ViV volume ({year_lo}-{year_hi})")
    plt.xlabel("Year"); plt.ylabel("Procedures / yr")
    plt.tight_layout(); plt.legend()
    plt.savefig(out_path); plt.close()

# ──────────────────────────────────────────────────────────────────────
# 8) Top-level run
# ──────────────────────────────────────────────────────────────────────

def run_simulation(cfg: Config, out_dir: Path):
    sim = ViVSimulator(cfg, out_dir)

    # Save index projection overlays + QC
    _plot_index_series(sim.tavi_df, "TAVI index (observed + projection)",
                       out_dir / "image_A_tavi_index.png")
    _plot_index_series(sim.savr_df, "SAVR index (observed + projection)",
                       out_dir / "image_B_savr_index.png")

    # Quick overlay (optional)
    tot_tavi = sim.tavi_df.groupby("year")["count"].sum()
    tot_savr = sim.savr_df.groupby("year")["count"].sum()
    plt.figure()
    plt.plot(tot_tavi.index, tot_tavi.values, label="TAVI", marker="o")
    plt.plot(tot_savr.index, tot_savr.values, label="SAVR", marker="s")
    plt.xlabel("Year"); plt.ylabel("Procedures / yr")
    plt.title("Observed + projected index procedures")
    plt.tight_layout(); plt.legend()
    plt.savefig(out_dir / "index_volume_overlay.png"); plt.close()

    # Run MC
    cand_runs, real_runs, flow_runs = [], [], []
    for i in range(cfg.simulation["n_runs"]):
        cand, real, flow = sim.run_once(i)
        cand["run"] = i; real["run"] = i; flow["run"] = i
        cand_runs.append(cand); real_runs.append(real); flow_runs.append(flow)

    cand_all = pd.concat(cand_runs, ignore_index=True) if len(cand_runs) else pd.DataFrame(columns=["year","viv_type","risk","age_bin","count","run"])
    real_all = pd.concat(real_runs, ignore_index=True) if len(real_runs) else pd.DataFrame(columns=["year","viv_type","count","run"])
    flow_all = pd.concat(flow_runs, ignore_index=True) if len(flow_runs) else pd.DataFrame(columns=["year","proc","index_cnt","deaths","failures","viable","run"])

    # Summaries (candidates and realized)
    cand_summary = (cand_all.groupby(["year","viv_type"])
                            .agg(mean=("count","mean"), sd=("count","std"))
                            .reset_index())
    real_summary = (real_all.groupby(["year","viv_type"])
                            .agg(mean=("count","mean"), sd=("count","std"))
                            .reset_index())

    cand_summary.to_csv(out_dir / "viv_candidates.csv", index=False)
    real_summary.to_csv(out_dir / "viv_forecast.csv", index=False)

    # Flow mean
    flow_mean = (flow_all.groupby(["year","proc"]).mean(numeric_only=True).reset_index())
    flow_mean.to_csv(out_dir / "patient_flow.csv", index=False)

    # Flow plots
    _plot_flow(flow_mean, "savr", out_dir)
    _plot_flow(flow_mean, "tavi", out_dir)

    # Optional adjustment to match absolute redo-SAVR numbers
    redo_targets = sim.redo_targets
    if redo_targets:
        realized_adj = real_summary.copy()
        realized_adj["realized"] = realized_adj["mean"]
        mask = realized_adj["viv_type"].eq("tavi_in_savr")
        # derive year totals so we can show QC vs targets
        tis_series = realized_adj[mask].set_index("year")["mean"]
        qc_rows = []
        adj_vals = []
        for y, m in tis_series.items():
            target = redo_targets.get(int(y), None)
            if target is None:
                adj = m
                qc_rows.append([int(y), int(round(m)), None, int(round(adj))])
                adj_vals.append((y, adj))
                continue
            # realized cannot go below zero
            adj = max(0.0, m - float(target))
            qc_rows.append([int(y), int(round(m)), int(target), int(round(adj))])
            adj_vals.append((y, adj))
        # write back
        adj_map = dict(adj_vals)
        realized_adj.loc[mask, "realized"] = realized_adj.loc[mask, "year"].map(adj_map)
        realized_adj.to_csv(out_dir / "viv_forecast_realized.csv", index=False)
        pd.DataFrame(qc_rows, columns=["year","tis_before","redo_target","tis_after"])\
          .to_csv(out_dir / "redo_savr_qc.csv", index=False)
    else:
        realized_adj = real_summary.copy()
        realized_adj["realized"] = realized_adj["mean"]
        log.info("No redo-SAVR absolute targets; 'viv_forecast_realized.*' not written.")

    # Image C — ViV pretty chart for requested window
    ylo, yhi = 2023, 2035
    if cfg.figure_ranges and cfg.figure_ranges.get("viv_years"):
        ylo, yhi = cfg.figure_ranges["viv_years"][0], cfg.figure_ranges["viv_years"][1]
    _plot_viv_pretty(
        realized_adj[["year","viv_type","realized"]].rename(columns={"realized":"mean"}),
        ylo, yhi,
        out_dir / f"image_C_viv_pretty_{ylo}_{yhi}.png"
    )

    # Optional: index projection QC table for requested range
    if cfg.figure_ranges and cfg.figure_ranges.get("index_projection_years"):
        a, b = cfg.figure_ranges["index_projection_years"]
    else:
        a, b = 2025, 2035

    def _write_projection_slice(df, name):
        s = df.groupby(["year","src"])["count"].sum().unstack(fill_value=0).sort_index()
        s = s[(s.index>=a) & (s.index<=b)]
        s.to_csv(out_dir / f"{name}_index_projection_{a}_{b}.csv")

        plt.figure()
        for col in s.columns:
            plt.plot(s.index, s[col], marker="o" if col=="observed" else None,
                     linestyle="-" if col=="observed" else "--", label=col)
        plt.title(f"{name.upper()} projection {a}-{b}")
        plt.xlabel("Year"); plt.ylabel("Procedures / yr")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"{name}_index_projection_{a}_{b}.png"); plt.close()

    _write_projection_slice(sim.tavi_df, "tavi")
    _write_projection_slice(sim.savr_df, "savr")

    return cand_summary, real_summary, flow_mean

# ──────────────────────────────────────────────────────────────────────
# 9) CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = Config.parse_obj(yaml.safe_load(fh))

    ts = dt.now().strftime("%Y-%m-%d-%H%M%S")
    tag = f"_{cfg.scenario_tag}" if cfg.scenario_tag else ""
    out_dir = Path("runs") / cfg.experiment_name / f"{ts}{tag}"
    setup_logging(args.log_level, out_dir)
    log.info("Outputs will be stored under %s", out_dir)

    cand, real, flow = run_simulation(cfg, out_dir)

    log.info("Files written:")
    log.info("  ViV candidates:      %s", out_dir / "viv_candidates.csv")
    log.info("  ViV forecast:        %s", out_dir / "viv_forecast.csv")
    log.info("  (Adj) ViV realized:  %s", out_dir / "viv_forecast_realized.csv")
    log.info("  Patient flow:        %s", out_dir / "patient_flow.csv")
    log.info("  Figures: image_A_tavi_index.png, image_B_savr_index.png, image_C_viv_pretty_*.png")
    log.info("  QC: tavi/savr_index_projection_*.csv + .png, redo_savr_qc.csv")

if __name__ == "__main__":
    main()
