#!/usr/bin/env python
"""
viv_simulator.py – Monte-Carlo forecast of future TAVI-in-TAVI / TAVI-in-SAVR demand
===================================================================================

Major additions vs the previous version
---------------------------------------
• risk_model.categories   → 2  (low vs IH)  | 3 (low / intermediate / high)
• risk_model.use_age_hazard → True/False
• open_age_bin_width      → width of final “≥xx” age bucket
• volume_extrapolation.*  → linear-fit window + fallback method
• experiment_name + scenario_tag → drive output folder organisation
• save_index_projection() → writes 2025-2035 projected counts + QC plot
• aggressive log filtering for matplotlib internals
"""

from __future__ import annotations
import argparse
import logging
import sys
import re
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator
from scipy.stats import norm
import yaml
import matplotlib.pyplot as plt          # only imported *after* log filter set

# ════════════════════════════════════════════════════════════════════════
# 1 ── Logging helpers
# ════════════════════════════════════════════════════════════════════════
def setup_logging(level: str = "INFO", out_dir: Path | None = None) -> None:
    """Tames matplotlib noise and streams logs to both console & disk."""
    numeric = getattr(logging, level.upper(), logging.INFO)

    # Silence matplotlib's font-manager DEBUG spam
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if out_dir:
        (out_dir / "run.log").parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(out_dir / "run.log", mode="w"))

    logging.basicConfig(level=numeric, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)

log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════
# 2 ── Config pydantic model
# ════════════════════════════════════════════════════════════════════════
class DurabilityMode(BaseModel):
    mean: float
    sd: float
    weight: float = 1.0

class Config(BaseModel):
    experiment_name: str
    scenario_tag:    str | None = None

    years: Dict[str, int]
    volume_extrapolation: Dict[str, int | str]
    open_age_bin_width: int

    procedure_counts: Dict[str, Path]

    durability: Dict[str, Dict[str, DurabilityMode]]
    survival_curves: Dict[str, Dict[str, float]]

    risk_model: Dict[str, object]      # validated below
    risk_mix: Dict[str, Dict[str, float] | Dict[str, Dict[str, float]]]
    age_hazard: Dict[str, object] | None = None

    penetration: Dict[str, Dict[str, float]]
    redo_rates: Dict[str, float]

    simulation: Dict[str, int | float]

    outputs: Dict[str, str]

    # -------------- validators -----------------------------
    @validator("procedure_counts")
    def _check_csvs(cls, v):
        for tag, path in v.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing csv for '{tag}': {path}")
        return v

    @validator("risk_model")
    def _risk_model_ok(cls, v):
        if v.get("categories") not in (2, 3):
            raise ValueError("risk_model.categories must be 2 or 3")
        return v

# ════════════════════════════════════════════════════════════════════════
# 3 ── Helper utilities
# ════════════════════════════════════════════════════════════════════════
def parse_age_band(band: str, open_width: int) -> Tuple[int, int]:
    """Inclusive-exclusive integer range from registry label."""
    s = re.sub(r'\s*(yrs?|years?)\s*', '', band.strip().lower())

    open_lo = None
    # >=NN or NN+
    m = re.match(r'(>=\s*(\d+))|(\s*(\d+)\+)', s)
    if m:
        open_lo = int(m.group(2) or m.group(4))
    # >NN
    if open_lo is None:
        m = re.match(r'>\s*(\d+)', s)
        if m:   open_lo = int(m.group(1)) + 1
    if open_lo is not None:
        return open_lo, open_lo + open_width

    m = re.match(r'<\s*(\d+)', s)
    if m:   return 0, int(m.group(1))

    m = re.match(r'(\d+)\s*-\s*(\d+)', s)
    if m:   return int(m.group(1)), int(m.group(2)) + 1

    raise ValueError(f"Unrecognised age band: '{band}'")

class DistFactory:
    """Mixture of (weighted) normals with .sample(n, rng)."""
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

# ════════════════════════════════════════════════════════════════════════
# 4 ── Extrapolate future index volumes & save QC artefacts
# ════════════════════════════════════════════════════════════════════════
def extrapolate_volumes(csv_path: Path, end_year: int,
                        window: int, open_width: int,
                        out_dir: Path) -> pd.DataFrame:
    """
    • Reads registry CSV (year, age_band, sex, count)
    • Appends synthetic rows up to end_year via linear trend on *totals*
    • Returns full DataFrame (observed + synthetic)
    • Saves CSV + a QC PNG plot
    """
    df = pd.read_csv(csv_path).assign(src="observed")
    last_year = df.year.max()
    totals = (df.groupby("year")["count"].sum()
                .sort_index()
                .tail(window))

    slope, intercept = np.polyfit(totals.index, totals.values, 1)
    pred_years = np.arange(last_year + 1, end_year + 1)
    preds = (slope * pred_years + intercept).clip(min=0).round().astype(int)

    synth = pd.DataFrame({
        "year": np.repeat(pred_years, 1),          # 1 row per yr holding placeholder age band
        "age_band": "75-79",                       # age pattern irrelevant – gets overwritten later
        "sex": "U",
        "count": preds,
        "src": "extrap"
    })

    full = pd.concat([df, synth], ignore_index=True)
    full.to_csv(out_dir / f"{csv_path.stem}_with_extrap.csv", index=False)

    # QC plot
    plt.figure()
    plt.plot(totals.index, totals.values, label="observed")
    plt.plot(pred_years, preds, "--", label="linear fit")
    plt.title(f"{csv_path.stem.upper()} volumes")
    plt.xlabel("Year")
    plt.ylabel("Total procedures")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{csv_path.stem}_volume_qc.png")
    plt.close()

    return full.drop(columns="src")

# ════════════════════════════════════════════════════════════════════════
# 5 ── Simulation engine
# ════════════════════════════════════════════════════════════════════════
class ViVSimulator:
    def __init__(self, cfg: Config, out_dir: Path):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.simulation.get("rng_seed"))
        self.open_width = cfg.open_age_bin_width
        self.start_forecast = cfg.years["simulate_from"]
        self.end_year       = cfg.years["end"]

        # 1. Load / extrapolate index volumes
        wx = cfg.volume_extrapolation["window"]
        self.tavi_df = extrapolate_volumes(cfg.procedure_counts["tavi"],
                                           self.end_year, wx, self.open_width, out_dir)
        self.savr_df = extrapolate_volumes(cfg.procedure_counts["savr"],
                                           self.end_year, wx, self.open_width, out_dir)

        log.info("TAVI rows %d, SAVR rows %d (observed+synthetic)",
                 len(self.tavi_df), len(self.savr_df))

        # 2. Prepare durability & survival samplers
        self.dur_tavi = DistFactory(cfg.durability["tavi"])
        self.dur_savr_lt70  = DistFactory(cfg.durability["savr_bioprosthetic"]["lt70"])
        self.dur_savr_gte70 = DistFactory(cfg.durability["savr_bioprosthetic"]["gte70"])

        s = cfg.survival_curves
        self.surv_low = norm(loc=s["low_risk"]["median"],  scale=s["low_risk"]["sd"])
        self.surv_int = norm(loc=s["intermediate"]["median"],  scale=s["intermediate"]["sd"])
        self.surv_high= norm(loc=s["high"]["median"], scale=s["high"]["sd"])

    # ------------------------------------------------------------------
    def _risk_mix(self, proc_type: str, year: int) -> Dict[str, float]:
        block = self.cfg.risk_mix["savr"] if proc_type == "savr" \
                else next(v for k, v in self.cfg.risk_mix["tavi"].items()
                          if int(k.split('-')[0]) <= year <= int(k.split('-')[1]))
        # Forced to 2 or 3 cats
        if self.cfg.risk_model["categories"] == 2:
            return {"low": block["low"],
                    "ih": 1 - block["low"]}
        return block     # already low/int/high

    # ------------------------------------------------------------------
    def _sample_survival(self, cat: str, ages: np.ndarray) -> np.ndarray:
        """Return survival years, applying optional age-hazard multiplier."""
        if cat == "low":
            base = self.surv_low.rvs(len(ages), random_state=self.rng)
        elif cat in ("int", "intermediate"):
            base = self.surv_int.rvs(len(ages), random_state=self.rng)
        else:
            base = self.surv_high.rvs(len(ages), random_state=self.rng)

        if self.cfg.risk_model.get("use_age_hazard"):
            hr_per5 = self.cfg.age_hazard["hr_per5"][cat[:3]]
            ref_age = self.cfg.age_hazard["ref_age"]
            adj = np.maximum(0.1, base / (hr_per5 ** ((ages - ref_age) / 5)))
            return adj
        return base.clip(min=0.1)

    # ------------------------------------------------------------------
    def run_once(self, run_id: int) -> pd.DataFrame:
        candidates: list[Tuple[int, str]] = []

        for proc_type, df in (("tavi", self.tavi_df), ("savr", self.savr_df)):
            for _, row in df.iterrows():
                year, n = int(row.year), int(row.count)
                lo, hi = parse_age_band(row.age_band, self.open_width)
                ages = self.rng.integers(lo, hi, size=n)

                # risk split
                mix = self._risk_mix(proc_type, year)
                if self.cfg.risk_model["categories"] == 2:
                    low_mask = self.rng.random(n) < mix["low"]
                    surv = np.where(low_mask,
                                    self._sample_survival("low", ages),
                                    self._sample_survival("ih", ages))
                else:
                    rnd = self.rng.random(n)
                    low_m = rnd < mix["low"]
                    int_m = (rnd >= mix["low"]) & (rnd < mix["low"] + mix["intermediate"])
                    surv = np.where(
                        low_m, self._sample_survival("low", ages),
                        np.where(int_m, self._sample_survival("int", ages),
                                 self._sample_survival("high", ages))
                    )

                # durability
                if proc_type == "tavi":
                    dur = self.dur_tavi.sample(n, self.rng)
                else:
                    dur = np.where(
                        ages < 70,
                        self.dur_savr_lt70.sample(n, self.rng),
                        self.dur_savr_gte70.sample(n, self.rng)
                    )

                fail_year  = year + dur.astype(int)
                death_year = year + surv.astype(int)
                valid = (fail_year <= death_year) \
                        & (self.start_forecast <= fail_year) \
                        & (fail_year <= self.end_year)
                if valid.any():
                    vtype = "tavi_in_tavi" if proc_type == "tavi" else "tavi_in_savr"
                    candidates.extend((fy, vtype) for fy in fail_year[valid])

        cand_df = (pd.DataFrame(candidates, columns=["year", "viv_type"])
                     .value_counts()
                     .rename("raw")
                     .reset_index()) if candidates else pd.DataFrame()

        if cand_df.empty:
            log.warning("Run %d produced 0 candidates", run_id)
            return cand_df

        # penetration
        cand_df["penetration"] = cand_df.apply(
            lambda r: np.interp(int(r.year),
                                *zip(*[(int(k.split('-')[0]), v)
                                       if '-' in k else (int(k), v)
                                       for k, v in self.cfg.penetration[r.viv_type].items()])),
            axis=1
        )
        cand_df["after_pen"] = (cand_df.raw * cand_df.penetration).round().astype(int)

        # redo-SAVR haircut
        rr = self.cfg.redo_rates
        cand_df["redo_rate"] = cand_df.viv_type.map(
            lambda vt: rr["savr_after_savr"] if vt == "tavi_in_savr"
                      else rr["savr_after_tavi"])
        cand_df["count"] = (cand_df.after_pen * (1 - cand_df.redo_rate)).round().astype(int)
        return cand_df[["year", "viv_type", "count"]]

# ════════════════════════════════════════════════════════════════════════
# 6 ── Top-level multi-run driver
# ════════════════════════════════════════════════════════════════════════
def run_simulation(cfg: Config, out_dir: Path) -> pd.DataFrame:
    sim = ViVSimulator(cfg, out_dir)
    runs = [sim.run_once(i).assign(run=i) for i in range(cfg.simulation["n_runs"])]
    all_df = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()

    agg = (all_df.groupby(["year", "viv_type"])
                 .agg(mean=("count", "mean"), sd=("count", "std"))
                 .reset_index())

    agg.to_csv(out_dir / "viv_forecast.csv", index=False)
    return agg

# ════════════════════════════════════════════════════════════════════════
# 7 ── CLI
# ════════════════════════════════════════════════════════════════════════
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

    agg = run_simulation(cfg, out_dir)
    if agg.empty:
        log.warning("Simulation returned empty result set.")
    else:
        log.info("Done – %d rows written to viv_forecast.csv", len(agg))

if __name__ == "__main__":
    main()
