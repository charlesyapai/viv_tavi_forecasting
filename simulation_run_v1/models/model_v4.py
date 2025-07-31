#!/usr/bin/env python
"""
viv_simulator.py - Monte-Carlo forecast of future TAVI-in-TAVI / TAVI-in-SAVR demand
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

/Users/charles/miniconda3/bin/python /Users/charles/Desktop/viv_tavi_forecasting/simulation_run_v1/models/model_v4.py --config configs/model_v4_configs.yaml --log-level DEBUG

"""


from __future__ import annotations
import argparse
import logging
import sys
import re
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator
from scipy.stats import norm
import yaml
import matplotlib.pyplot as plt          # only imported *after* log filter set

import warnings
import pydantic

from collections import defaultdict
# ════════════════════════════════════════════════════════════════════════
# 1 ── Logging helpers
# ════════════════════════════════════════════════════════════════════════

warnings.filterwarnings("ignore", category=pydantic.PydanticDeprecatedSince20)

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
    age_bins: List[int]    

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
    
    @validator("age_bins")
    def _age_bins_ok(cls, v):
        if len(v) < 2 or any(b >= v[idx + 1] for idx, b in enumerate(v[:-1])):
            raise ValueError("age_bins must be an ascending list with ≥2 entries")
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
        "age_band": "75-79",                       # age pattern irrelevant - gets overwritten later
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

# ------
# Helpers
# ------



# ──────────────────────────────────────────────────────────────
def _bump(d: defaultdict[tuple, int], key: tuple, n: int = 1) -> None:
    d[key] += n

def _plot_flow(stats: pd.DataFrame, proc: str, out_dir: Path) -> None:
    """
    Line-plot for a single procedure type (savr | tavi) showing:
        • index volume
        • deaths
        • valve failures
        • viable ViV candidates
    """
    sub = stats[stats.proc == proc].set_index("year").sort_index()

    plt.figure()
    plt.plot(sub.index, sub.index_cnt,   label="Index procedures")
    plt.plot(sub.index, sub.deaths,      label="Deaths")
    plt.plot(sub.index, sub.failures,    label="Valve failures")
    plt.plot(sub.index, sub.viable,      label="ViV candidates")
    plt.title(f"Patient flow – {proc.upper()}")
    plt.xlabel("Calendar year"); plt.ylabel("Head-count")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"flow_{proc}.png"); plt.close()


# ──────────────────────────────────────────────────────────────────────
# 5 ── ViVSimulator  (full replacement)
# ──────────────────────────────────────────────────────────────────────
class ViVSimulator:
    """
    One Monte-Carlo world.  Handles
    • loading + extrapolating index-volume tables
    • sampling durability / survival
    • filtering to ViV candidates
    """

    # ..................................................................
    def __init__(self, cfg: Config, out_dir: Path):
        self.cfg   = cfg
        self.rng   = np.random.default_rng(cfg.simulation.get("rng_seed"))
        self.open_width      = cfg.open_age_bin_width
        self.start_forecast  = cfg.years["simulate_from"]
        self.end_year        = cfg.years["end"]
        self.n_cat           = cfg.risk_model["categories"]     # 2 or 3

        # 1 ── observed + synthetic index volumes ----------------------
        win = cfg.volume_extrapolation["window"]
        self.tavi_df = extrapolate_volumes(cfg.procedure_counts["tavi"],
                                           self.end_year, win, self.open_width, out_dir)
        self.savr_df = extrapolate_volumes(cfg.procedure_counts["savr"],
                                           self.end_year, win, self.open_width, out_dir)
        log.info("TAVI rows %d, SAVR rows %d (observed+synthetic)",
                 len(self.tavi_df), len(self.savr_df))

        # 2 ── durability samplers ------------------------------------
        dur = cfg.durability
        self.dur_tavi       = DistFactory(dur["tavi"])
        self.dur_savr_lt70  = DistFactory(dur["savr_bioprosthetic"]["lt70"])
        self.dur_savr_gte70 = DistFactory(dur["savr_bioprosthetic"]["gte70"])

        # 3 ── survival samplers (name depends on risk-model size) ----
        s = cfg.survival_curves
        if self.n_cat == 2:
            # YAML keys must be  low_risk / int_high_risk
            self.surv_low = norm(loc=s["low_risk"]["median"],
                                 scale=s["low_risk"]["sd"])
            self.surv_ih  = norm(loc=s["int_high_risk"]["median"],
                                 scale=s["int_high_risk"]["sd"])
        else:
            self.surv_low = norm(loc=s["low"]["median"],          scale=s["low"]["sd"])
            self.surv_int = norm(loc=s["intermediate"]["median"], scale=s["intermediate"]["sd"])
            self.surv_high= norm(loc=s["high"]["median"],         scale=s["high"]["sd"])

    # ..................................................................
    def _risk_mix(self, proc_type: str, year: int) -> Dict[str, float]:
        """
        Return risk-mix dict for a given procedure & calendar year.
        • If year is beyond last range, carry last period forward.
        • Works for both 2- and 3-category modes.
        """
        if proc_type == "savr":
            return self.cfg.risk_mix["savr"]

        # search for matching TAVI period
        for k, v in self.cfg.risk_mix["tavi"].items():
            a, b = (int(x) for x in k.split('-'))
            if a <= year <= b:
                return v

        # Fallback: extend the final block indefinitely
        last_key = max(self.cfg.risk_mix["tavi"].keys(),
                    key=lambda s: int(s.split('-')[1]))
        return self.cfg.risk_mix["tavi"][last_key]

    # ..................................................................
    def _sample_survival(self, tag: str, ages: np.ndarray) -> np.ndarray:
        """
        Draw survival years → apply optional age-hazard scaling → clip ≥0.1.
        tag  ∈  {'low','ih'}  or  {'low','int','high'}
        """
        if tag == "low":
            base = self.surv_low.rvs(len(ages), random_state=self.rng)
        elif tag in ("ih", "int", "intermediate"):
            base = self.surv_ih.rvs(len(ages), random_state=self.rng) \
                   if self.n_cat == 2 else \
                   self.surv_int.rvs(len(ages), random_state=self.rng)
        else:
            base = self.surv_high.rvs(len(ages), random_state=self.rng)

        # ── optional age-hazard scaling ---------------------------------
        if self.cfg.risk_model.get("use_age_hazard") and self.cfg.age_hazard:
            # Map any tag variant → canonical YAML key
            key_map = {
                "low": "low",
                "ih":  "intermediate",
                "int": "intermediate",
                "intermediate": "intermediate",
                "high": "high",
            }
            k = key_map.get(tag, tag)
            hr_per5 = self.cfg.age_hazard["hr_per5"].get(k, 1.0)
            ref_age = self.cfg.age_hazard["ref_age"]
            base = base / (hr_per5 ** ((ages - ref_age) / 5))

        return np.maximum(0.1, base)

    # ..................................................................
    @staticmethod
    def _interp_penetration(year: int, anchors: Dict[str, float]) -> float:
        """Piece-wise linear helper (anchor keys may be '2007-2022' or '2035')."""
        pts = []
        for k, v in anchors.items():
            if '-' in k:
                a, b = map(int, k.split('-'))
                pts += [(a, v), (b, v)]
            else:
                pts.append((int(k), v))
        pts.sort()
        xs, ys = zip(*pts)
        return float(np.interp(year, xs, ys))

    # ..................................................................
# ──────────────────────────────────────────────────────────────────────
    def run_once(self, run_id: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute **one** Monte-Carlo replication.

        Returns
        -------
        cand_df  : tidy table [year, viv_type, risk, age_bin, count]
        flow_df  : yearly patient-flow [year, proc, index_cnt, deaths,
                                        failures, viable]
        """
        # --------- DEBUG line templates ----------------------------------
        dbg_hdr  = ("Run {rid} │ {ptype:<4} │ yr={yr:<4} │ n={n:>5} │ "
                    "kept {kept:>5} → {vtype:<13}")
        dbg_tail = "Run %d │ TOTAL index=%d   kept=%d   cand=%d"

        edges  = self.cfg.age_bins
        labels = _age_labels(edges)

        # ---------- collectors ------------------------------------------
        candidates: list[tuple[int,str,str,str]] = []       # final tidy rows
        from collections import defaultdict
        idx_cnt  = defaultdict(int)   # (year, proc) → head-count
        deaths   = defaultdict(int)
        fails    = defaultdict(int)
        viable   = defaultdict(int)

        total_index = total_kept = 0

        # ---------- loop over registry rows ------------------------------
        for proc_type, df in (("tavi", self.tavi_df), ("savr", self.savr_df)):
            for _, row in df.iterrows():
                year = int(row["year"])
                n    = int(row["count"])
                total_index += n
                idx_cnt[(year, proc_type)] += n

                # Ages ----------------------------------------------------
                lo, hi = parse_age_band(row["age_band"], self.open_width)
                ages   = self.rng.integers(lo, hi, size=n)

                # Risk flags ---------------------------------------------
                mix = self._risk_mix(proc_type, year)
                if self.n_cat == 2:
                    low_mask = self.rng.random(n) < mix["low"]
                    risk = np.where(low_mask, "low", "ih")
                else:
                    rnd  = self.rng.random(n)
                    risk = np.where(rnd < mix["low"], "low",
                        np.where(rnd < mix["low"] + mix["intermediate"],
                                    "int", "high"))

                # Survival yrs -------------------------------------------
                surv = np.empty(n)
                for tag in ("low", "ih" if self.n_cat==2 else "int", "high"):
                    m = risk == tag
                    if m.any():
                        surv[m] = self._sample_survival(tag, ages[m])

                # Durability yrs -----------------------------------------
                dur = (self.dur_tavi.sample(n, self.rng) if proc_type=="tavi"
                    else np.where(ages < 70,
                                    self.dur_savr_lt70.sample(n, self.rng),
                                    self.dur_savr_gte70.sample(n, self.rng)))

                fail_y  = year + dur.astype(int)
                death_y = year + surv.astype(int)

                # Flow tallies -------------------------------------------
                for y in np.unique(death_y):
                    deaths[(y, proc_type)] += np.sum(death_y == y)
                for y in np.unique(fail_y):
                    fails[(y, proc_type)]  += np.sum(fail_y == y)

                valid = (fail_y <= death_y) & \
                        (self.start_forecast <= fail_y) & (fail_y <= self.end_year)
                kept = valid.sum()
                total_kept += kept

                if kept:
                    vtype = "tavi_in_tavi" if proc_type=="tavi" else "tavi_in_savr"
                    log.debug(dbg_hdr.format(rid=run_id, ptype=proc_type.upper(),
                                            yr=year, n=n, kept=kept, vtype=vtype))

                    age_idx = np.digitize(ages[valid], edges, right=False)
                    for fy, rk, ai in zip(fail_y[valid], risk[valid], age_idx):
                        viable[(int(fy), proc_type)] += 1
                        candidates.append((int(fy), vtype, rk, labels[ai]))

        # --------- end for-loops ----------------------------------------
        log.debug(dbg_tail, run_id, total_index, total_kept, len(candidates))

        # ----- candidates tidy frame ------------------------------------
        cand_df = (pd.DataFrame(candidates,
                                columns=["year","viv_type","risk","age_bin"])
                    .value_counts()
                    .rename("count")
                    .reset_index())

        # ----- yearly flow frame ----------------------------------------
        years = range(self.start_forecast, self.end_year + 1)
        rows  = []
        for y in years:
            for p in ("tavi", "savr"):
                rows.append([y, p,
                            idx_cnt[(y,p)],
                            deaths[(y,p)],
                            fails[(y,p)],
                            viable[(y,p)]])
        flow_df = pd.DataFrame(rows, columns=["year","proc",
                                            "index_cnt","deaths",
                                            "failures","viable"])

        return cand_df, flow_df

# ════════════════════════════════════════════════════════════════════════
# 5b --- Helper plotting function
# ════════════════════════════════════════════════════════════════════════


# ------------------------------------------------------------------
def _age_labels(edges):
    """Return list like ['<60', '60-64', ... ] matching `edges`."""
    labs = [f"<{edges[0]}"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        labs.append(f"{lo}-{hi-1}")
    return labs

# ------------------------------------------------------------------
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
    wide["total"] = wide.sum(axis=1)
    wide.to_csv(out_dir / "viv_forecast_total.csv", index=True)

    plt.figure()
    for col in wide.columns:
        plt.plot(wide.index, wide[col], label=col.replace("_", " "))
    plt.title("Predicted ViV-TAVI volumes")
    plt.xlabel("Calendar year")
    plt.ylabel("Procedures / yr")
    plt.tight_layout(); plt.legend()
    plt.savefig(out_dir / "viv_forecast.png"); plt.close()

    # ————————————————— stacked risk ——————————————
    if "risk" in detail.columns:
        risk_wide = (detail.groupby(["year", "risk"])["mean"].sum()
                            .unstack(fill_value=0))
        plt.figure()
        plt.stackplot(risk_wide.index, risk_wide.T.values, labels=risk_wide.columns)
        plt.title("ViV split by risk category")
        plt.xlabel("Calendar year"); plt.ylabel("Procedures / yr")
        plt.tight_layout(); plt.legend()
        plt.savefig(out_dir / "viv_by_risk.png"); plt.close()

    # ————————————————— stacked age ——————————————
    if "age_bin" in detail.columns:
        age_wide = (detail.groupby(["year", "age_bin"])["mean"].sum()
                           .unstack(fill_value=0)
                           .reindex(columns=_age_labels(age_edges), fill_value=0))
        plt.figure()
        plt.stackplot(age_wide.index, age_wide.T.values, labels=age_wide.columns)
        plt.title("ViV split by index-age band")
        plt.xlabel("Calendar year"); plt.ylabel("Procedures / yr")
        plt.tight_layout(); plt.legend()
        plt.savefig(out_dir / "viv_by_age.png"); plt.close()



# ————————————————————————————————————————————————————————————————
def _save_volume_overlay(tavi_df: pd.DataFrame,
                         savr_df: pd.DataFrame,
                         out_dir: Path) -> None:
    """
    One PNG that overlays total SAVR vs TAVI volumes
    (observed + extrapolated).
    """
    tot_tavi = tavi_df.groupby("year")["count"].sum()
    tot_savr = savr_df.groupby("year")["count"].sum()

    plt.figure()
    plt.plot(tot_tavi.index, tot_tavi.values, label="TAVI", marker="o")
    plt.plot(tot_savr.index, tot_savr.values, label="SAVR", marker="s")
    plt.xlabel("Year"); plt.ylabel("Procedures / yr")
    plt.title("Observed + projected index procedures")
    plt.tight_layout(); plt.legend()
    plt.savefig(out_dir / "index_volume_overlay.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# 6 ── Top-level multi-run driver
# ════════════════════════════════════════════════════════════════════════
def run_simulation(cfg: Config, out_dir: Path):
    sim = ViVSimulator(cfg, out_dir)
    _save_volume_overlay(sim.tavi_df, sim.savr_df, out_dir)

    cand_runs  = []
    flow_runs  = []
    for i in range(cfg.simulation["n_runs"]):
        cand, flow = sim.run_once(i)
        cand["run"] = i
        flow["run"] = i
        cand_runs.append(cand)
        flow_runs.append(flow)

    cand_all = pd.concat(cand_runs, ignore_index=True)
    flow_all = pd.concat(flow_runs, ignore_index=True)

    # —— traditional per-type ViV summary ———————————
    summary = (cand_all.groupby(["year","viv_type"])
                        .agg(mean=("count","mean"), sd=("count","std"))
                        .reset_index())
    summary.to_csv(out_dir / "viv_forecast.csv", index=False)

    # —— new flow summary ————————————————————————
    flow_mean = (flow_all.groupby(["year","proc"])
                         .mean(numeric_only=True)
                         .reset_index())
    flow_mean.to_csv(out_dir / "patient_flow.csv", index=False)

    # —— plots ————————————————————————————————
    _plot_flow(flow_mean, "savr", out_dir)
    _plot_flow(flow_mean, "tavi", out_dir)
    _save_viv_qc_plots(cand_all.groupby(["year","viv_type"])
                                   .agg(mean=("count","mean"))
                                   .reset_index(),
                       summary, out_dir, cfg.age_bins)

    return summary, flow_mean

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
    # if agg.empty:
    #     log.warning("Simulation returned empty result set.")
    # else:
    log.info("Done - %d rows (per-type) → viv_forecast.csv ; "
            "plot + total CSV also saved.", len(agg))
    
    summary, flow = run_simulation(cfg, out_dir)

    log.info("ViV file:     %s", out_dir / "viv_forecast.csv")
    log.info("Flow file:    %s", out_dir / "patient_flow.csv")
    log.info("Plots:        flow_savr.png, flow_tavi.png, index_volume_overlay.png")


    if summary.empty:
        log.warning("Simulation returned empty result set.")
    else:
        log.info("Done - %d yrs × %d ViV types written; plots ready.",
                summary.year.nunique(), summary.viv_type.nunique())



if __name__ == "__main__":
    main()
