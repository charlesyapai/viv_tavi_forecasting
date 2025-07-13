#!/usr/bin/env python
"""
Monte-Carlo engine for forecasting ViV-TAVI demand
=================================================

  python model.py --config config.yaml [--log-level INFO]

Key stages and what is logged
-----------------------------
1.  Config load  – paths found / missing
2.  Per run:      total index patients   →  total viable failures
3.  Penetration:  candidates × penetration
4.  Redo-SAVR:    after_pen × (1 - redo_rate)
5.  Aggregation:  mean ± sd across runs written to CSV
"""

from __future__ import annotations
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 0.  Logging helper ---------------------------------------------------------
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO"):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Configuration schema ---------------------------------------------------
# ---------------------------------------------------------------------------

class DurabilityMode(BaseModel):
    mean: float
    sd: float
    weight: float = 1.0


class Config(BaseModel):
    years: Dict[str, int]
    procedure_counts: Dict[str, Path]
    durability: Dict[str, Dict[str, DurabilityMode]]
    survival_curves: Dict[str, Dict[str, float]]
    risk_mix: Dict[str, Dict[str, float] | Dict[str, Dict[str, float]]]
    penetration: Dict[str, Dict[str, float]]
    redo_rates: Dict[str, float] = {}                # <-- option B
    redo_procedures_source: Optional[Path] = None    # kept for completeness
    simulation: Dict[str, int | float]
    outputs: Dict[str, str]

    # -----------------------------------------------------
    @validator('procedure_counts')
    def check_csv_exists(cls, v):
        for tag, path in v.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing csv for '{tag}': {path}")
        return v


# ---------------------------------------------------------------------------
# 2.  Helper classes / functions --------------------------------------------
# ---------------------------------------------------------------------------

def parse_age_band(band: str) -> Tuple[int, int]:
    if band.startswith(">="):
        lo = int(band[2:])
        hi = lo + 5
    else:
        lo, hi = map(int, band.split('-'))
    return lo, hi + 1           # make hi exclusive for np.random


def linear_interp(year: int, anchors: Dict[str, float]) -> float:
    points = []
    for k, v in anchors.items():
        if '-' in k:
            a, b = map(int, k.split('-'))
            points.append((a, v))
            points.append((b, v))
        else:
            points.append((int(k), v))
    points.sort()
    xs, ys = zip(*points)
    return float(np.interp(year, xs, ys))


class DistributionFactory:
    """Wraps one or more normal modes into a sampler."""
    def __init__(self, modes: Dict[str, DurabilityMode]):
        self._dists, self._weights = [], []
        for mode in modes.values():
            self._dists.append(norm(loc=mode.mean, scale=mode.sd))
            self._weights.append(mode.weight)
        self._weights = np.array(self._weights) / np.sum(self._weights)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        which = rng.choice(len(self._dists), p=self._weights, size=n)
        out = np.empty(n)
        for i, dist in enumerate(self._dists):
            mask = which == i
            if mask.any():
                out[mask] = dist.rvs(mask.sum(), random_state=rng)
        return out.clip(min=0.1)


# ---------------------------------------------------------------------------
# 3.  Core simulator ---------------------------------------------------------
# ---------------------------------------------------------------------------

class ViVSimulator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.start_forecast = cfg.years['simulate_from']
        self.end_year = cfg.years['end']
        self.rng = np.random.default_rng(cfg.simulation.get('rng_seed'))

        # ----------------------- data -----------------------
        self.tavi_df = pd.read_csv(cfg.procedure_counts['tavi'])
        self.savr_df = pd.read_csv(cfg.procedure_counts['savr'])

        # ------------------- distributions ------------------
        self.dur_tavi = DistributionFactory(cfg.durability['tavi'])
        self.dur_savr_lt70 = DistributionFactory(cfg.durability['savr_bioprosthetic']['lt70'])
        self.dur_savr_gte70 = DistributionFactory(cfg.durability['savr_bioprosthetic']['gte70'])

        low = cfg.survival_curves['low_risk']
        ih  = cfg.survival_curves['int_high_risk']
        self.surv_low = norm(loc=low['median'], scale=low['sd'])
        self.surv_ih  = norm(loc=ih['median'],  scale=ih['sd'])

    # -------------------------------------------------------
    def _risk_mix(self, proc_type: str, year: int) -> Dict[str, float]:
        if proc_type == 'savr':
            return self.cfg.risk_mix['savr']
        for period, mix in self.cfg.risk_mix['tavi'].items():
            a, b = map(int, period.split('-'))
            if a <= year <= b:
                return mix
        raise ValueError(f"No TAVI risk mix defined for year {year}")

    # -------------------------------------------------------
    def run_once(self, run_id: int = 0) -> pd.DataFrame:
        cand: list[Tuple[int, str]] = []
        total_index = 0

        # ---------- loop over TAVI and SAVR counts ----------
        for proc_type, df in (('tavi', self.tavi_df), ('savr', self.savr_df)):
            for _, row in df.iterrows():
                year = int(row.year)
                n    = int(row.count)
                total_index += n
                lo, hi = parse_age_band(row.age_band)
                ages   = self.rng.integers(lo, hi, size=n)

                # survival draw
                mix = self._risk_mix(proc_type, year)
                low_risk_mask = self.rng.random(n) < mix['low']
                surv = np.where(
                    low_risk_mask,
                    self.surv_low.rvs(n, random_state=self.rng),
                    self.surv_ih.rvs(n,  random_state=self.rng)
                )

                # durability draw
                if proc_type == 'tavi':
                    dur = self.dur_tavi.sample(n, self.rng)
                else:
                    dur = np.where(
                        ages < 70,
                        self.dur_savr_lt70.sample(n, self.rng),
                        self.dur_savr_gte70.sample(n, self.rng)
                    )

                fail_year  = year + dur.astype(int)
                death_year = year + surv.astype(int)

                mask = (fail_year <= death_year) & \
                       (self.start_forecast <= fail_year) & \
                       (fail_year <= self.end_year)
                if mask.any():
                    vtype = 'tavi_in_tavi' if proc_type == 'tavi' else 'tavi_in_savr'
                    cand.extend((fy, vtype) for fy in fail_year[mask])

        if not cand:
            log.warning("Run %d produced no ViV candidates!", run_id)
            return pd.DataFrame(columns=['year', 'viv_type', 'count'])

        cand_df = (pd.DataFrame(cand, columns=['year', 'viv_type'])
                     .value_counts()
                     .rename('n_raw')
                     .reset_index())

        total_candidates = cand_df.n_raw.sum()
        log.info("Run %d | index pts %s → viable failures %s",
                 run_id, f"{total_index:,}", f"{total_candidates:,}")

        # ---------- apply penetration -----------------------
        cand_df['penetration'] = cand_df.apply(
            lambda r: linear_interp(int(r.year),
                                    self.cfg.penetration[r.viv_type]),
            axis=1
        )
        cand_df['after_pen'] = (cand_df.n_raw * cand_df.penetration).round().astype(int)

        log.info("Run %d | after penetration %s (%.1f%%)",
                 run_id, f"{cand_df.after_pen.sum():,}",
                 100 * cand_df.after_pen.sum() / total_candidates)

        # ---------- subtract redo-SAVR ----------------------
        rates = self.cfg.redo_rates
        cand_df['redo_rate'] = cand_df.viv_type.map(
            lambda vt: rates.get(
                'savr_after_savr' if vt == 'tavi_in_savr' else 'savr_after_tavi',
                0.0
            )
        )
        cand_df['count'] = (cand_df.after_pen * (1 - cand_df.redo_rate)).round().astype(int)

        log.info("Run %d | after redo-SAVR %s", run_id, f"{cand_df.count.sum():,}")

        return cand_df[['year', 'viv_type', 'count']]


# ---------------------------------------------------------------------------
# 4.  Simulation driver ------------------------------------------------------
# ---------------------------------------------------------------------------

def run_simulation(cfg: Config) -> pd.DataFrame:
    sim = ViVSimulator(cfg)
    runs = []
    for r in range(cfg.simulation['n_runs']):
        runs.append(sim.run_once(run_id=r).assign(run=r))

    all_df = pd.concat(runs, ignore_index=True)
    agg = (all_df.groupby(['year', 'viv_type'])
                 .agg(mean=('count', 'mean'),
                      sd=('count',   'std'))
                 .reset_index())
    return agg


# ---------------------------------------------------------------------------
# 5.  CLI --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--log-level', default='INFO',
                        help='DEBUG | INFO | WARNING | ERROR')
    args = parser.parse_args()

    setup_logging(args.log_level)

    with open(args.config) as fh:
        cfg_dict = yaml.safe_load(fh)
    cfg = Config.parse_obj(cfg_dict)

    out_df = run_simulation(cfg)
    out_path = Path(cfg.outputs['csv'])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_path, index=False)
    log.info("Saved aggregate forecast → %s", out_path)


if __name__ == '__main__':
    main()
