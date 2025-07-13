#!/usr/bin/env python
"""
Monte-Carlo engine for forecasting ViV-TAVI demand.

Usage:
    python model.py --config config.yaml
"""

from __future__ import annotations
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Tuple
from pydantic import BaseModel, validator


# ---------------------------------------------------------------------------
# 1. Configuration schema ----------------------------------------------------
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
    redo_procedures_source: Path
    simulation: Dict[str, int | float]

    # ---------- validators --------------------------------------------------
    @validator('procedure_counts')
    def check_csv_exists(cls, v):
        for k, path in v.items():
            if not Path(path).exists():
                raise FileNotFoundError(f'Missing csv for {k}: {path}')
        return v


# ---------------------------------------------------------------------------
# 2. Helper classes ----------------------------------------------------------
# ---------------------------------------------------------------------------

class DistributionFactory:
    """Create frozen scipy distributions; currently uses Normal."""
    def __init__(self, mode_cfg: Dict[str, DurabilityMode]):
        self.modes: list[Tuple[norm, float]] = []
        for m in mode_cfg.values():
            self.modes.append((norm(loc=m.mean, scale=m.sd), m.weight))
        # normalise weights
        tot = sum(w for _, w in self.modes)
        self.modes = [(d, w / tot) for d, w in self.modes]

    def sample(self, size: int) -> np.ndarray:
        # choose mode per sample, then draw
        mode_choices = np.random.choice(len(self.modes), p=[w for _, w in self.modes], size=size)
        samples = np.empty(size)
        for i, (dist, _) in enumerate(self.modes):
            idx = mode_choices == i
            if idx.any():
                samples[idx] = dist.rvs(idx.sum())
        return samples.clip(min=0.1)  # avoid negatives


def parse_age_band(band: str) -> Tuple[int, int]:
    if band.startswith(">="):
        lo = int(band[2:])
        hi = lo + 5  # open upper band width for sampling
    else:
        lo, hi = map(int, band.split('-'))
    return lo, hi + 1  # make hi exclusive


def linear_interp(year: int, anchor_dict: Dict[str, float]) -> float:
    """Piece-wise linear penetration interpolation."""
    # anchor_dict keys are 'YYYY' or 'YYYY-YYYY'
    anchors = []
    for k, v in anchor_dict.items():
        if '-' in k:
            start, end = map(int, k.split('-'))
            anchors.append((start, v))
            anchors.append((end, v))
        else:
            anchors.append((int(k), v))
    anchors.sort()
    years, vals = zip(*anchors)

    return np.interp(year, years, vals)


# ---------------------------------------------------------------------------
# 3. Core patient-level simulation ------------------------------------------
# ---------------------------------------------------------------------------

class ViVSimulator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.start_forecast = cfg.years['simulate_from']
        self.end_year = cfg.years['end']
        self.rng = np.random.default_rng(cfg.simulation.get('rng_seed', None))

        # pre-load csvs
        self.tavi_df = pd.read_csv(cfg.procedure_counts['tavi'])
        self.savr_df = pd.read_csv(cfg.procedure_counts['redo_savr'])
        self.redo_df = pd.read_csv(cfg.redo_procedures_source)

        # build distributions
        self.dur_savr_lt70 = DistributionFactory(cfg.durability['savr_bioprosthetic']['lt70'])
        self.dur_savr_gte70 = DistributionFactory(cfg.durability['savr_bioprosthetic']['gte70'])
        self.dur_tavi = DistributionFactory(cfg.durability['tavi'])

        # survival (Normal proxy; can swap to lifetable later)
        surv_low = cfg.survival_curves['low_risk']
        self.surv_low = norm(loc=surv_low['median'], scale=surv_low['sd'])
        surv_ih = cfg.survival_curves['int_high_risk']
        self.surv_ih = norm(loc=surv_ih['median'], scale=surv_ih['sd'])

    # ---------------------------------------------------------
    def run_once(self) -> pd.DataFrame:
        candidates = []  # each row: year, viv_type

        # ---------- loop over input counts --------------------
        for proc_type, df in [('tavi', self.tavi_df), ('savr', self.savr_df)]:
            for _, row in df.iterrows():
                year = int(row['year'])
                n = int(row['count'])
                lo, hi = parse_age_band(row['age_band'])
                ages = self.rng.integers(lo, hi, size=n)
                sex = row['sex']   # not used yet, but stored for extensibility

                # --- sample risk group --------------------------
                risk_mix = self._risk_mix(proc_type, year)
                risk_is_low = self.rng.uniform(0, 1, size=n) < risk_mix['low']
                surv_years = np.where(
                    risk_is_low,
                    self.surv_low.rvs(n, random_state=self.rng),
                    self.surv_ih.rvs(n, random_state=self.rng)
                ).clip(min=0.1)

                # --- sample durability -------------------------
                if proc_type == 'tavi':
                    durability = self.dur_tavi.sample(n)
                else:  # SAVR: choose age-dependent distribution
                    durability = np.where(
                        ages < 70,
                        self.dur_savr_lt70.sample(n),
                        self.dur_savr_gte70.sample(n)
                    )

                event_years = year + durability.astype(int)
                death_years = year + surv_years.astype(int)

                alive_at_failure = event_years <= death_years
                event_years = event_years[alive_at_failure]

                # keep only yrs inside forecast horizon
                mask = (event_years >= self.start_forecast) & (event_years <= self.end_year)
                event_years = event_years[mask]
                if len(event_years) == 0:
                    continue

                viv_type = 'tavi_in_tavi' if proc_type == 'tavi' else 'tavi_in_savr'
                candidates.extend([(y, viv_type) for y in event_years])

        cand_df = pd.DataFrame(candidates, columns=['year', 'viv_type'])
        out = cand_df.value_counts().rename('n_raw').reset_index()

        # ---- apply penetration -------------------------------
        out['penetration'] = out.apply(
            lambda r: linear_interp(int(r['year']), self.cfg.penetration[r['viv_type']]), axis=1
        )
        out['n_after_pen'] = (out['n_raw'] * out['penetration']).round().astype(int)

        # ---- subtract redo-surgery counts --------------------
        redo_map = {
            'tavi_in_savr': 'savr_after_savr',
            'tavi_in_tavi': 'savr_after_tavi'
        }
        redo_subset = self.redo_df[self.redo_df['redo_type'].isin(redo_map.values())]
        out = out.merge(
            redo_subset.rename(columns={'redo_type': 'viv_type'}),
            how='left',
            left_on=['year', 'viv_type'],
            right_on=['year', 'viv_type']
        )




        
        out['count'] = out['n_after_pen'] - out['count'].fillna(0).astype(int)
        out.loc[out['count'] < 0, 'count'] = 0
        return out[['year', 'viv_type', 'count']]

    # ---------------------------------------------------------
    def _risk_mix(self, proc_type: str, year: int) -> Dict[str, float]:
        if proc_type == 'savr':
            return self.cfg.risk_mix['savr']
        # TAVI – need to find which block contains year
        for period, mix in self.cfg.risk_mix['tavi'].items():
            start, end = map(int, period.split('-'))
            if start <= year <= end:
                return mix
        raise ValueError(f'No risk mix for TAVI year {year}')


def run_simulation(cfg: Config) -> pd.DataFrame:
    mc = ViVSimulator(cfg)
    runs = []
    for _ in range(cfg.simulation['n_runs']):
        runs.append(mc.run_once().assign(run=_))
    all_df = pd.concat(runs, ignore_index=True)
    agg = (all_df.groupby(['year', 'viv_type'])
                  .agg(mean=('count', 'mean'),
                       sd=('count', 'std'))
                  .reset_index())
    return agg


# ---------------------------------------------------------------------------
# 4. CLI entry-point ---------------------------------------------------------
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        cfg_dict = yaml.safe_load(fh)
    cfg = Config.parse_obj(cfg_dict)

    out_df = run_simulation(cfg)
    out_path = Path(cfg_dict['outputs']['csv'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f'Saved forecast to {out_path}')


if __name__ == '__main__':
    main()
