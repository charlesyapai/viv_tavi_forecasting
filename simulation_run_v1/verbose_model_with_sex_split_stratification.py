#!/usr/bin/env python
"""
===============================================================================
  model.py • Monte-Carlo engine for forecasting future ViV-TAVI demand
===============================================================================

This file is intentionally **self-documenting**:

• Section numbering in comments matches the logical pipeline.
• Each public class / function carries a concise doc-string.
• DEBUG-level logging traces every major data-mass reduction step so you can
  sanity-check a run without ever opening a debugger.

Pipeline overview
-----------------
┌─ 1  Config & logging
│
├─ 2  Helpers
│   ├─ parse_age_band()      → convert "65-69" → (65,70)
│   ├─ linear_interp()       → piece-wise linear helper used for penetration
│   └─ DistributionFactory   → wraps one- or multi-modal Normal distributions
│
├─ 3  ViVSimulator
│   ├─ __init__()            → load registry csv, build samplers
│   ├─ _risk_mix()           → map year→low/int-high risk probabilities
│   └─ run_once()            → ONE Monte-Carlo world; returns per-year ViV counts
│
├─ 4  run_simulation()       → N Monte-Carlo worlds → mean ± SD
│
└─ 5  CLI                    → yaml→Config, run_simulation, save CSV



run with:

cd simulation_run_v1

python verbose_model.py --config config.yaml --log-level WARNING

OR

python verbose_model.py --config config.yaml --log-level DEBUG <- more details

python verbose_model.py --config config.yaml --log-level DEBUG | tee run.log

===============================================================================
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import re

import yaml
import numpy as np
import pandas as pd
from pydantic import BaseModel, validator
from scipy.stats import norm


# ════════════════════════════════════════════════════════════════════════════
# 1. Logging & configuration schema
# ════════════════════════════════════════════════════════════════════════════

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with nice timestamp & level column."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


log = logging.getLogger(__name__)


class DurabilityMode(BaseModel):
    """One component of a (possibly multi-modal) durability distribution."""
    mean: float
    sd: float
    weight: float = 1.0  # weight within its mixture


class SurvivalStat(BaseModel):
    median: float
    sd: float


class AgeHazard(BaseModel):
    ref_age: int
    hr_per5: Dict[str, float]        # keys: low / intermediate / high

class Config(BaseModel):
    """
    Pydantic-validated mirror of config.yaml.

    We validate csv paths eagerly so a missing input fails fast.
    """
    years: Dict[str, int]
    procedure_counts: Dict[str, Path]
    durability: Dict[str, Dict[str, DurabilityMode]]

    # NEW:  sex  →  risk  →  SurvivalStat
    survival_curves: Dict[str, Dict[str, SurvivalStat]]

    risk_mix: Dict[str, Dict[str, float] | Dict[str, Dict[str, float]]]
    penetration: Dict[str, Dict[str, float]]
    redo_rates: Dict[str, float] = {}
    redo_procedures_source: Optional[Path] = None

    # NEW:  structured age-hazard
    age_hazard: AgeHazard

    simulation: Dict[str, int | float]
    outputs: Dict[str, str]

    @validator('procedure_counts')
    def _csv_exists(cls, v):
        for tag, path in v.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing csv for '{tag}': {path}")
        return v

    



# ════════════════════════════════════════════════════════════════════════════
# 2. Helper utilities
# ════════════════════════════════════════════════════════════════════════════

def parse_age_band(band: str) -> Tuple[int, int]:
    """
    Convert any common registry age-band label to an *inclusive-exclusive*
    integer range  (lo, hi).

    Accepted patterns
    -----------------
    • "5-9" , "70-74 yrs"                → (5, 10)  , (70, 75)
    • "<5" , "<5yr" , "< 5 yrs"         → (0, 5)
    • ">=80" , ">= 80 yrs" , "80+"      → (80, 85)   (assume 5-year open bin)
    • ">90"                              → (91, 96)  (rare but handled)
    Any stray "yr/yrs/years" or spaces are ignored.
    """
    s = band.strip().lower()
    s = re.sub(r'\s*(yrs?|years?)\s*', '', s)  # strip 'yr', 'yrs', 'years'

    # >=NN  or NN+  ----------------------------------------------------------
    m = re.match(r'(>=\s*(\d+))|(\s*(\d+)\+)', s)
    if m:
        lo = int(m.group(2) or m.group(4))
        return lo, lo + 5

    # >NN  ---------------------------------------------------------------
    m = re.match(r'>\s*(\d+)', s)
    if m:
        lo = int(m.group(1)) + 1
        return lo, lo + 5

    # <NN  ---------------------------------------------------------------
    m = re.match(r'<\s*(\d+)', s)
    if m:
        hi = int(m.group(1))
        return 0, hi

    # NN-MM  -------------------------------------------------------------
    m = re.match(r'(\d+)\s*-\s*(\d+)', s)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2)) + 1       # make hi exclusive
        return lo, hi

    raise ValueError(f"Unrecognised age band label: '{band}'")

def linear_interp(year: int, anchors: Dict[str, float]) -> float:
    """
    Piece-wise linear interpolation between anchor years.

    `anchors` keys may be single years ("2035") OR ranges ("2007-2022").
    """
    pts = []
    for k, v in anchors.items():
        if '-' in k:
            a, b = map(int, k.split('-'))
            pts.append((a, v))
            pts.append((b, v))
        else:
            pts.append((int(k), v))
    pts.sort()
    xs, ys = zip(*pts)
    return float(np.interp(year, xs, ys))


class DistributionFactory:
    """
    Wrap one- or multi-modal Normal distributions into a sampler.

    Accepts *either*

    • a single `DurabilityMode`, or  
    • a dict[str, DurabilityMode] — use 'weight' to build a mixture.
    """
    def __init__(self, modes: DurabilityMode | Dict[str, DurabilityMode]):
        # ---- normalise input to a dict ------------------------------
        if isinstance(modes, DurabilityMode):
            modes = {'_single': modes}

        self._dists, self._weights = [], []
        for m in modes.values():
            self._dists.append(norm(loc=m.mean, scale=m.sd))
            self._weights.append(m.weight)

        # ensure weights sum to 1 even if user entered any numbers
        self._weights = np.asarray(self._weights, dtype=float)
        self._weights /= self._weights.sum()

    # ----------------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return `n` samples from the (mixture) distribution."""
        choice = rng.choice(len(self._dists), p=self._weights, size=n)
        out = np.empty(n)
        for i, dist in enumerate(self._dists):
            mask = choice == i
            if mask.any():
                out[mask] = dist.rvs(mask.sum(), random_state=rng)
        return out.clip(min=0.1)



# ════════════════════════════════════════════════════════════════════════════
# 3. ViVSimulator - patient-level Monte-Carlo world
# ════════════════════════════════════════════════════════════════════════════

class ViVSimulator:
    # ------------------------------------------------------------------
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.start_forecast = cfg.years['simulate_from']
        self.end_year       = cfg.years['end']
        self.rng = np.random.default_rng(cfg.simulation.get('rng_seed'))

        # ---- load registry inputs ------------------------------------
        self.tavi_df = pd.read_csv(cfg.procedure_counts['tavi'])
        self.savr_df = pd.read_csv(cfg.procedure_counts['savr'])
        log.debug("Loaded %s TAVI rows, %s SAVR rows",
                  len(self.tavi_df), len(self.savr_df))

        # ---- durability samplers -------------------------------------
        dur = cfg.durability
        self.dur_tavi       = DistributionFactory(dur['tavi'])
        self.dur_savr_lt70  = DistributionFactory(dur['savr_bioprosthetic']['lt70'])
        self.dur_savr_gte70 = DistributionFactory(dur['savr_bioprosthetic']['gte70'])

        # ---- survival normals (M/F × risk) ---------------------------
        self.surv: Dict[str, Dict[str, norm]] = {}
        for sex_key, bucket in cfg.survival_curves.items():         # 'male' / 'female'
            sex = 'M' if sex_key.lower().startswith('m') else 'F'
            self.surv[sex] = {
                rk: norm(loc=spec.median, scale=spec.sd)            # spec is SurvivalStat
                for rk, spec in bucket.items()                      # 'low' / 'intermediate' / 'high'
            }

        # ---- age-hazard ---------------------------------------------
        self.ref_age = cfg.age_hazard.ref_age
        self.hr_per5 = cfg.age_hazard.hr_per5                       # dict keyed by risk bucket

    # ------------------------------------------------------------------
    def _risk_mix(self, proc_type: str, year: int) -> Dict[str, float]:
        """
        Return {'low': p, 'intermediate': q, 'high': r} for the given
        procedure & year.  Legacy configs containing {'low', 'ih'}
        are auto-upgraded by splitting 'ih' 50 / 50.
        """
        if proc_type == 'savr':
            mix = self.cfg.risk_mix['savr']
        else:
            for period, m in self.cfg.risk_mix['tavi'].items():
                a, b = map(int, period.split('-'))
                if a <= year <= b:
                    mix = m
                    break
            else:
                raise KeyError(f"No TAVI risk mix defined for year {year}")

        # legacy two-bucket → three-bucket
        if 'ih' in mix:
            ih_half = mix['ih'] * 0.5
            mix = {'low': mix['low'], 'intermediate': ih_half, 'high': ih_half}

        # guarantee normalisation
        tot = sum(mix.values())
        return {k: v / tot for k, v in mix.items()}
    

    # ──────────────────────────────────────────────────────────────────────
    # 3.  ViVSimulator - patient-level Monte-Carlo world
    # ──────────────────────────────────────────────────────────────────────
    def run_once(self, run_id: int = 0) -> pd.DataFrame:
        """
        Execute **one** Monte-Carlo replication and return a tidy
        DataFrame with columns = year • viv_type • count.
        """
        # ── scratch containers & running tallies ──────────────────────
        candidates: list[tuple[int, str]] = []
        total_index = 0      # all registry rows this run
        total_kept  = 0      # alive + failed within forecast window
        dbg_row = "Run %d │ %-4s │ y=%d │ n=%4d │"     # helper format

        # ── 3-A  Registry rows → patient-level arrays ─────────────────
        for proc_type, df in (('tavi', self.tavi_df), ('savr', self.savr_df)):
            required = {'year', 'age_band', 'sex', 'count'}
            missing  = required.difference({c.lower() for c in df.columns})
            if missing:
                raise KeyError(f"{proc_type.upper()} csv is missing column(s): "
                               f"{', '.join(missing)}")

            for _, r in df.iterrows():
                year  = int(r['year'])
                n     = int(r['count'])
                lo, hi = parse_age_band(r['age_band'])
                total_index += n

                log.debug(dbg_row + " ages %d-%d", run_id,
                          proc_type.upper(), year, n, lo, hi-1)

                # --- draw ages -------------------------------------------------
                ages = self.rng.integers(lo, hi, size=n)

                # ---- survival  (sex- & risk-specific) ---------------
                mix   = self._risk_mix(proc_type, year)

                # assign each patient a risk bucket
                risks = self.rng.choice(list(mix.keys()),
                                        p=list(mix.values()),
                                        size=n)

                # map registry 'sex' (“Men” / “Women”) → 'M' / 'F'
                sex_flag = 'M' if r['sex'].strip().lower().startswith('m') else 'F'

                surv = np.empty(n)
                for rk in ('low', 'intermediate', 'high'):
                    msk = risks == rk
                    if not msk.any():
                        continue

                    # baseline draw
                    vals = self.surv[sex_flag][rk].rvs(msk.sum(),
                                                       random_state=self.rng)

                    # age-hazard tilt
                    hr = self.hr_per5.get(rk, 1.0)
                    if hr != 1.0:
                        age_factor = ((ages[msk] - self.ref_age) / 5.0)
                        vals /= hr ** age_factor

                    surv[msk] = vals



                # --- durability -------------------------------------------------
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
                    kept = mask.sum()
                    total_kept += kept
                    log.debug(
                        dbg_row + "   → kept %-13s %4d (%.1f%% of row)",
                        run_id,                 # %d
                        proc_type.upper(),      # %-4s
                        year,                   # %d
                        n,                      # %4d  ← row size, not kept
                        vtype,                  # %-13s
                        kept,                   # %4d
                        100 * kept / n          # %.1f
                    )
                              
                    candidates.extend((fy, vtype) for fy in fail_year[mask])

        # if nothing survived the funnel, return an empty frame -----------
        if not candidates:
            log.warning("Run %d produced 0 candidates → empty frame returned",
                        run_id)
            return pd.DataFrame(columns=['year', 'viv_type', 'count'])

        # ── 3-B  Raw ViV candidates (=alive & failed in horizon) ────────
        cand_df = (pd.DataFrame(candidates, columns=['year', 'viv_type'])
                     .value_counts()
                     .rename('n_raw')
                     .reset_index())

        total_cand = cand_df.n_raw.sum()
        log.debug("Run %d │ summary │ index=%s  kept=%s (%.1f%%)  "
                  "raw_candidates=%s",
                  run_id, f'{total_index:,}', f'{total_kept:,}',
                  100*total_kept/total_index if total_index else 0,
                  f'{total_cand:,}')

        for y, vt, k in cand_df[['year', 'viv_type', 'n_raw']].itertuples(False):
            log.debug("Run %d │ raw │ %d │ %-13s │ %s", run_id, y, vt, k)

        # ── 3-C  Apply penetration curves ───────────────────────────────
        cand_df['penetration'] = cand_df.apply(
            lambda r: linear_interp(int(r.year),
                                    self.cfg.penetration[r.viv_type]),
            axis=1
        )
        cand_df['after_pen'] = (cand_df.n_raw * cand_df.penetration)\
                                    .round().astype(int)

        for y, vt, pen, ap in cand_df[['year','viv_type',
                                       'penetration','after_pen']].itertuples(False):
            log.debug("Run %d │ pen │ %d │ %-13s │ %.2f → %s",
                      run_id, y, vt, pen, ap)

        # ── 3-D  Remove redo-SAVR share ─────────────────────────────────
        rates = self.cfg.redo_rates
        cand_df['redo_rate'] = cand_df.viv_type.map(
            lambda vt: rates.get(
                'savr_after_savr' if vt == 'tavi_in_savr' else 'savr_after_tavi',
                0.0
            )
        )
        cand_df['count'] = (cand_df.after_pen * (1 - cand_df.redo_rate))\
                               .round().astype(int)

        for y, vt, ap, rr, c in cand_df[['year','viv_type',
                                         'after_pen','redo_rate','count']]\
                                         .itertuples(False):
            log.debug("Run %d │ redo │ %d │ %-13s │ after_pen=%5d  "
                      "redo=%.1f%% → keep=%5d",
                      run_id, y, vt, ap, 100*rr, c)

        # final tidy return ------------------------------------------------
        return cand_df[['year', 'viv_type', 'count']]

# ════════════════════════════════════════════════════════════════════════════
# 4. Multi-run wrapper
# ════════════════════════════════════════════════════════════════════════════

def run_simulation(cfg: Config) -> pd.DataFrame:
    """Run N Monte-Carlo replications and aggregate mean ± SD per year/type."""
    sim = ViVSimulator(cfg)
    runs = []
    for i in range(cfg.simulation['n_runs']):
        log.info("═════════ Monte-Carlo run %d / %d ═════════",
                 i + 1, cfg.simulation['n_runs'])
        runs.append(sim.run_once(run_id=i).assign(run=i))

    all_df = pd.concat(runs, ignore_index=True)

    # aggregate over runs ---------------------------------------------------
    agg = (all_df.groupby(['year', 'viv_type'])
                 .agg(mean=('count', 'mean'),
                      sd  =('count', 'std'))
                 .reset_index())
    log.info("Aggregated %d runs → %d rows (mean ± SD)",
             cfg.simulation['n_runs'], len(agg))
    return agg


# ════════════════════════════════════════════════════════════════════════════
# 5. Command-line entry-point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='YAML config file')
    parser.add_argument('--log-level', default='INFO',
                        help='DEBUG | INFO | WARNING | ERROR | CRITICAL')
    args = parser.parse_args()

    setup_logging(args.log_level)

    # -- load & validate yaml ----------------------------------------------
    with open(args.config) as fh:
        cfg_dict = yaml.safe_load(fh)
    cfg = Config.parse_obj(cfg_dict)
    log.info("Config loaded; simulation years %s-%s, %d Monte-Carlo runs",
             cfg.years['simulate_from'], cfg.years['end'], cfg.simulation['n_runs'])

    # -- run simulation -----------------------------------------------------
    out_df = run_simulation(cfg)

    # -- save CSV -----------------------------------------------------------
    out_path = Path(cfg.outputs['csv'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log.info("Saved aggregate forecast → %s", out_path)


if __name__ == '__main__':
    main()
