"""
model_v8.py — Demography-driven Monte-Carlo model for ViV-TAVI forecasting

Overview
--------
This script implements the full pipeline you requested, split into clearly
sectioned components and with verbose, step-by-step logging. It is designed
to be executed either with a configuration file:

    $ python model_v8.py --config path/to/config.yaml --scenario korea_v8

…or with no CLI arguments, in which case it uses the **defaults defined at the
top of this file** (see: DEFAULT_CONFIG_PATH and DEFAULT_SCENARIO).

Major steps
-----------
1) Load configuration and inputs
2) Build annual **Year x AgeBand x Sex → Population** table from the provided
   aggregate age groups and senior sex ratios (or directly if already provided)
3) Learn **agexsex-specific baseline rates** for TAVI, SAVR, redo-SAVR from
   historical registry data (pre-COVID windows)
4) Apply **time multipliers (anchors)** to reflect adoption/substitution
5) Project **index counts** per year via:  Population x Rate x Multiplier
6) Prepare Monte-Carlo inputs (age allocation, durability, survival, penetration)
7) Run Monte-Carlo to generate ViV candidates and treatments
8) Apply **redo-SAVR annual targets** to subtract competing redo-open cases
9) Write **tables**, **QC**, **figures** (with `viz_v8.py`), and a Markdown
   **run report** describing each step with links to the outputs

Design notes
------------
• Pure Python + NumPy + pandas + matplotlib. No SciPy dependency.
• Logging is verbose and saved to logs/run.log; a narrative report is saved to
  logs/report.md to aid inspection and debugging.
• All functions carry docstrings describing WHAT they do and WHEN they are used.
• Plotting/formatting is separated into `viz_v8.py`.


Can run with:
```
python simulation_run_v1/models/model_v8.py \
  --config simulation_run_v1/configs/model_v8_config.yaml \
  --scenario korea_demo
```


"""

from __future__ import annotations 

# ──────────────────────────────────────────────────────────────────────
# Pathing Fixes
# ──────────────────────────────────────────────────────────────────────


from pathlib import Path

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "model_v8_config.yaml")
DEFAULT_SCENARIO = "korea_demo"



# ──────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────


import argparse
import dataclasses
import logging
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # PyYAML; required to parse YAML configs
except Exception:
    yaml = None

# Optional plotting (kept in a separate module)
try:
    import simulation_run_v1.models.viz_v8 as viz
except Exception:
    viz = None










# ──────────────────────────────────────────────────────────────────────
# Utilities: filesystem, logging, and helper math
# ──────────────────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists and return its Path.
    WHEN USED: whenever we need to guarantee an output folder exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(outdir: Path, level: str = "INFO") -> logging.Logger:
    """
    Configure logging (console + file) and return the module logger.
    WHEN USED: at program start to capture step-by-step progress.
    """
    ensure_dir(outdir / "logs")
    logger = logging.getLogger("model_v8")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers in case of repeated runs within same interpreter
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(outdir / "logs" / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def write_report_md(outdir: Path, sections: List[Tuple[str, str]]) -> None:
    """
    Write a Markdown report (logs/report.md) combining human-readable sections.
    WHEN USED: at the end of a run to document each step’s key points, file paths,
    and checks so reviewers can trace the logic.
    """
    ensure_dir(outdir / "logs")
    md_path = outdir / "logs" / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ViV-TAVI Monte-Carlo — Run Report (v8)\n\n")
        for title, body in sections:
            f.write(f"## {title}\n\n{body}\n\n")
    # Also mirror to console hint
    print(f"[report] Wrote {md_path}")


def anchored_linear_interpolator(anchors: Dict[int, float], years: Iterable[int]) -> Dict[int, float]:
    """
    Build a {year → multiplier} from sparse anchor points using piecewise-linear interpolation.
    WHEN USED: to turn simple anchor dictionaries in the config into annual multipliers.
    """
    if not anchors:
        return {int(y): 1.0 for y in years}
    xs = sorted(int(k) for k in anchors.keys())
    ys = [float(anchors[int(k)]) for k in xs]
    y_min, y_max = min(years), max(years)
    # Fill outside anchors with edge values
    result = {}
    for y in range(y_min, y_max + 1):
        if y <= xs[0]:
            result[y] = ys[0]
        elif y >= xs[-1]:
            result[y] = ys[-1]
        else:
            # find segment
            for i in range(len(xs) - 1):
                if xs[i] <= y <= xs[i + 1]:
                    x0, x1 = xs[i], xs[i + 1]
                    y0, y1 = ys[i], ys[i + 1]
                    t = (y - x0) / (x1 - x0) if x1 != x0 else 0.0
                    result[y] = y0 + t * (y1 - y0)
                    break
    return result


def choose_rng(seed: Optional[int]) -> np.random.Generator:
    """
    Return a NumPy Generator with a fixed seed (if provided).
    WHEN USED: to make simulation runs reproducible across machines, when desired.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


# ──────────────────────────────────────────────────────────────────────
# Configuration dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class LearningWindows:
    """Years to learn baseline rates from (pre-COVID typically)."""
    tavi: Tuple[int, int] = (2015, 2019)
    savr: Tuple[int, int] = (2012, 2019)
    redo_savr: Tuple[int, int] = (2012, 2019)


@dataclasses.dataclass
class PopulationInput:
    """
    Paths and mode for building YearxAgeBandxSex → Population.

    mode="combined": a single CSV containing BOTH age totals and senior sex ratios.
        Expected structure (wide):
            - Rows labeled with age groups: "0-14","15-24","25-49","50-64","≥65","≥70","≥75","≥85"
            - A row "Total (n)" may appear (ignored if present)
            - A block titled "Sex ratio (number of men per 100 women)" with rows:
              "≥65","≥70","≥75","≥85" and the same year columns
            - Columns are years, e.g. 2022,2023,2024,2025,2030,2040,2050
        We automatically derive disjoint bands: 65-69, 70-74, 75-84, 85+
        and split male/female using the sex ratios.

    mode="separate": two CSVs:
        age_totals_csv: rows are age groups; columns are years (integers)
        sex_ratio_csv:  senior "≥" groups; columns are years; values are men per 100 women

    WHEN USED: Step 1 to build demography table.
    """
    mode: Literal["combined", "separate"] = "combined"
    combined_csv: Optional[str] = None
    age_totals_csv: Optional[str] = None
    sex_ratio_csv: Optional[str] = None


@dataclasses.dataclass
class DurabilityMode:
    """A single normal mode for durability (in years) with probability weight."""
    mean: float
    sd: float
    weight: float


@dataclasses.dataclass
class DurabilitySpec:
    """
    Durability specification for TAVI and SAVR.

    SAVR can be age-dependent with a threshold (e.g., 70):
        - under_thresh: list[DurabilityMode]
        - overeq_thresh: list[DurabilityMode]
        - age_threshold: int (e.g., 70)
    TAVI is a simple list of modes (e.g., 20% @ 4y, 80% @ 11.5y).
    """
    tavi_modes: List[DurabilityMode]
    savr_under_modes: List[DurabilityMode]
    savr_overeq_modes: List[DurabilityMode]
    savr_age_threshold: int = 70


@dataclasses.dataclass
class SurvivalCurves:
    """
    Survival curves for risk classes. CSVs must have columns: time_years, survival (0-1).

    risk_mix_anchors: how the index risk mix changes by year; dictionary mapping years to
    proportions of "low" risk; the remainder is intermediate+high ("ih"). Separate maps
    for TAVI and SAVR are supported.

    Example:
        risk_mix_anchors = {
            "tavi": {2014: 0.00, 2019: 0.33, 2024: 0.50, 2035: 0.50},
            "savr": {2012: 0.85, 2035: 0.85}
        }
    """
    low_csv: str
    ih_csv: str
    risk_mix_anchors: Dict[str, Dict[int, float]]  # keys: "tavi","savr" → {year → low_fraction}


@dataclasses.dataclass
class PenetrationSpec:
    """
    Penetration anchors for treatment at failure.

    anchors_tavr_in_tavr: year → fraction among failed TAVR treated with TAVR-in-TAVR
    anchors_tavr_in_savr: year → fraction among failed SAVR treated with TAVR-in-SAVR
    The complements go to other treatments (e.g., surgical redo, medical therapy).
    """
    anchors_tavr_in_tavr: Dict[int, float]
    anchors_tavr_in_savr: Dict[int, float]


@dataclasses.dataclass
class RedoSpec:
    """
    Redo-SAVR subtraction policy.

    mode: "from_projection" uses the projected redo-SAVR counts (Step 3) as targets;
          "from_file" reads a CSV with columns: year,count and uses that.
    fill:  "linear" | "forward" | "zero" — how to fill missing years (inclusive range).
    yrange: (start,end) for filling.
    """
    mode: Literal["from_projection", "from_file"] = "from_projection"
    file: Optional[str] = None
    fill: Literal["linear", "forward", "zero"] = "linear"
    yrange: Tuple[int, int] = (2025, 2040)


@dataclasses.dataclass
class SimulationSpec:
    """Parameters controlling the Monte-Carlo runs."""
    n_runs: int = 20
    min_dur_years: float = 1.0
    start_year: int = 2025
    end_year: int = 2040
    seed: Optional[int] = None


@dataclasses.dataclass
class Config:
    """
    Master configuration for a scenario.

    WHEN USED: loaded at program start and passed across steps.
    """
    experiment_name: str
    output_dir: str

    population_input: PopulationInput
    learning_windows: LearningWindows

    # Registry (historical) counts (long format preferred)
    registry_tavi_csv: str
    registry_savr_csv: str
    registry_redo_csv: Optional[str] = None  # optional if mode=from_projection

    # Rate multipliers (anchors)
    rate_multipliers: Dict[str, Dict[int, float]] = dataclasses.field(default_factory=dict)

    # Monte-Carlo specs
    durability: DurabilitySpec = None
    survival: SurvivalCurves = None
    penetration: PenetrationSpec = None
    redo: RedoSpec = None
    simulation: SimulationSpec = None


# ──────────────────────────────────────────────────────────────────────
# Reading configuration
# ──────────────────────────────────────────────────────────────────────

def load_config(path: Path, scenario: str) -> Config:
    """
    Load YAML config and construct a Config dataclass for the given scenario key.
    Paths inside the YAML (e.g., "data/...") are resolved **relative to the project root**
    (the parent folder of 'models/' and 'configs/'), so you can keep tidy paths like
    "data/registry_tavi.csv" regardless of where you run the script from.

    WHEN USED: at program start.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required. Please `pip install pyyaml`.")

    # --- NEW: project root resolver -----------------------------------------
    # We treat the parent of 'models/' (i.e., simulation_run_v1/) as PROJECT_ROOT.
    # Example: simulation_run_v1/models/model_v8.py → PROJECT_ROOT = simulation_run_v1/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def _resolve_from_root(maybe_path: Optional[str]) -> Optional[str]:
        """
        Resolve a file/directory path relative to PROJECT_ROOT, unless it's absolute or None.
        Accepts None/"null"/"" and returns None.
        """
        if maybe_path is None:
            return None
        s = str(maybe_path).strip()
        if s == "" or s.lower() in {"none", "null"}:
            return None
        p = Path(s)
        return str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())
    # -------------------------------------------------------------------------

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if scenario not in raw:
        raise KeyError(f"Scenario '{scenario}' not found in {path}.")

    cfg = raw[scenario]

    def _dm_list(lst):
        return [DurabilityMode(**d) for d in lst]

    # --- NEW: resolve all file/dir paths from PROJECT_ROOT -------------------
    # output_dir
    output_dir = _resolve_from_root(cfg["output_dir"])

    # population_input (handle both modes but safely resolve any file fields)
    pop_cfg = dict(cfg["population_input"])  # shallow copy
    if "combined_csv" in pop_cfg:
        pop_cfg["combined_csv"] = _resolve_from_root(pop_cfg.get("combined_csv"))
    if "age_totals_csv" in pop_cfg:
        pop_cfg["age_totals_csv"] = _resolve_from_root(pop_cfg.get("age_totals_csv"))
    if "sex_ratio_csv" in pop_cfg:
        pop_cfg["sex_ratio_csv"] = _resolve_from_root(pop_cfg.get("sex_ratio_csv"))

    # registries
    reg_tavi = _resolve_from_root(cfg["registry"]["tavi"])
    reg_savr = _resolve_from_root(cfg["registry"]["savr"])
    reg_redo = _resolve_from_root(cfg["registry"].get("redo_savr"))

    # survival csvs
    surv_low = _resolve_from_root(cfg["survival"]["low_csv"])
    surv_ih  = _resolve_from_root(cfg["survival"]["ih_csv"])

    # redo file (optional)
    redo_cfg = dict(cfg.get("redo", {}))
    if "file" in redo_cfg:
        redo_cfg["file"] = _resolve_from_root(redo_cfg.get("file"))
    # -------------------------------------------------------------------------

    conf = Config(
        experiment_name=cfg["experiment_name"],
        output_dir=output_dir,
        population_input=PopulationInput(**pop_cfg),
        learning_windows=LearningWindows(**cfg.get("learning_windows", {})),
        registry_tavi_csv=reg_tavi,
        registry_savr_csv=reg_savr,
        registry_redo_csv=reg_redo,
        rate_multipliers=cfg.get("rate_multipliers", {}),
        durability=DurabilitySpec(
            tavi_modes=_dm_list(cfg["durability"]["tavi"]),
            savr_under_modes=_dm_list(cfg["durability"]["savr"]["under70"]),
            savr_overeq_modes=_dm_list(cfg["durability"]["savr"]["overeq70"]),
            savr_age_threshold=int(cfg["durability"]["savr"].get("age_threshold", 70)),
        ),
        survival=SurvivalCurves(
            low_csv=surv_low,
            ih_csv=surv_ih,
            risk_mix_anchors=cfg["survival"]["risk_mix_anchors"],
        ),
        penetration=PenetrationSpec(
            anchors_tavr_in_tavr=cfg["penetration"]["tavr_in_tavr"],
            anchors_tavr_in_savr=cfg["penetration"]["tavr_in_savr"],
        ),
        redo=RedoSpec(**redo_cfg),
        simulation=SimulationSpec(**cfg.get("simulation", {})),
    )
    return conf


# ──────────────────────────────────────────────────────────────────────
# Step 1 — Build Year x AgeBand x Sex → Population
# ──────────────────────────────────────────────────────────────────────

TARGET_BANDS = ["50-64", "65-69", "70-74", "75-84", "85+"]

def _disjoint_from_ge_counts(ge: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """
    Turn overlapping ≥-style totals into disjoint bands.

    Input:
        ge = {
          '>=65': pd.Series(year→total),
          '>=70': ...,
          '>=75': ...,
          '>=85': ...
        }
    Output:
        {
          '65-69': ...,
          '70-74': ...,
          '75-84': ...,
          '85+':   ...
        }
    WHEN USED: Step 1, internal helper.
    """
    out = {}
    ge65 = ge.get("≥65") or ge.get(">=65")
    ge70 = ge.get("≥70") or ge.get(">=70")
    ge75 = ge.get("≥75") or ge.get(">=75")
    ge85 = ge.get("≥85") or ge.get(">=85")
    if ge65 is None or ge70 is None or ge75 is None or ge85 is None:
        raise ValueError("Missing one of ≥65, ≥70, ≥75, ≥85 in senior totals.")
    out["65-69"] = ge65 - ge70
    out["70-74"] = ge70 - ge75
    out["75-84"] = ge75 - ge85
    out["85+"]   = ge85.copy()
    return out


def _split_by_sex(total: pd.Series, sr: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Split totals into male/female using sex ratio (men per 100 women).

    total: pd.Series(year → total count)
    sr:    pd.Series(year → men per 100 women)

    Returns: (male_series, female_series)

    WHEN USED: Step 1 for senior bands where we have sex ratios.
    """
    r = sr / 100.0
    female = total / (1.0 + r)
    male = total - female
    return male, female


def read_population_table(cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    """
    Build a tidy population DataFrame with columns: year, age_band, sex, population.

    • Handles either a combined CSV (age totals + sex ratios) or two separate CSVs.
    • Derives disjoint senior bands (65-69, 70-74, 75-84, 85+).
    • Splits senior bands by sex using the provided sex ratios.
    • For 50-64, if no sex ratio is available, assumes 1:1 (can be replaced later).

    WHEN USED: Step 1.
    """
    pi = cfg.population_input
    if pi.mode == "combined":
        if not pi.combined_csv:
            raise ValueError("population_input.combined_csv is required for mode='combined'.")
        df = pd.read_csv(pi.combined_csv)
        # Normalize header names
        df.columns = [str(c).strip() for c in df.columns]
        # Identify year columns (numeric)
        year_cols = [c for c in df.columns if str(c).isdigit()]
        # First block: age totals
        # We locate these by known labels in first column
        key_col = df.columns[0]
        df[key_col] = df[key_col].astype(str).str.strip()

        # Split into two parts
        # Find the row that starts the sex ratio block
        sr_start_idx = df[df[key_col].str.contains("Sex ratio", case=False, na=False)].index
        if len(sr_start_idx) == 0:
            raise ValueError("Combined population CSV must contain a 'Sex ratio' section header.")
        sr_start = sr_start_idx[0]

        totals_block = df.loc[:sr_start-1].copy()
        ratios_block = df.loc[sr_start+1:].copy()

        # Extract relevant rows
        # Age totals of interest: 50-64, ≥65, ≥70, ≥75, ≥85 (0-14/15-24/25-49 are ignored for index)
        totals_block[key_col] = totals_block[key_col].str.replace(" ", "")
        need_labels = ["50-64", "≥65", "≥70", "≥75", "≥85", ">=65", ">=70", ">=75", ">=85"]
        totals = {}
        for lbl in need_labels:
            row = totals_block[totals_block[key_col] == lbl]
            if not row.empty:
                totals[lbl] = row[year_cols].squeeze().astype(float)
        if not any(k in totals for k in ("≥65",">=65")):
            raise ValueError("Missing ≥65 senior totals in the combined CSV.")

        disjoint_totals = _disjoint_from_ge_counts(totals)

        # 50-64 total (no sex ratio provided)
        if "50-64" in totals:
            total_50_64 = totals["50-64"]
        else:
            raise ValueError("Missing 50-64 totals in the combined CSV.")

        # Sex ratios for seniors
        ratios_block[key_col] = ratios_block[key_col].str.replace(" ", "")
        sr_ge = {}
        for lbl in ["≥65","≥70","≥75","≥85",">=65",">=70",">=75",">=85"]:
            row = ratios_block[ratios_block[key_col] == lbl]
            if not row.empty:
                sr_ge[lbl] = row[year_cols].squeeze().astype(float)

        if not any(k in sr_ge for k in ("≥65",">=65")):
            raise ValueError("Missing senior sex ratio rows (≥65, ≥70, ≥75, ≥85).")

        # Build male/female series for seniors
        mf = {}
        for band, series in disjoint_totals.items():
            # Map band back to the corresponding ≥ label boundaries to get SR for endpoints
            # We have SR for ≥65, ≥70, ≥75, ≥85 only; use SR at the lower bound of each band
            if band == "65-69":
                sr = sr_ge.get("≥65") or sr_ge.get(">=65")
            elif band == "70-74":
                sr = sr_ge.get("≥70") or sr_ge.get(">=70")
            elif band == "75-84":
                sr = sr_ge.get("≥75") or sr_ge.get(">=75")
            elif band == "85+":
                sr = sr_ge.get("≥85") or sr_ge.get(">=85")
            else:
                raise AssertionError("Unexpected band.")
            m, f = _split_by_sex(series, sr)
            mf[band] = {"M": m, "F": f}

        # 50-64: assume 1:1 if no ratio provided
        m_50_64 = total_50_64 * 0.5
        f_50_64 = total_50_64 - m_50_64

        # Assemble tidy table
        rows = []
        for band in ["50-64", "65-69", "70-74", "75-84", "85+"]:
            if band == "50-64":
                for y, tot in total_50_64.items():
                    rows.append({"year": int(y), "age_band": band, "sex": "M", "population": float(m_50_64.loc[y])})
                    rows.append({"year": int(y), "age_band": band, "sex": "F", "population": float(f_50_64.loc[y])})
            else:
                m = mf[band]["M"]
                f = mf[band]["F"]
                for y, v in m.items():
                    rows.append({"year": int(y), "age_band": band, "sex": "M", "population": float(v)})
                for y, v in f.items():
                    rows.append({"year": int(y), "age_band": band, "sex": "F", "population": float(v)})
        pop = pd.DataFrame(rows)

    else:  # separate
        if not (pi.age_totals_csv and pi.sex_ratio_csv):
            raise ValueError("population_input.age_totals_csv and sex_ratio_csv are required for mode='separate'.")
        age_tot = pd.read_csv(pi.age_totals_csv)
        sexrat  = pd.read_csv(pi.sex_ratio_csv)
        # Expect first column is 'age_group' and others are years
        age_key = age_tot.columns[0]
        year_cols = [c for c in age_tot.columns if str(c).isdigit()]
        age_tot[age_key] = age_tot[age_key].astype(str).str.strip()

        ge = {}
        for lbl in ["≥65","≥70","≥75","≥85",">=65",">=70",">=75",">=85"]:
            row = age_tot[age_tot[age_key] == lbl]
            if not row.empty:
                ge[lbl] = row[year_cols].squeeze().astype(float)
        disjoint_totals = _disjoint_from_ge_counts(ge)

        row_50_64 = age_tot[age_tot[age_key].str.replace(" ", "") == "50-64"]
        if row_50_64.empty:
            raise ValueError("50-64 totals missing.")
        total_50_64 = row_50_64[year_cols].squeeze().astype(float)

        # Sex ratios
        sr_key = sexrat.columns[0]
        sexrat[sr_key] = sexrat[sr_key].astype(str).str.strip()
        sr_ge = {}
        for lbl in ["≥65","≥70","≥75","≥85",">=65",">=70",">=75",">=85"]:
            row = sexrat[sexrat[sr_key] == lbl]
            if not row.empty:
                sr_ge[lbl] = row[year_cols].squeeze().astype(float)

        mf = {}
        for band, series in disjoint_totals.items():
            if band == "65-69":
                sr = sr_ge.get("≥65") or sr_ge.get(">=65")
            elif band == "70-74":
                sr = sr_ge.get("≥70") or sr_ge.get(">=70")
            elif band == "75-84":
                sr = sr_ge.get("≥75") or sr_ge.get(">=75")
            elif band == "85+":
                sr = sr_ge.get("≥85") or sr_ge.get(">=85")
            m, f = _split_by_sex(series, sr)
            mf[band] = {"M": m, "F": f}

        m_50_64 = total_50_64 * 0.5
        f_50_64 = total_50_64 - m_50_64

        rows = []
        for band in ["50-64", "65-69", "70-74", "75-84", "85+"]:
            if band == "50-64":
                for y, tot in total_50_64.items():
                    rows.append({"year": int(y), "age_band": band, "sex": "M", "population": float(m_50_64.loc[y])})
                    rows.append({"year": int(y), "age_band": band, "sex": "F", "population": float(f_50_64.loc[y])})
            else:
                m = mf[band]["M"]
                f = mf[band]["F"]
                for y, v in m.items():
                    rows.append({"year": int(y), "age_band": band, "sex": "M", "population": float(v)})
                for y, v in f.items():
                    rows.append({"year": int(y), "age_band": band, "sex": "F", "population": float(v)})
        pop = pd.DataFrame(rows)

    # Interpolate annual values if needed (e.g., 2025,2030,2040,2050 → fill 2026…2039)
    all_years = sorted({int(y) for y in pop["year"].unique()})
    min_y, max_y = min(all_years), max(all_years)
    logger.info(f"[Step 1] Population table loaded (sparse). Years span {min_y}-{max_y}. Interpolating annually.")
    # Pivot and interpolate by age_bandxsex
    pivot = pop.pivot_table(index=["age_band","sex"], columns="year", values="population", aggfunc="sum")
    full_years = list(range(min_y, max_y + 1))
    pivot = pivot.reindex(columns=full_years)
    pivot = pivot.interpolate(axis=1, method="linear", limit_direction="both")
    pop_full = pivot.stack().reset_index().rename(columns={0: "population", "level_2": "year"})
    pop_full["year"] = pop_full["year"].astype(int)
    logger.info(f"[Step 1] Population table prepared: {len(pop_full):,} rows "
                f"({len(TARGET_BANDS)} age bands x 2 sexes x {len(full_years)} years).")
    return pop_full


# ──────────────────────────────────────────────────────────────────────
# Step 2 — Learn agexsex baseline rates from registry and Step 3 — Project
# ──────────────────────────────────────────────────────────────────────

def read_registry_long(csv_path: str, value_col: str = "count") -> pd.DataFrame:
    """
    Read a registry CSV in either long or wide format and return a long DataFrame
    with columns: year, age_band, sex, count.

    • Long format accepted: year, age_band, sex, count
    • Wide format accepted: first col age_band (and optional sex), remaining cols = years

    WHEN USED: Step 2 input loading for TAVI, SAVR, redo-SAVR.
    """
    df = pd.read_csv(csv_path)
    cols = [c.lower() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def normcol(name): return lower_map.get(name, name)

    if set(["year","age_band", "count"]).issubset(cols):
        # Long format; ensure sex exists
        if "sex" not in cols:
            df["sex"] = "ALL"
        return df.rename(columns={normcol("year"): "year",
                                  normcol("age_band"): "age_band",
                                  normcol(value_col): "count",
                                  normcol("sex"): "sex"})
    else:
        # Wide format: first col is age band (and maybe sex if second col is 'sex')
        first = df.columns[0]
        if df.columns[1].lower() in ("sex","gender"):
            id_cols = [df.columns[0], df.columns[1]]
        else:
            id_cols = [df.columns[0]]
            df["sex"] = "ALL"
            id_cols.append("sex")
        long = df.melt(id_vars=id_cols, var_name="year", value_name="count")
        long["year"] = long["year"].astype(int)
        long = long.rename(columns={df.columns[0]: "age_band", id_cols[1]: "sex"})
        return long[["year","age_band","sex","count"]]


def subset_years(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Filter rows between [start, end] years inclusive. WHEN USED: learning windows."""
    return df[(df["year"] >= start) & (df["year"] <= end)].copy()


def learn_baseline_rates(
    registry: pd.DataFrame,
    pop: pd.DataFrame,
    window: Tuple[int, int],
    logger: logging.Logger,
    procedure_name: str,
) -> pd.DataFrame:
    """
    Compute baseline rates per (age_band, sex) as the average of annual crude rates
    over the learning window (pre-COVID recommended).

    Returns a DataFrame: age_band, sex, rate

    WHEN USED: Step 2.
    """
    reg = subset_years(registry, *window)
    # Align with population by merging on year, age_band, sex
    merged = reg.merge(pop, on=["year","age_band","sex"], how="left", validate="many_to_one")
    merged["rate"] = merged["count"] / merged["population"].replace(0, np.nan)
    # Average across years for each (age_band, sex)
    rates = (merged.groupby(["age_band","sex"])["rate"]
             .mean().reset_index())
    rates["procedure"] = procedure_name
    logger.info(f"[Step 2] Learned baseline rates for {procedure_name}: "
                f"{len(rates)} agexsex cells")
    return rates


def project_index_counts(
    rates: pd.DataFrame,
    pop: pd.DataFrame,
    anchors: Dict[int, float],
    years: Iterable[int],
    logger: logging.Logger,
    procedure_name: str,
) -> pd.DataFrame:
    """
    Project index counts by year via:  population x rate x multiplier(year).

    Returns: DataFrame with columns year, age_band, sex, projected, procedure

    WHEN USED: Step 3.
    """
    mult = anchored_linear_interpolator(anchors or {}, years)
    df = pop[pop["year"].isin(list(years))].merge(rates, on=["age_band","sex"], how="left")
    df["rate"] = df["rate"].fillna(0.0)
    df["multiplier"] = df["year"].map(mult).astype(float)
    df["projected"] = df["population"] * df["rate"] * df["multiplier"]
    df["procedure"] = procedure_name
    logger.info(f"[Step 3] Projected {procedure_name} index counts for {len(years)} years.")
    return df[["year","age_band","sex","procedure","projected"]]


# ──────────────────────────────────────────────────────────────────────
# Step 4 — Monte-Carlo building blocks: durability and survival sampling
# ──────────────────────────────────────────────────────────────────────

def sample_mixture_norm(n: int, modes: List[DurabilityMode], rng: np.random.Generator, clip_min: float) -> np.ndarray:
    """
    Sample 'n' durations from a mixture of normal modes (in years), clipped at clip_min.

    WHEN USED: to generate durability times for TAVI and SAVR index patients.
    """
    if n <= 0:
        return np.zeros((0,), dtype=float)
    weights = np.array([m.weight for m in modes], dtype=float)
    weights = weights / weights.sum()
    # Choose a mode for each draw
    choices = rng.choice(len(modes), size=n, p=weights)
    means = np.array([modes[i].mean for i in choices], dtype=float)
    sds   = np.array([modes[i].sd for i in choices], dtype=float)
    draws = rng.normal(loc=means, scale=sds).astype(float)
    # Clip at minimum (avoid zero/negative durations)
    draws = np.maximum(draws, clip_min)
    return draws


def load_survival_curve(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a survival curve from CSV with columns: time_years, survival (0-1).
    Returns (time, survival) arrays sorted by time ascending.

    WHEN USED: Step 4 for mortality sampling.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("time_years")
    return df["time_years"].to_numpy(dtype=float), df["survival"].to_numpy(dtype=float)


def sample_from_survival_curve(
    n: int,
    time: np.ndarray,
    surv: np.ndarray,
    rng: np.random.Generator,
    t_max: float = 50.0
) -> np.ndarray:
    """
    Inverse-transform sampling from a discrete survival curve S(t).

    For each draw u~U(0,1), we return the earliest time t at which S(t) <= u.
    If S(t) > u for all provided t, we return t_max.

    WHEN USED: Step 4 to generate survival times for a risk class.
    """
    if n <= 0:
        return np.zeros((0,), dtype=float)
    u = rng.uniform(0.0, 1.0, size=n)
    # Ensure S(t) is non-increasing
    surv = np.minimum.accumulate(surv[::-1])[::-1]
    out = np.empty(n, dtype=float)
    for i, ui in enumerate(u):
        # Find first index where S(t) <= u
        idx = np.argmax(surv <= ui)
        if surv[idx] <= ui:
            out[i] = time[idx]
        else:
            out[i] = t_max
    return out


def risk_mix_for_year(anchors: Dict[int, float], years: Iterable[int]) -> Dict[int, float]:
    """
    Build year→low_risk_fraction using anchored interpolation.

    WHEN USED: Step 4 to split survival sampling between low and IH risk.
    """
    return anchored_linear_interpolator(anchors, years)


# ──────────────────────────────────────────────────────────────────────
# Step 5 — Monte-Carlo core
# ──────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    proj_tavi: pd.DataFrame,
    proj_savr: pd.DataFrame,
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Execute the Monte-Carlo over the forecast horizon.

    Inputs:
        proj_tavi: year, age_band, sex, projected (index counts for TAVI)
        proj_savr: year, age_band, sex, projected (index counts for SAVR)
        cfg: configuration with durability, survival, penetration, simulation specs

    Returns:
        viv_pre:  DataFrame with year, tavr_in_tavr, tavr_in_savr, total_viv (means across runs)
        viv_post: same, after redo-SAVR subtraction
        qc:       dict of QC DataFrames (e.g., failures, viable counts)

    WHEN USED: Step 5-8.
    """
    sim = cfg.simulation
    rng = choose_rng(sim.seed)

    # Load survival curves
    t_low, s_low = load_survival_curve(cfg.survival.low_csv)
    t_ih,  s_ih  = load_survival_curve(cfg.survival.ih_csv)

    years = list(range(sim.start_year, sim.end_year + 1))
    pen_tit = anchored_linear_interpolator(cfg.penetration.anchors_tavr_in_tavr, years)
    pen_tis = anchored_linear_interpolator(cfg.penetration.anchors_tavr_in_savr, years)
    mix_low_tavi = risk_mix_for_year(cfg.survival.risk_mix_anchors.get("tavi", {}), years)
    mix_low_savr = risk_mix_for_year(cfg.survival.risk_mix_anchors.get("savr", {}), years)

    # Pre-allocate collectors
    rows_pre = []   # pre-redo results per run x year
    rows_qc  = []   # per run x year QC summaries

    # Vectorized helper to sample survival given low/IH mix
    def sample_survival(n: int, low_frac: float):
        n_low = int(round(n * low_frac))
        n_ih  = n - n_low
        surv_low = sample_from_survival_curve(n_low, t_low, s_low, rng)
        surv_ih  = sample_from_survival_curve(n_ih,  t_ih,  s_ih,  rng)
        return np.concatenate([surv_low, surv_ih], axis=0)

    # Repeat runs
    for run in range(sim.n_runs):
        # DURABILITY draws are done per group in bulk for efficiency
        # Build per-year totals from projected counts (can be floats; we use Poisson to add noise)
        # Note: we convert projected means to integer patients by Poisson sampling
        df_tavi = proj_tavi.copy()
        df_savr = proj_savr.copy()
        df_tavi["n"] = rng.poisson(lam=np.maximum(df_tavi["projected"].values, 0.0))
        df_savr["n"] = rng.poisson(lam=np.maximum(df_savr["projected"].values, 0.0))

        # Sample durability per "patient" group
        # Expand to arrays per (year, age_band) — sex is not used for durability
        failures_by_year_tit = {}  # year → number of TAVR-in-TAVR candidates *before* penetration
        failures_by_year_tis = {}  # year → number of TAVR-in-SAVR candidates *before* penetration
        viable_by_year_tit   = {}  # passed survival filter
        viable_by_year_tis   = {}

        # TAVI index
        for (year, age_band), sub in df_tavi.groupby(["year","age_band"]):
            n = int(sub["n"].sum())
            if n <= 0:
                continue
            # Durability
            durs = sample_mixture_norm(n, cfg.durability.tavi_modes, rng, clip_min=sim.min_dur_years)
            fail_years = year + np.floor(durs).astype(int)
            # Survival (risk mix depends on index year)
            low_frac = mix_low_tavi.get(int(year), 0.5)
            surv_t = sample_survival(n, low_frac)
            # Viable = survival >= durability
            viable = surv_t >= durs
            # Count failures landing within horizon
            for fy in np.unique(fail_years):
                cnt_all = int(np.sum(fail_years == fy))
                cnt_viab = int(np.sum((fail_years == fy) & viable))
                if sim.start_year <= fy <= sim.end_year:
                    failures_by_year_tit[fy] = failures_by_year_tit.get(fy, 0) + cnt_all
                    viable_by_year_tit[fy]   = viable_by_year_tit.get(fy, 0) + cnt_viab

        # SAVR index
        thr = cfg.durability.savr_age_threshold
        for (year, age_band), sub in df_savr.groupby(["year","age_band"]):
            n = int(sub["n"].sum())
            if n <= 0:
                continue
            # Choose modes by age
            # Parse numeric lower bound from band
            a0 = 0
            try:
                a0 = int(age_band.split("-")[0].replace("≥","").replace(">=","").replace("+",""))
            except Exception:
                a0 = 70
            modes = cfg.durability.savr_under_modes if a0 < thr else cfg.durability.savr_overeq_modes
            durs = sample_mixture_norm(n, modes, rng, clip_min=sim.min_dur_years)
            fail_years = year + np.floor(durs).astype(int)
            # Survival (risk mix depends on index year)
            low_frac = mix_low_savr.get(int(year), 0.85)
            surv_t = sample_from_survival_curve(n, t_low, s_low, rng) if low_frac >= 0.999 else \
                     sample_from_survival_curve(n, t_ih,  s_ih,  rng) if low_frac <= 0.001 else \
                     np.concatenate([
                        sample_from_survival_curve(int(round(n*low_frac)), t_low, s_low, rng),
                        sample_from_survival_curve(n - int(round(n*low_frac)), t_ih, s_ih, rng)
                     ], axis=0)
            viable = surv_t >= durs
            for fy in np.unique(fail_years):
                cnt_all = int(np.sum(fail_years == fy))
                cnt_viab = int(np.sum((fail_years == fy) & viable))
                if sim.start_year <= fy <= sim.end_year:
                    failures_by_year_tis[fy] = failures_by_year_tis.get(fy, 0) + cnt_all
                    viable_by_year_tis[fy]   = viable_by_year_tis.get(fy, 0) + cnt_viab

        # Apply penetration by failure year
        years_arr = np.arange(sim.start_year, sim.end_year + 1, dtype=int)
        tit_after_pen = np.array([int(round(viable_by_year_tit.get(y,0) * pen_tit.get(y, 0.1))) for y in years_arr])
        tis_after_pen = np.array([int(round(viable_by_year_tis.get(y,0) * pen_tis.get(y, 0.6))) for y in years_arr])

        # Record pre-redo rows for this run
        for i, y in enumerate(years_arr):
            rows_pre.append({
                "run": run,
                "year": int(y),
                "tavr_in_tavr": int(tit_after_pen[i]),
                "tavr_in_savr": int(tis_after_pen[i]),
                "total_viv": int(tit_after_pen[i] + tis_after_pen[i]),
            })
            rows_qc.append({
                "run": run,
                "year": int(y),
                "fail_tavi_all": int(failures_by_year_tit.get(y,0)),
                "fail_tavi_viable": int(viable_by_year_tit.get(y,0)),
                "fail_savr_all": int(failures_by_year_tis.get(y,0)),
                "fail_savr_viable": int(viable_by_year_tis.get(y,0)),
            })

    pre = pd.DataFrame(rows_pre)
    qc  = pd.DataFrame(rows_qc)

    # Aggregate across runs (means, sds)
    summary_pre = (pre.groupby("year")[["tavr_in_tavr","tavr_in_savr","total_viv"]]
                      .agg(["mean","std"]).reset_index())
    summary_pre.columns = ["year",
                           "tavr_in_tavr_mean","tavr_in_tavr_sd",
                           "tavr_in_savr_mean","tavr_in_savr_sd",
                           "total_viv_mean","total_viv_sd"]
    # QC means
    qc_summary = (qc.groupby("year")[["fail_tavi_all","fail_tavi_viable","fail_savr_all","fail_savr_viable"]]
                    .agg(["mean","std"]).reset_index())
    qc_summary.columns = ["year",
                          "fail_tavi_all_mean","fail_tavi_all_sd",
                          "fail_tavi_viable_mean","fail_tavi_viable_sd",
                          "fail_savr_all_mean","fail_savr_all_sd",
                          "fail_savr_viable_mean","fail_savr_viable_sd"]

    return pre, summary_pre, {"qc_runs": qc, "qc_summary": qc_summary}


# ──────────────────────────────────────────────────────────────────────
# Step 6 — Redo-SAVR subtraction
# ──────────────────────────────────────────────────────────────────────

def build_redo_targets(cfg: Config, proj_redo: Optional[pd.DataFrame], logger: logging.Logger) -> pd.DataFrame:
    """
    Build a series of redo-SAVR *targets* by year using either:
        • the projected redo-SAVR counts (from Step 3) — mode="from_projection"
        • an external CSV with columns: year,count — mode="from_file"

    We also fill/extend years per cfg.redo.fill and cfg.redo.yrange.

    Returns: DataFrame with 'year','redo_savr_target'.
    """
    sim = cfg.simulation
    years = list(range(sim.start_year, sim.end_year + 1))

    if cfg.redo.mode == "from_file":
        if not cfg.redo.file:
            raise ValueError("redo.mode='from_file' but redo.file not provided")
        df = pd.read_csv(cfg.redo.file)
        base = df.set_index("year")["count"].astype(float)
    else:
        if proj_redo is None:
            raise ValueError("redo.mode='from_projection' but proj_redo is None")
        base = (proj_redo.groupby("year")["projected"].sum().astype(float))

    # Fill according to policy
    y0, y1 = cfg.redo.yrange
    idx = pd.Index(range(y0, y1 + 1), name="year")
    series = base.reindex(idx)
    if cfg.redo.fill == "linear":
        series = series.interpolate(method="linear", limit_direction="both")
    elif cfg.redo.fill == "forward":
        series = series.ffill().bfill()
    else:
        series = series.fillna(0.0)
    out = series.reindex(years).fillna(method="ffill").fillna(0.0).reset_index()
    out.columns = ["year","redo_savr_target"]
    logger.info(f"[Step 6] Redo-SAVR targets prepared ({cfg.redo.mode}).")
    return out


def apply_redo_subtraction(summary_pre: pd.DataFrame, redo_targets: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Apply annual redo-SAVR targets (deterministic subtraction) to the **TAVR-in-SAVR** stream.

    We subtract min(target, tavr_in_savr_mean) to avoid negative values.
    The **TAVR-in-TAVR** stream is unchanged.

    Returns: DataFrame with year and *_post columns.
    """
    merged = summary_pre.merge(redo_targets, on="year", how="left")
    merged["redo_savr_target"] = merged["redo_savr_target"].fillna(0.0)
    # Apply subtraction to means, carry SDs unchanged (documented limitation)
    merged["tavr_in_savr_post"] = np.maximum(0.0, merged["tavr_in_savr_mean"] - merged["redo_savr_target"])
    merged["tavr_in_tavr_post"] = merged["tavr_in_tavr_mean"]
    merged["total_viv_post"]    = merged["tavr_in_savr_post"] + merged["tavr_in_tavr_post"]
    logger.info("[Step 7] Applied redo-SAVR subtraction to TAVR-in-SAVR stream.")
    return merged[["year","tavr_in_tavr_post","tavr_in_savr_post","total_viv_post"]]


# ──────────────────────────────────────────────────────────────────────
# I/O helpers: write tables and (optionally) plots
# ──────────────────────────────────────────────────────────────────────

def write_tables(
    outdir: Path,
    rates_tavi: pd.DataFrame,
    rates_savr: pd.DataFrame,
    rates_redo: pd.DataFrame,
    proj_tavi: pd.DataFrame,
    proj_savr: pd.DataFrame,
    proj_redo: pd.DataFrame,
    pre_runs: pd.DataFrame,
    pre_summary: pd.DataFrame,
    redo_targets: pd.DataFrame,
    post_summary: pd.DataFrame,
    qc: Dict[str, pd.DataFrame],
    logger: logging.Logger,
) -> Dict[str, Path]:
    """
    Save CSV tables into organized subfolders; return a dict of key paths.
    WHEN USED: end of run for auditability.
    """
    paths = {}
    tb_dir = ensure_dir(outdir / "tables")
    qc_dir = ensure_dir(outdir / "qc")

    def w(df: pd.DataFrame, rel: str):
        p = tb_dir / rel
        ensure_dir(p.parent)
        df.to_csv(p, index=False)
        paths[rel] = p
        return p

    w(rates_tavi, "rates/rates_tavi.csv")
    w(rates_savr, "rates/rates_savr.csv")
    w(rates_redo, "rates/rates_redo_savr.csv")

    w(proj_tavi, "index/projected_tavi_by_age_sex_year.csv")
    w(proj_savr, "index/projected_savr_by_age_sex_year.csv")
    w(proj_redo, "index/projected_redo_savr_by_age_sex_year.csv")

    w(pre_runs, "viv/viv_pre_runs.csv")
    w(pre_summary, "viv/viv_pre_summary.csv")
    w(redo_targets, "viv/redo_savr_targets.csv")
    w(post_summary, "viv/viv_post_summary.csv")

    qc["qc_runs"].to_csv(qc_dir / "failures_by_year_runs.csv", index=False)
    qc["qc_summary"].to_csv(qc_dir / "failures_by_year_summary.csv", index=False)

    logger.info(f"[I/O] Tables written under {tb_dir}")
    return paths


def try_plots(outdir: Path, proj_tavi: pd.DataFrame, proj_savr: pd.DataFrame,
              pre_summary: pd.DataFrame, post_summary: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Create basic figures using `viz_v8.py` (if import succeeded).

    WHEN USED: end of run (non-blocking if viz is unavailable).
    """
    if viz is None:
        logger.warning("[plots] viz_v8.py not found or matplotlib unavailable; skipping figures.")
        return
    fig_dir = ensure_dir(outdir / "figures")
    try:
        viz.plot_index_projection(proj_tavi, proj_savr, savepath=fig_dir / "index_projection.png")
        viz.plot_viv_pre_post(pre_summary, post_summary, savepath=fig_dir / "viv_pre_post.png")
        logger.info(f"[plots] Figures written to {fig_dir}")
    except Exception as e:
        logger.warning(f"[plots] Failed to generate figures: {e}")


# ──────────────────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: Config) -> None:
    """
    Run the full pipeline end-to-end. Writes logs, tables, QC, optional figures,
    and a Markdown report describing every step with the file paths.

    WHEN USED: main entry point.
    """
    outdir = ensure_dir(Path(cfg.output_dir))
    logger = setup_logging(outdir, level="INFO")

    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info("Step 1: Building YearxAgeBandxSex → Population…")
    pop = read_population_table(cfg, logger)

    # Limit population to simulation years for downstream steps
    years = list(range(cfg.simulation.start_year, cfg.simulation.end_year + 1))
    pop_f = pop[pop["year"].between(years[0], years[-1])].copy()

    logger.info("Step 2: Loading registries and learning baseline rates (pre-COVID windows)…")
    reg_tavi = read_registry_long(cfg.registry_tavi_csv)
    reg_savr = read_registry_long(cfg.registry_savr_csv)
    reg_redo = read_registry_long(cfg.registry_redo_csv) if cfg.registry_redo_csv else None

    rates_tavi = learn_baseline_rates(reg_tavi, pop, cfg.learning_windows.tavi, logger, "TAVI")
    rates_savr = learn_baseline_rates(reg_savr, pop, cfg.learning_windows.savr, logger, "SAVR")
    if reg_redo is not None:
        rates_redo = learn_baseline_rates(reg_redo, pop, cfg.learning_windows.redo_savr, logger, "REDO_SAVR")
    else:
        # If no historical redo registry, set rates to zeros (forces redo from file or zero)
        idx = rates_savr[["age_band","sex"]].drop_duplicates()
        rates_redo = idx.assign(rate=0.0, procedure="REDO_SAVR")

    logger.info("Step 3: Projecting index counts via Population x Rate x Multiplier…")
    mult_tavi = cfg.rate_multipliers.get("tavi", {})
    mult_savr = cfg.rate_multipliers.get("savr", {})
    mult_redo = cfg.rate_multipliers.get("redo_savr", {})

    proj_tavi = project_index_counts(rates_tavi, pop_f, mult_tavi, years, logger, "TAVI")
    proj_savr = project_index_counts(rates_savr, pop_f, mult_savr, years, logger, "SAVR")
    proj_redo = project_index_counts(rates_redo, pop_f, mult_redo, years, logger, "REDO_SAVR")

    logger.info("Step 4-5: Running Monte-Carlo to obtain ViV candidates and pre-redo totals…")
    pre_runs, pre_summary, qc = run_monte_carlo(proj_tavi, proj_savr, cfg, logger)

    logger.info("Step 6: Building redo-SAVR annual targets…")
    redo_targets = build_redo_targets(cfg, proj_redo, logger)

    logger.info("Step 7: Applying redo-SAVR subtraction (deterministic) to TAVR-in-SAVR…")
    post_summary = apply_redo_subtraction(pre_summary, redo_targets, logger)

    logger.info("Step 8: Writing tables and (optional) figures…")
    paths = write_tables(outdir, rates_tavi, rates_savr, rates_redo,
                         proj_tavi, proj_savr, proj_redo,
                         pre_runs, pre_summary, redo_targets, post_summary, qc, logger)
    try_plots(outdir, proj_tavi, proj_savr, pre_summary, post_summary, logger)

    # Build Markdown report
    sections = []
    sections.append(("Inputs",
                     f"- Population input mode: **{cfg.population_input.mode}**\n"
                     f"- Registry TAVI: `{cfg.registry_tavi_csv}`\n"
                     f"- Registry SAVR: `{cfg.registry_savr_csv}`\n"
                     f"- Registry redo-SAVR: `{cfg.registry_redo_csv or 'N/A'}`\n"
                     f"- Output dir: `{cfg.output_dir}`"))
    sections.append(("Population table",
                     f"- Built YearxAgeBandxSex table for **{years[0]}-{years[-1]}**\n"
                     f"- Bands: {', '.join(TARGET_BANDS)}\n"
                     f"- Interpolated annually from sparse points\n"))
    sections.append(("Rates & Multipliers",
                     f"- Learned baseline rates on windows:\n"
                     f"  • TAVI: {cfg.learning_windows.tavi}\n"
                     f"  • SAVR: {cfg.learning_windows.savr}\n"
                     f"  • redo-SAVR: {cfg.learning_windows.redo_savr}\n"
                     f"- Multipliers (anchors) applied per year for TAVI/SAVR/redo-SAVR\n"
                     f"- Tables: `tables/rates/…`"))
    sections.append(("Index projections",
                     "- Tables:\n"
                     "  • `tables/index/projected_tavi_by_age_sex_year.csv`\n"
                     "  • `tables/index/projected_savr_by_age_sex_year.csv`\n"
                     "  • `tables/index/projected_redo_savr_by_age_sex_year.csv`"))
    sections.append(("Monte-Carlo (pre-redo)",
                     f"- Runs: **{cfg.simulation.n_runs}** (seed: {cfg.simulation.seed})\n"
                     f"- PRE summary: `tables/viv/viv_pre_summary.csv`\n"
                     f"- QC (failures/viable by year): `qc/…`"))
    sections.append(("Redo-SAVR subtraction",
                     f"- Target series source: **{cfg.redo.mode}**\n"
                     f"- Targets: `tables/viv/redo_savr_targets.csv`\n"
                     f"- POST summary: `tables/viv/viv_post_summary.csv`"))
    write_report_md(outdir, sections)

    logger.info("DONE.")
    logger.info(f"Key output tables under: {outdir / 'tables'}")
    logger.info(f"Run report: {outdir / 'logs' / 'report.md'}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demography-driven Monte-Carlo ViV-TAVI model (v8)")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--scenario", type=str, default=None, help="Scenario key in the YAML")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config or DEFAULT_CONFIG_PATH)
    scenario = args.scenario or DEFAULT_SCENARIO
    print(f"[model_v8] Using config={cfg_path} scenario={scenario}")
    cfg = load_config(cfg_path, scenario)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
