# model_v9 quick start

**Run (same pattern as v7):**
```bash
cd simulation_run_v1
python models/model_v9.py --config configs/model_v9_configs.yaml --log-level DEBUG
```

**What `model_v9.py` does:**
1. **Precompute (new):**
   - Reads `data/korea_population_combined.csv` (units=10,000 people).
   - Builds annual **age×sex** population for **2025–2035** by *linear interpolation* of both counts and sex ratios.
   - Computes **per‑capita risk** for **TAVI**, **SAVR**, **redo‑SAVR** by age×sex for **2023 & 2024**.
   - Projects **redo‑SAVR absolute targets** for 2025–2035 (risk×population).

   Files emitted under `derived/`:
   - `age_projections_by_year_age_sex.csv`
   - `age_projections_by_year_age_allsex.csv`  ← used by the engine
   - `age_projections_by_year_band_sex.csv`
   - `baseline_risk_scores_by_sex.csv`
   - `redo_savr_targets.csv`

2. **Monte‑Carlo (unchanged vs v7):**
   - Uses **population‑rate** index projection (rates from 2023–2024).
   - Standard durability/survival, risk‑mix, penetration.
   - Subtracts redo‑SAVR targets from TAVR‑in‑SAVR (mode=`replace_rates`).

**Outputs (under `runs/<experiment>/<timestamp>/`)** replicate v7:
- `tables/index/*.csv`, `tables/viv/*.csv`, `tables/flow/*.csv`
- `figures/index/*.png`, `figures/flow/*.png`, `figures/viv/*.png`
- `qc/*` for index and redo reconciliation.

Adjust paths to your registry CSVs in `configs/model_v9_configs.yaml` under `procedure_counts:`.
