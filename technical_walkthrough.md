# ViV–TAVI Monte‑Carlo Forecast (`model_v7.py`) — Technical Walkthrough

---

## 1) What the model does (high level)

**Goal.** Forecast future annual volumes of **valve‑in‑valve (ViV) TAVR** originating from:

* **TAVR index procedures** that later fail (→ *TAVR‑in‑TAVR*), and
* **SAVR index procedures** with bioprosthetic valves that later fail (→ *TAVR‑in‑SAVR*).

**Core idea.** Combine **observed registry counts** (by year × age band) with:

1. **Index projection** (to extend TAVR and SAVR series into the future),
2. **Patient‑level Monte Carlo** draws for **durability** and **survival**, plus risk mix and age hazard,
3. **Behavioral filters**: redo‑SAVR probabilities and **ViV penetration** by year/type,
4. **Post‑run reconciliation** against **absolute redo‑SAVR targets** (if provided).

**Outputs.** Organized **tables**, **QC slices**, and **figures** under a timestamped `runs/<experiment>/<YYYY-mm-dd-HHMMSS>...` directory:

```
logs/, meta/, inputs_snapshot/, tables/{index,flow,viv, viv/per_run}, qc/{index_projection,redo}, figures/{index,flow,viv}
```

---

## 2) Inputs & data contracts

### Required files (from `config.yaml`)

* **Registry counts**

  * `procedure_counts.tavi`: `data/registry_tavi.csv`
  * `procedure_counts.savr`: `data/registry_savr.csv`
    Each should contain **at least**: `year, age_band, count` (optional `sex` is handled).
* **Population projection** (optional but recommended for age redistribution / pop‑rate method)

  * `population_projection.path`: `data/korea_population_projection.csv`
  * Columns specified by `year_col, age_col, pop_col` (e.g., `year, age, population`).
* **Redo‑SAVR counts** (optional)

  * Either `redo_savr_numbers.path` **or** `procedure_counts.redo_savr`
    CSV may be either `year,count` **or** `sex,age_band,year,count`. Script sums by year.

### Key configuration knobs (selected)

| Section                       | Purpose                                         | Examples / Notes                                              |                                             |           |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------- | --------- |
| `years`                       | forecast window                                 | `simulate_from: 2015`, `end: 2035`                            |                                             |           |
| \`index\_projection.tavi      | savr.method\`                                   | project observed series                                       | `population_rate` \| `linear` \| `constant` |           |
| `index_projection.*.pop_rate` | age‑specific pop‑rate projection                | `obs_years`, `min_age`, `annual_rate_growth` **or** `anchors` |                                             |           |
| `open_age_bin_width`          | width for open bands (e.g., `≥85`)              | default used to convert open bands to a finite range          |                                             |           |
| `age_bins`                    | **reporting** bins used to label candidates     | e.g., `[60,65,70,75,80,85,100]`                               |                                             |           |
| `durability`                  | Normal mixtures per valve type / age            | TAVR mixture; SAVR <70 vs ≥70                                 |                                             |           |
| `survival_curves`             | Normal by risk category                         | 2 or 3 categories; see `risk_model.categories`                |                                             |           |
| `age_hazard`                  | age hazard ratio effect on survival             | scales survival time around `ref_age`                         |                                             |           |
| `risk_mix`                    | risk share by year (TAVR), fixed (SAVR)         | piecewise ranges for TAVR                                     |                                             |           |
| `penetration`                 | ViV adoption by type over time                  | linear interpolation between anchors                          |                                             |           |
| `redo_rates`                  | per‑event redo probability (used conditionally) | `savr_after_savr`, `savr_after_tavi`                          |                                             |           |
| `redo_savr_numbers`           | absolute redo targets & fill policy             | `mode`, `fill_missing.method` (\`zero                         | forward                                     | linear\`) |
| `simulation`                  | runs, RNG seed, min durability                  | `n_runs: 20`, `rng_seed`, `min_dur_years`                     |                                             |           |

---

## 3) Deterministic preprocessing: **Index projection** (observed → future)

The script constructs **TAVR** and **SAVR** “index” time series per year × age band, tagging rows with a source in `src ∈ {observed, pop_rate, extrap}`.

### 3.1 Age‑band parsing and labels

* Registry **age bands** like `"75-79"`, `">85"`, `"≥85"`, `"<60"` are parsed with `parse_age_band`, with open bands turned into finite ranges using `open_age_bin_width` (e.g., `≥85` → `[85, 105)` if width=20).
* **Reporting labels** for Monte Carlo outputs come from `age_bins` (edges) via `_age_labels`, yielding bins like `"<60", "60–64", …, "≥100"`.

### 3.2 Projection methods

**A) Population‑rate (preferred for TAVR in your config)**
`_project_index_by_pop_rate(...)`

1. **Estimate base per‑age rates** from **observed** registry years (`obs_years`), dividing band counts by matching population and spreading band rates uniformly over single ages in that band; set rates to zero for ages `< min_age`.
2. **Scale** those base rates into the future using either:

   * **Anchors** (piecewise linear multipliers by year), or
   * **Annual growth** (compound).
3. For each **future year**: multiply per‑age rates by that year’s population to get expected counts, **sum to band level**, and **integer‑allocate** totals with **largest‑remainder** rounding to keep annual totals consistent. Rows are tagged `src="pop_rate"`.

**B) Linear / Constant projection (v5‑style)**
`_linear_or_constant(...)`

* Fit a **linear trend** to the last `window` observed **totals** (or use a fixed `constant` from a given year).
* Produce **future totals** in a placeholder band; if a population table exists, call `redistribute_future_by_population(...)` to **redistribute those totals across the observed bands** proportionally to population in each band. Rows are tagged `src="extrap"`.

> The model saves full observed+projection tables to
> `tables/index/tavi_with_projection.csv` and `.../savr_with_projection.csv` and produces index QC plots.

---

## 4) The Monte Carlo engine — **what happens per patient**

The engine draws **patient‑level** trajectories for each **index** record (year × band × count). It repeats this for **TAVR** and **SAVR**, and across **n\_runs**.

### 4.1 Random sources and distributions

* **RNG**: `np.random.default_rng(rng_seed)` for reproducibility.
* **Risk mix** (`_risk_mix`):

  * **SAVR:** fixed shares (`low, intermediate, high`).
  * **TAVR:** shares vary by **year range** (piecewise).
* **Survival time** by risk (`_sample_survival`):

  * Draw from Normal with risk‑specific `(median, sd)`.
  * Optional **age hazard**: divide time by `hr_per5 ** ((age - ref_age)/5)` (older → shorter survival).
  * Times are floored to **≥ 0.1 yr**.
* **Durability**:

  * **TAVR:** mixture of Normals (“early/late” modes, with weights).
  * **SAVR:** **age‑split** distribution: `<70` vs `≥70`.
  * Enforced **minimum durability**: `min_dur_years` (default **1.0**), preventing same‑year failures.

### 4.2 Event‑year discretization

For an index procedure in **calendar year `Y`**:

* Failure year `Y_fail = Y + floor(durability)`,
* Death year `Y_death = Y + floor(survival)`.
* A patient becomes a **ViV candidate** iff:

  * `Y_fail ≤ Y_death`, and
  * `simulate_from ≤ Y_fail ≤ end`.

### 4.3 Redo logic and ViV penetration

For each **candidate** in year `Y_fail`:

1. **Redo gate (optional)** — With probability:

   * `savr_after_tavi` for **TAVR‑in‑TAVR** candidates,
   * `savr_after_savr` for **TAVR‑in‑SAVR** candidates,
     the patient is recorded as **redo‑SAVR** and **does not realize ViV**.

   **Important:** If an **absolute redo‑SAVR target** file is provided **and** `redo_savr_numbers.mode == "replace_rates"`, the simulation **turns off per‑event redo rates** to avoid double counting. (Per‑event redo rates are used otherwise.)

2. **Penetration gate** — With probability given by **`penetration[viv_type]`** at year `Y_fail` (interpolated between anchors), the candidate **realizes** a ViV procedure of type:

   * **`tavi_in_tavi`** (if the index was TAVR),
   * **`tavi_in_savr`** (if the index was SAVR).

### 4.4 Aggregation within a run

* **Candidates** are recorded by `(year, viv_type, risk, age_bin)` (age binned by your `age_bins` edges).
* **Realized ViV** are recorded by `(year, viv_type)`.
* **Flow accounting** per `(year, proc)` tracks `index_cnt`, `deaths`, `failures`, `viable`.

### 4.5 Repeating runs and **correct averaging**

For `simulation.n_runs`:

* The script aggregates **within each run** to **year × type** totals, then
* Averages those **per‑run totals across runs** to get **means and SDs**.
  This fixes the common pitfall of averaging per‑bin means directly (which can bias totals).
  For QA, it also writes a **v5‑like** “mean‑of‑bins” table.

---

## 5) Pseudocode of one Monte Carlo run

```text
for proc_type in {"tavi","savr"}:
  for each row (year, age_band, count) in projected index table:
    ages  ← Uniform integers in parsed age band, size=count
    risks ← Sample by year-specific mix (proc_type)
    surv  ← Draw survival by risk; apply age-hazard (optional); clamp ≥0.1
    dur   ← Draw durability by valve type (and age split for SAVR); clamp ≥ min_dur_years

    Y_fail  ← year + floor(dur)
    Y_death ← year + floor(surv)

    record failures/deaths tallies by Y_fail/Y_death

    viable_mask ← (Y_fail ≤ Y_death) ∧ (simulate_from ≤ Y_fail ≤ end)
    for each viable patient:
      vtype ← "tavi_in_tavi" if proc_type=="tavi" else "tavi_in_savr"
      record candidate (Y_fail, vtype, risk, age_bin(ages[i]))

      # (1) Redo gate
      p_redo ← redo_rate[vtype]   # possibly 0 if absolute targets with mode=replace_rates
      if U(0,1) < p_redo:
        record redo log; continue

      # (2) Penetration gate
      if U(0,1) < penetration[vtype](Y_fail):
        record realized ViV (Y_fail, vtype)
```

---

## 6) Post‑processing & reconciliation

### 6.1 Multi‑run summaries

* **Candidates** (totals): `tables/viv/viv_candidates_totals.csv`
* **Realized (PRE)** (means across runs): `tables/viv/viv_forecast.csv`
* **Per‑run totals**: `tables/viv/per_run/{candidates,realized}_per_run.csv`
* **Patient flow (mean)**: `tables/flow/patient_flow_mean.csv`

### 6.2 **Absolute redo‑SAVR targets** (TAVR‑in‑SAVR only)

If a year → count mapping is available (from `redo_savr_numbers` or `procedure_counts.redo_savr`), the script **subtracts** these from the **TAVR‑in‑SAVR realized means** to produce **POST** volumes:

* Input targets may be **forward‑filled** or **linearly interpolated** per `fill_missing`.
* Output CSV includes both **before** and **after** (`viv_forecast_realized.csv`, with `mean` and `realized`), and a QC file (`qc/redo/redo_savr_qc.csv`).

> **Note:** Even when `mode != "replace_rates"`, the script still applies the **post‑hoc subtraction** if targets exist. With `mode="replace_rates"`, it **also** disables per‑event redo during the MC to avoid double counting.

### 6.3 Figures & QC slices

* **Index series** (“Image A/B”) with observed vs projection: `figures/index/*.png`
* **Flow plots** for TAVR/SAVR: `figures/flow/*.png`
* **ViV rollups (“Image C”)**

  * `plotting.image_c_source: post|pre|both` controls whether to show post‑reconciliation bars/lines, pre‑ lines, or both overlay. Output under `figures/viv/`.
* **v5‑style line charts** (candidates, pre, post) and corresponding **total CSVs** are written under `figures/viv/lines_*` and copied into `tables/viv/*v5style*.csv`.
* **Index projection QC** slices saved to `qc/index_projection/`.

---

## 7) Key modeling choices & assumptions (worth knowing)

* **Uniform ages within a registry band.** When sampling patient ages from an age band, draws are **uniform** across single‑year ages in that band.
* **Durability and survival are Normal‑based**, clipped to be positive; durability is **min‑clamped** to avoid same‑year failures; both are **floored to calendar years** before adding the index year.
* **Candidate viability** requires failure **before or at** death and within the forecast horizon.
* **Penetration** interpolates between anchors; if only one anchor is provided, it acts as a flat rate.
* **Population‑rate projection** assumes band rates are **constant across single ages within each band** for the selected observed years, then applied to the population forecast.
* **Correct totals.** The script **sums within runs** before averaging across runs; the extra `*v5like*` table is present explicitly for QA only.
* **Redo reconciliation.** **Absolute targets** (if present) are **subtracted** from **TAVR‑in‑SAVR** realized means to form the **POST** series, irrespective of `mode`. With `mode="replace_rates"`, redo probabilities inside the MC are disabled to avoid double subtraction.

---

## 8) How to run

From the repo root (paths in your config must exist):

```bash
python models/model_v7.py --config configs/model_v7_configs.yaml --log-level DEBUG
# (example from the header)
# /Users/charles/miniconda3/bin/python .../models/model_v7.py --config configs/model_v7_configs.yaml --log-level DEBUG
```

* A timestamped run folder will be created under `runs/<experiment_name>/`.
* Raw and **effective** configs are snapshotted under `meta/`.
* Input CSVs (TAVR, SAVR, redo) are copied under `inputs_snapshot/` for provenance.

---

## 9) Quick map from **config** → **behavior**

* **`index_projection.tavi.method: population_rate`** (as in your file)
  → Derive **per‑age rates** from **2018–2022** observed TAVR, zero below age 60, apply **0% annual rate growth**, scale by **Korean population forecast** each year, and **integer‑allocate** to the registry’s age bands.

* **`index_projection.savr.method: constant`**
  → Total SAVR volumes fixed at **2,000/yr from 2024**, redistributed across bands by population shares.

* **`risk_model.categories: 3`** + `risk_mix` blocks
  → Draw **low/intermediate/high** survival distributions with the **TAVR risk mix changing over time**.

* **`age_hazard`**
  → Survival times are scaled around **ref\_age 75** with per‑5‑year hazard multipliers (`low: 1.30`, `intermediate: 1.04`, `high: 1.01`).

* **`durability`**
  → TAVR: 20% “early” (mean 4y) + 80% “late” (mean 11.5y).
  → SAVR: mean 10y (<70) vs 17y (≥70). All subject to **min\_dur\_years = 1.0**.

* **`penetration`**
  → ViV uptake interpolates from the left anchor (2007–2022) to **2035**:
  `tavi_in_tavi`: 10% → 60% ; `tavi_in_savr`: 60% → 80%.

* **`redo_savr_numbers`**
  → Read absolute redo counts by year from `data/registry_redo_savr.csv`.
  `mode: replace_rates` disables redo probabilities **inside** the MC.
  Missing years are **forward‑filled** across 2015–2035.

* **`simulation.n_runs: 20`, `rng_seed: 20250714`**
  → 20 independent runs averaged for means/SDs; results are reproducible.

---

## 10) What to look at after a run (minimum viable QC)

1. **Index series** (figures/index): do observed vs projection transitions look plausible given registry and population?
2. **`tables/viv/viv_forecast.csv`** (PRE) vs **`viv_forecast_realized.csv`** (POST): confirm the size and pattern of **redo subtraction** for TAVR‑in‑SAVR (see `qc/redo/redo_savr_qc.csv`).
3. **`tables/flow/patient_flow_mean.csv`**: do failures and viable candidates scale sensibly vs index volumes?
4. **`figures/viv/image_C_*.png`**: at‑a‑glance ViV totals and the split between **TAVR‑in‑SAVR** and **TAVR‑in‑TAVR**.

---

## 11) Common adaptations

* **Change projection method**: switch `index_projection.tavi.method` between `population_rate`, `linear`, `constant`.
* **Tune ViV adoption**: adjust `penetration` anchors; the model interpolates between them.
* **Scenario toggles**:

  * Lift **min durability** (e.g., set `min_dur_years: 0.1`) to test sensitivity to near‑term failures.
  * Use **anchors** for population‑rate growth (e.g., 2022:1.0 → 2035:1.25).
  * Flip redo handling by supplying (or removing) `redo_savr_numbers` and switching `mode`.
