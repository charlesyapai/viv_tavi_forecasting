# Implementation Plan: TAVI/SAVR Adoption-Based Index Forecasting

## Goal

Develop a new forecasting script (`models/project_index_adoption.py`) that projects TAVI and SAVR index procedure volumes for Korea. This model will explicitly account for:

1.  **Adoption Dynamics**: TAVI volume growth driven by the increasing number of centers.
2.  **Center Size Heterogeneity**: Acknowledging that early adopters are larger centers ("diminishing returns" on new centers).
3.  **Demographic Demand**: Total intervention volume driven by the aging population (TAM).
4.  **Substitution**: TAVI displacing SAVR over time.

## User Review Required

> [!IMPORTANT]
> **Adoption Logic Assumption**: We assume a **Zipfian (Power Law) distribution** for center sizes to model the "large centers first" effect.
> Center $k$ capacity $\propto k^{-\alpha}$.
> We will fit $\alpha$ and the scaling factor $\beta$ to minimize error against historical TAVI volumes (2012-2023).

> [!NOTE]
> **Future Center Projection**: We need to project the _number of TAVI centers_ into the future (2025-2050).
> We will assume a logistic saturation curve or simple logarithmic growth for the number of centers, capping at a reasonable estimate (e.g., 80-100 centers in Korea?). **Please verify if there is a known cap.**

## Proposed Changes

### [NEW] `models/project_index_adoption.py`

A standalone script to generate `tavi_adoption_projection.csv` and `savr_adoption_projection.csv`.

**Key Steps:**

1.  **Load Data**:
    - Historical TAVI/SAVR (Registry CSVs).
    - Historical Center Counts ([data/raw/TAVI_adoption_hospital_numbers_korea](file:///Users/charles/Desktop/viv_tavi_forecasting/data/raw/TAVI_adoption_hospital_numbers_korea)).
    - Population Projections (UN Data, >75 demographic).

2.  **Model 1: Total Addressable Market (TAM)**
    - Calculate Historical Rate: $R(t) = \frac{TAVI(t) + SAVR(t)}{Pop_{>75}(t)}$
    - Forecast $R(t)$ (Linear trend or constant match to 2023).
    - Project $TAM(t) = R(t) \times Pop_{>75}(t)$.

3.  **Model 2: TAVI Adoption Curve**
    - **Metric**: Weighted Hospital Capacity $C(N) = \sum_{k=1}^N k^{-\alpha}$.
    - **Fit**: Model TAVI Market Share $S_{tavi}(t) = \frac{TAVI(t)}{TAM(t)}$ as a Sigmoid function of $C(N(t))$.
      - $S_{tavi} = \frac{MaxShare}{1 + e^{-m(C(N(t)) - C_0)}}$
    - **Optimize**: Find $\alpha$ (distribution shape) and Sigmoid params ($m, C_0$) to best fit historical shares.

4.  **Forecast**:
    - Project $N(t)$ (Hospital Count) into future.
    - Calculate $S_{tavi}(future)$.
    - $TAVI_{proj} = TAM_{proj} \times S_{tavi}$.
    - $SAVR_{proj} = TAM_{proj} \times (1 - S_{tavi})$.

5.  **Output**:
    - Save CSVs in `simulation_run_v1/data/adoption_projections/`.
    - Generate plots showing the fit and the forecast (Adoption Curve, Volume Split).

## Verification Plan

### Automated Verification

- The script will calculate **R-squared** or **MAE** (Mean Absolute Error) for the historical fit (2012-2023).
- Visual check: Does the TAVI curve look like an S-curve? Does SAVR decline or plateau?

### Manual Steps

1.  Run `python models/project_index_adoption.py --country korea`.
2.  Inspect the generated plots in `simulation_run_v1/outputs/adoption_plots/`.
3.  Check if TAVI overtaking SAVR happens at a plausible year (based on user intuition).
