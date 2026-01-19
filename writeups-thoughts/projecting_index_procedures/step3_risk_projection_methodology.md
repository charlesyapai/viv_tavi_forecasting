# Methodology: Age-Stratified Logarithmic Risk Projection

## 1. Abstract

This document details the statistical methodology used to project the future incidence of aortic valve intervention ("Risk") for each age cohort. Building on the normalized historical data (Step 2), we employ a **Logarithmic Growth Model**. This choice is grounded in epidemiological theory, positing that adoption of a new therapy (TAVI) initially accelerates but eventually decelerates as it approaches the natural saturation limit of the prevalent disease pool. This contrasts with linear models (which would project infinite disease prevalence) or exponential models (which overestimate growth).

## 2. Problem Statement

We aim to project the risk rate $R_{age}(t)$ (procedures per 100,000 population) for the years $t \in [2025, ..., 2050]$.
The input data is the _Normalized Historical Series_ $N_{age}(t)$ derived in Step 2, which corrects for pandemic-era volatility while preserving total case volume.

## 3. Mathematical Model

For each age band $b$ (e.g., 75-79, 80+), we fit the following Logarithmic Function:

$$ R*b(t) = \alpha_b + \beta_b \cdot \ln(t - t*{offset}) $$

Where:

- $t$: The calendar year.
- $t_{offset}$: A fixed temporal offset (set to 2010) to anchor the logarithmic curve near the start of the TAVI era.
- $\alpha_b$: The baseline intercept parameter for age band $b$.
- $\beta_b$: The growth rate parameter for age band $b$.

### Justification for Logarithmic Form

The derivative of the risk rate is:
$$ \frac{dR}{dt} = \frac{\beta}{t - t\_{offset}} $$
This implies that the _rate of growth_ inversely proportional to time. As time progresses, the year-over-year increase in procedure risk diminishes. This models the "Low Hanging Fruit" effect:

1.  **Early Phase**: High growth as the therapy (TAVI) expands to untreated severe aortic stenosis patients.
2.  **Late Phase**: Growth slows as the treated population approaches the true prevalence limit of the disease in that age group.

## 4. Fitting Procedure

We utilize Non-Linear Least Squares (Levenberg-Marquardt algorithm) to estimate $\alpha_b$ and $\beta_b$ for each group.

- **Input Domain**: $t \in [2012, ..., 2024]$.
- **Target Variable**: The Normalized Risk Rates derived in Step 2.
- **Constraints**: $\beta > 0$ (Risk cannot decrease in the adoption phase).

## 5. Results & Visualization

The plot below illustrates the fitted curves.

- **Dots**: Normalized historical data points.
- **Lines**: The projected Logarithmic trends.
- **Observation**: The older cohorts (75-79, 80+) show steep initial growth ($\beta_{high}$) consistent with TAVI uptake, while younger cohorts show flatter trajectories.

![Projected Risk Trends](step3_projected_risk_trends.png)

## 6. Integration into TAM

These projected rates $R_b(t)$ are multiplied by the UN Population Projections $P_b(t)$ to derive the Total Addressable Market (TAM) in Step 4.
$$ \text{TAM}(t) = \sum\_{b} R*b(t) \cdot P_b(t) $$
This decouples the \_biological risk* (saturating) from the _demographic force_ (exponentially growing), allowing for a precise dissection of market drivers.
