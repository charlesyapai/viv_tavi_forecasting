# Methodology: Demographic-Driven Total Addressable Market (TAM) Projection

## 1. Abstract

This document details the methodology for calculating the Total Addressable Market (TAM) for aortic valve intervention. Unlike "Top-Down" market sizing which often relies on crude broad-population multipliers, we utilize a granular "Bottom-Up" approach. We derive the TAM by interacting the age-stratified risk rates (Step 3) with high-fidelity longitudinal population projections from the United Nations. This allows us to strictly decouple the _clinical propensity_ for treatment from the _demographic momentum_ of an aging society.

## 2. Problem Statement

We aim to quantify $TAM(t)$, the total count of patients in South Korea theoretically eligible for aortic valve intervention in year $t$, for $t \in [2025, ..., 2050]$.

## 3. Mathematical Formulation

The TAM is defined as the scalar product of the Population vector and the Risk vector across all age cohorts $B$:

$$ TAM(t) = \sum\_{b \in B} \left( P_b(t) \cdot R_b(t) \right) $$

Where:

- $b$: Index for age cohorts (50-54, ..., 80+).
- $P_b(t)$: Propjected population count for age group $b$ in year $t$ (Source: UN World Population Prospects 2024).
- $R_b(t)$: Projected clinical risk rate (per capita) for age group $b$ in year $t$ (Source: Step 3 Logarithmic Model).

## 4. Why TAM Grows Exponentially

A critical finding of this model is that while the _Risk Rate_ $R_b(t)$ saturates (follows a logarithmic curve), the _TAM_ grows exponentially. This is driven entirely by the term $P_{80+}(t)$.

- **Clinical Saturation**: We assume the percentage of 80-year-olds needing TAVI stabilizes (Step 3).
- **Demographic Explosion**: The _number_ of 80-year-olds in South Korea is projected to triple over the next 20 years.

Therefore, the market growth is not driven by "doing more procedures per person", but by "having significantly more eligible people".

## 5. Visualization

The plot below visualizes this relationship.

- **Black Line**: The calculated TAM.
- **Green Dashed Line**: The raw population count of the 80+ cohort.
- **Correlation**: The almost perfect lock-step growth confirms that demography is the primary driver of the future market size.

![TAM vs Aging Population](step4_projected_tam.png)
