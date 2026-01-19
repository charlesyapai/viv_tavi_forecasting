# Methodology: Residual SAVR Projection (Step 6)

## 1. Abstract

This document details the final step of the forecasting pipeline: deriving the volume of Surgical Aortic Valve Replacement (SAVR). Rather than forecasting SAVR as an independent variable (which would risk decoupling it from the TAVI market), we model it as the **Residual Demand**. This ensures that the total market volume ($TAVI + SAVR$) is always consistent with the biologically derived TAM, respecting the "Must Treat" principle for severe aortic stenosis.

## 2. Problem Statement

We aim to project the annual volume of SAVR procedures $S(t)$ for $t \in [2025, ..., 2050]$.
Historically, SAVR volume has declined as TAVI adoption has grown. However, modeling this decline purely as a "decay rate" is insufficient because it ignores the expanding pool of patients (TAM).

## 3. Mathematical Formulation

The SAVR volume is calculated as the difference between the Total Addressable Market and the projected TAVI volume:

$$ S(t) = \max \left( 0, \ TAM(t) - T(t) \right) $$

Where:

- $TAM(t)$: The Total Addressable Market derived in Step 4 (Demographically driven).
- $T(t)$: The TAVI Volume derived in Step 5 (Infrastructure constrained).

## 4. Interpretation: The Equilibrium Gap

This formulation has two critical implications for the forecast:

1.  **SAVR does not necessarily vanish**: Even if TAVI grows significantly, if the _TAM grows faster_ (due to the 80+ population explosion), there will remain a "Gap" of untreated incidence that SAVR must fill.
2.  **Infrastructure Bottlenecks benefit SAVR**: If TAVI centers saturate (as modeled in Step 5), the TAVI curve flattens. The excess demand from the rising TAM spills over into the SAVR residual, potentially causing SAVR volumes to stabilize or even slightly recover in the long term, rather than crashing to zero.

## 5. Visualization

The final forecast plot illustrates this dynamic balance.

- **Black Dashed Line**: The TAM (Total Demand).
- **Blue Line**: TAVI (Supply-Constrained Growth).
- **Red Line**: SAVR (The Residual Gap).
- **Insight**: Notice how the Red Line eventually stabilizes, indicating a long-term role for surgery in meeting the demographic tidal wave.

![Final Market Projection](step6_final_projection.png)
