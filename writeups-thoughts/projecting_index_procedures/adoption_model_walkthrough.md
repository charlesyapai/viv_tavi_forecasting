# TAVI/SAVR Detailed Projection Report (Volume-Preserved & Center-Driven)

This document details the complete calculation methodology.

## 1. Data Sources

- **Registry**: 2012-2024 Observed Data (TAVI + SAVR).
  - _Note_: 2015 data is verified present and included.
- **Population**: UN Projections for Republic of Korea.

![Data Sources](step1_data_overview.png)

## 2. Normalization (Volume-Preserved Redistribution)

**The Issue**: The pandemic caused volatility (Dip & Spike) in 2020-2023.
**The Solution**: We "redistribute" the patients from this 4-year block.

- **Constraint 1 (Volume Preservation)**: The **Total Sum** of normalized cases (2020-2023) MUST EQUAL the **Total Sum** of actual observed cases (2020-2023). We are not inventing or removing patients.
- **Constraint 2 (Trend Shape)**: The normalized data points are distributed to best fit the **Pre-COVID Trend (2012-2019)**.
- **Result**: The Blue Line smoothes out the dip/spike into a coherent curve that has the exact same "Area Under the Curve" as the actual data.

![Normalization Effect](step2_normalization_effect.png)

## 3. Risk Trends (Combined TAVI+SAVR)

We projected the **Total Risk** (TAVI+SAVR per 100k) for each age group using the Redistributed Data.

- **Logarithmic Saturation**: Risks grow but slow down over time.

![Risk Trends](step3_projected_risk_trends.png)

## 4. Total Addressable Market (TAM)

### Definition & Derivation

The **Total Addressable Market (TAM)** represents the theoretical maximum number of patients who _could_ require aortic valve intervention in a given year, based on demographics and clinical risk factors.

$$
\text{TAM}_{\text{Year}} = \sum_{\text{Age Band}} \left( \text{Population}_{\text{Age, Year}} \times \text{Risk Rate}_{\text{Age, Year}} \right)
$$

### Why is it Growing Exponentially?

The **Green Dotted Line** shows the **Population of 80+**. This cohort is projected to triple. This demographic explosion drives the TAM up, even if clinical risk rates saturate.

![TAM](step4_projected_tam.png)

## 5. TAVI Adoption (Center-Driven)

**Methodology**: The TAVI projection is strictly coupled to **Center Growth**.

- **Centers (Green)**: Projected to saturate at the Cap (94 centers).
- **TAVI Volume (Blue)**: = `Projected Centers` × `Projected Volume Per Center`.
- This ensures that if infrastructure stops growing, adoption slows down.

![TAVI & Centers](step5_tavi_projection.png)

## 6. Final SAVR Calculation (Residual)

`SAVR = TAM - TAVI`

The final result integrates Demographics (TAM), Infrastructure (Centers/TAVI), and the Residual (SAVR).

![Final Projection](step6_final_projection.png)

---

**Execution**:
`python models/final_adoption_model.py`
