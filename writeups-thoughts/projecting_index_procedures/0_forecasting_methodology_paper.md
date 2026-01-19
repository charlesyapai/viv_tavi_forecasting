
# Methodology: Comprehensive TAVI & SAVR Forecasting Framework

## 1. Abstract
This document details the complete statistical framework used to forecast Transcatheter Aortic Valve Implantation (TAVI) and Surgical Aortic Valve Replacement (SAVR) volumes in South Korea through 2050. The model integrates demographic projections, procedure-specific risk trends, and infrastructure constraints to produce a robust demand-supply forecast. The methodology is divided into six distinct analytical steps.

---

## 2. Step 1 & 2: Data Normalization (Volume-Preserved Trend Redistribution)

### Problem Statement
The raw procedure data $O(t)$ exhibits varying volatility during the COVID-19 pandemic window ($W = [2020, 2021, 2022, 2023]$), characterized by an initial suppression of cases followed by a recovery spike. Direct use of this data introduces noise into long-term trend analysis.

### Derivation
We apply a **Volume-Preserved Trend Redistribution** to normalize the data $N(t)$ for the window $W$. This ensures two key constraints:
1.  **Shape Consistency**: $N(t)$ follows the pre-pandemic secular trend derived from 2012–2019.
2.  **Conservation of Volume**: The total number of procedures is preserved ($\sum_{t \in W} N(t) = \sum_{t \in W} O(t)$).

We derive a scaling factor $k = V_{obs} / V_{trend}$, where $V_{obs}$ is the total observed volume in $W$ and $V_{trend}$ is the projected trend volume. The normalized series is given by $N(t) = k \cdot T(t)$, ensuring the forecasting model uses a smooth curve that faithfully represents 100% of the clinical activity.

---

## 3. Step 3: Age-Stratified Risk Trend Modeling

### Rationale
The "Risk Rate" (procedures per 100,000 population) serves as a proxy for disease incidence + treatment penetration. While treatment rates have historically increased, they cannot grow indefinitely; they are biologically limited by the prevalence of Aortic Stenosis in the population.

### Formulation
We model the risk rate $R_{i}(t)$ for each age band $i$ (e.g., 75-79, 80+) using a logarithmic saturation function rather than a linear or exponential one.
$$ R_{i}(t) = \alpha_{i} + \beta_{i} \ln(t - t_{0}) $$

Where:
*   $\beta_{i} > 0$ models the diminishing returns of growth as the treated population approaches the true disease prevalence.
*   The model is fitted to the Normalized Historical Data (from Step 2) to prevent pandemic noise from distorting the long-term saturation level.

---

## 4. Step 4: Total Addressable Market (TAM) Derivation

### Demographic Integration
The TAM represents the theoretical maximum demand for aortic intervention. It is calculated by projecting the current risk rates onto the future demographic structure provided by UN Population Projections.

$$ \text{TAM}(t) = \sum_{i \in \text{Age Bands}} \left( P_{i}(t) \times \hat{R}_{i}(t) \right) $$

Where:
*   $P_{i}(t)$ is the projected population of age band $i$ in year $t$.
*   $\hat{R}_{i}(t)$ is the projected risk rate from Step 3.

### Mechanism of Growth
While $\hat{R}_{i}(t)$ saturates (flattens) over time, $\text{TAM}(t)$ exhibits continued exponential growth. This is driven by the **Demographic Multiplier**: the population of the highest-risk cohort ($80+$ years) is projected to triple between 2024 and 2050 (Demographic Force).

---

## 5. Step 5: Infrastructure-Constrained TAVI Adoption

### Concept
Traditional diffusion models (e.g., Bass Diffusion) assume unconstrained adoption. However, TAVI requires specialized TAVI Centers. We explicitly model adoption as a function of infrastructure capacity.

### Centers Projection ($C(t)$)
The number of TAVI centers is modeled with a Sigmoid growth curve, capped at a structural limit $C_{max}$ (determined by health policy or hospital constraints, set to ~94).
$$ C(t) = \frac{C_{max}}{1 + e^{-k(t - t_{mid})}} $$

### Volume per Center ($V_{pc}(t)$)
The efficiency or throughput of each center is modeled to trend towards a maturity level based on historical data.

### TAVI Forecast ($F_{tavi}(t)$)
The final TAVI forecast is the product of capacity and throughput:
$$ F_{tavi}(t) = C(t) \times V_{pc}(t) $$
This ensures that TAVI growth is physically constrained. If the number of centers stops growing (saturates), the total TAVI volume will also decelerate, preventing unrealistic infinite growth.

---

## 6. Step 6: SAVR Residual Calculation

### Substitution Interaction
We assume SAVR acts as the "Alternative Therapy" for the calculated TAM. As TAVI captures a larger share of the market, SAVR volume is determined as the residual demand.

$$ F_{savr}(t) = \max \left( 0, \text{TAM}(t) - F_{tavi}(t) \right) $$

This formulation inherently captures the "substitution effect". If TAVI grows faster than the TAM (due to aggressive center expansion), SAVR volume will decline. If TAVI growth slows (due to center saturation) while TAM continues to rise (due to demographics), SAVR may stabilize or even rebound to meet the excess demand.
