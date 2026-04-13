# Forecasting TAVI and SAVR Procedure Volumes: A Demographic-Anchored Adoption Model

---

## 1. Introduction

### 1.1 Aortic Stenosis: The Clinical Problem

Aortic stenosis (AS) is the most common acquired valvular heart disease in the developed world, affecting approximately 2–7% of adults aged over 65 years and rising sharply to 12.4% in those aged over 75 years. The disease results from progressive calcification and fibrosis of the aortic valve leaflets, leading to obstruction of left ventricular outflow. Without intervention, symptomatic severe AS carries a dismal prognosis: 2-year mortality rates exceed 50% once symptoms of heart failure, syncope, or angina develop. For decades, SAVR — performed via median sternotomy under cardiopulmonary bypass — was the sole definitive treatment for severe AS, offering substantial symptomatic and survival benefits in operable patients.

### 1.2 The TAVI Revolution

The introduction of TAVI in 2002 by Cribier and colleagues fundamentally altered the therapeutic landscape. TAVI enables bioprosthetic valve deployment via a percutaneous or transapical approach without the need for open-heart surgery or cardiopulmonary bypass. The landmark PARTNER and CoreValve trials established TAVI as non-inferior — and in some populations, superior — to SAVR across a progressive broadening of risk categories:

- **High-risk patients** (PARTNER 1, 2010): TAVI demonstrated non-inferiority to SAVR with respect to 1-year mortality in high-risk surgical candidates, and superiority to medical therapy in inoperable patients.
- **Intermediate-risk patients** (PARTNER 2, 2016; SURTAVI, 2017): TAVI was shown to be non-inferior to SAVR in intermediate-risk populations, with lower rates of acute kidney injury, atrial fibrillation, and shorter hospitalisation.
- **Low-risk patients** (PARTNER 3, 2019; Evolut Low Risk, 2019): TAVI demonstrated non-inferiority to SAVR in low-risk patients, effectively expanding the eligible population to the vast majority of AS patients.

These successive trial results, combined with improvements in valve technology, delivery systems, and operator experience, have driven a rapid expansion in TAVI utilisation. In many developed Western countries, TAVI now accounts for more than 50% of all aortic valve interventions, with some centres performing TAVI in over 75% of cases.

### 1.3 The Asian Context: A Confluence of Demographic Forces

While TAVI adoption has been extensively studied in North America and Europe, the Asian context presents unique features that warrant dedicated analysis. Several converging factors make East and Southeast Asia a particularly important arena for TAVI forecasting:

**Rapid population ageing.** Asia is experiencing the most rapid demographic transition in human history. South Korea's ≥65 population is projected to grow from 17% (2022) to 46% (2050) of the total — the fastest ageing rate among OECD nations. Singapore follows a similar trajectory, with the ≥65 cohort projected to grow from 18% to 34% over the same period. The ≥80 population — the cohort with the highest AS prevalence and the primary driver of aortic valve procedure demand — is projected to increase 3.0-fold in Korea (2.4 million to 7.3 million) and 3.6-fold in Singapore (159,000 to 567,000) between 2024 and 2050.

**Delayed but accelerating TAVI adoption.** TAVI adoption in Asia has lagged behind Western countries by approximately 5–10 years, reflecting differences in regulatory timing, reimbursement frameworks, and institutional accreditation requirements. South Korea's TAVI programme began in earnest in 2015 following national health insurance coverage, and experienced a 31-fold increase in volume by 2024. Singapore has had a longer-established but smaller programme since 2012, with more conservative growth. This asynchrony between Asia and the West means that Asia's TAVI adoption is currently in its steepest growth phase — precisely the period where accurate forecasting is most valuable for planning.

**Infrastructure and workforce constraints.** Unlike Western healthcare systems where catheterisation laboratory capacity is abundant, Asian TAVI programmes often face constraints in specialised infrastructure, trained interventional teams, and centre accreditation. These constraints shape the trajectory of adoption in ways that purely demand-driven models fail to capture.

### 1.4 The Valve-in-Valve Problem

All bioprosthetic valves — whether deployed surgically (SAVR) or transcatheter (TAVI) — are subject to structural valve deterioration (SVD) over time. SVD results from progressive leaflet calcification, thrombosis, and mechanical wear, eventually leading to clinically significant stenosis or regurgitation that necessitates reintervention. The durability of current-generation bioprosthetic valves is estimated at 10–20 years, with younger patients experiencing faster deterioration due to greater haemodynamic stress and immunological activity.

Valve-in-valve (ViV) TAVI — the deployment of a transcatheter valve within a previously implanted bioprosthetic valve — has emerged as the preferred reintervention strategy for patients with failed bioprostheses. ViV TAVI avoids the morbidity of redo open-heart surgery, which carries operative mortality rates of 5–10% in elderly patients. As TAVI adoption has accelerated, the future burden of ViV procedures has become a pressing question for healthcare systems, as the cohort of patients with implanted bioprostheses grows cumulatively over time.

Accurate forecasting of ViV demand requires two upstream components: (1) reliable projections of index TAVI and SAVR volumes over the forecast horizon, and (2) patient-level modelling of valve durability and survival to estimate when implanted valves will fail. The present work addresses the first component — the forecasting of index procedure volumes.

### 1.5 Limitations of Existing Forecasting Approaches

Existing approaches to procedural volume forecasting for structural heart interventions suffer from several methodological shortcomings:

1. **Naive trend extrapolation.** Linear or exponential models fitted to recent historical volumes fail to account for the nonlinear dynamics of technology adoption (S-shaped diffusion) and the saturation effects that arise as TAVI captures an increasing share of the addressable market.

2. **Pandemic sensitivity.** The COVID-19 pandemic caused acute suppression of elective cardiac procedures in 2020–2021, followed by variable recovery patterns across countries. Models that include raw pandemic-era data in trend fitting will systematically underestimate future volumes, while models that exclude this period entirely discard clinically relevant volume information.

3. **Failure to disentangle demographic drivers.** Aggregate volume trends conflate two distinct forces: (a) the technology adoption effect (TAVI replacing SAVR), and (b) the demographic effect (growing elderly populations expanding total demand). Without separating these, it is impossible to determine whether observed volume growth will continue, accelerate, or decelerate.

4. **Absence of age-stratification.** Procedure rates vary dramatically across age bands — differing by 10–30 fold between the 50–54 and ≥80 cohorts. Models that project aggregate volumes without accounting for the shifting age composition of the population will produce biased forecasts, particularly in rapidly ageing societies where the contribution of the oldest cohorts is increasing nonlinearly.

### 1.6 Study Objectives

We present a six-step forecasting framework that explicitly addresses these limitations by:

1. **Anchoring projections in demographic structure** — using United Nations population projections to account for the age-specific drivers of demand.
2. **Normalising for pandemic disruption** — applying a volume-preserved redistribution that corrects the shape of pandemic-era data while conserving observed clinical volume.
3. **Modelling age-specific risk rates** with a saturation function that prevents biologically implausible growth.
4. **Separating technology adoption from demographic growth** — decomposing the forecast into a demographically driven TAM and a sigmoid adoption share.
5. **Providing a country-agnostic framework** that can be parameterised for different healthcare systems.

The framework is applied to South Korea and Singapore as two contrasting case studies representing different stages of TAVI adoption, healthcare system structures, and population scales. The resulting TAVI and SAVR volume projections serve as upstream inputs to a patient-level Monte Carlo simulation that forecasts future ViV TAVI demand.

---

## 2. Methodology

### 2.1 Data Sources

Three categories of input data are required:

1. **National Registry Procedure Counts.** Annual TAVI and SAVR procedure counts stratified by five-year age band (50–54, 55–59, 60–64, 65–69, 70–74, 75–79, ≥80) and sex. For South Korea, raw registry data from the national HIRA database (2012–2024) are used. For Singapore, procedure counts from the national cardiac registry (2016–2024) are used.

2. **United Nations World Population Prospects.** Sex-specific population projections by five-year age group for the country of interest, obtained from the UN Department of Economic and Social Affairs (2024 revision). Male and female projections are summed to obtain total population by age band and year from 2012 to 2050.

3. **Configurable Model Parameters.** The maximum TAVI adoption share ($L_{\max}$, default: 0.50), the sigmoid fitting window start year (default: 2016), and the COVID-19 redistribution window (2020–2023) are specified as model inputs.

### 2.2 Age-Band Harmonisation

UN population projections are reported in standard five-year bands (e.g., 80–84, 85–89, …, 100+). These are mapped to the registry age-band schema by summing all UN bands with a lower bound ≥80 — including the open-ended 100+ category — into a single ≥80 band. For bands 50–54 through 75–79, a direct one-to-one mapping is applied.

### 2.3 Step 1: Data Overview

Historical TAVI and SAVR procedure counts are aggregated by year to produce an annual combined total:

$$
C(t) = N_{\mathrm{TAVI}}(t) + N_{\mathrm{SAVR}}(t), \quad t \in [t_{\mathrm{start}}, 2024]
$$

This combined series, together with the population projections for the ≥80 and 50–79 cohorts, provides the foundational inputs for trend analysis.

### 2.4 Step 2: Pandemic-Period Volume-Preserved Redistribution

The COVID-19 pandemic introduced systematic distortion into the 2020–2023 procedural data, characterised by acute volume suppression in 2020 followed by a compensatory rebound in subsequent years. Direct use of these observations would bias trend-based projections.

We apply a **volume-preserved redistribution** that satisfies two constraints simultaneously:

1. **Shape consistency** — the normalised series follows the pre-pandemic secular trend.
2. **Volume conservation** — total observed procedures within the window are exactly preserved.

**Procedure.** A first-order polynomial (linear trend) is fitted to the pre-COVID data ($t \leq 2019$):

$$
T(t) = \alpha + \beta \cdot t
$$

The redistribution scaling factor $\kappa$ is computed as:

$$
\kappa = \frac{\sum_{t \in W} O(t)}{\sum_{t \in W} T(t)}
$$

where $W = \{2020, 2021, 2022, 2023\}$ is the pandemic window and $O(t)$ are the observed counts. The normalised values for the pandemic window are then:

$$
N(t) = \kappa \cdot T(t), \quad t \in W
$$

For years outside $W$, no adjustment is made: $N(t) = O(t)$.

This redistribution is applied independently to three series: the combined TAVI + SAVR series (for TAM computation), each age-specific risk rate series (Step 3), and the TAVI-only series (Step 5).

### 2.5 Step 3: Age-Stratified Risk Rate Projection

The **procedure risk rate** $R_i(t)$ for age band $i$ in year $t$ is defined as the number of combined TAVI + SAVR procedures per unit population:

$$
R_i(t) = \frac{N_{\mathrm{TAVI},i}(t) + N_{\mathrm{SAVR},i}(t)}{P_i(t)}
$$

where $P_i(t)$ is the population of age band $i$ in year $t$.

Raw risk rates are first normalised using the redistribution procedure (Section 2.4). The normalised rates are then projected forward using a **logarithmic saturation model**:

$$
\hat{R}_i(t) = a_i + b_i \cdot \ln(t - t_{\mathrm{offset}})
$$

where $t_{\mathrm{offset}} = 2010$ and parameters $(a_i, b_i)$ are estimated by nonlinear least-squares fitting (Levenberg–Marquardt algorithm) to the historical normalised rates for each age band $i$.

**Rationale.** The logarithmic functional form captures the empirical observation that treatment rates grow rapidly during the early adoption phase but decelerate as the treated fraction of the diseased population approaches the true prevalence ceiling. Unlike exponential or linear models, the logarithmic form asymptotically saturates — preventing biologically implausible infinite growth in procedural risk.

### 2.6 Step 4: Total Addressable Market (TAM)

The TAM represents the theoretical upper bound on aggregate demand for aortic valve procedures in a given year. It is constructed by applying the projected age-specific risk rates to the projected age-specific populations:

$$
\mathrm{TAM}(t) = \sum_{i \in \mathcal{B}} P_i(t) \cdot \hat{R}_i(t)
$$

where $\mathcal{B} = \{50\text{–}54, \ 55\text{–}59, \ \ldots, \ \geq 80\}$ is the set of age bands.

Two mechanisms drive TAM growth:

- **Risk rate growth** ($\hat{R}_i(t)$), which increases but saturates over time due to the logarithmic model.
- **Demographic expansion**, particularly the rapid growth of the ≥80 cohort. In South Korea, this cohort is projected to increase from approximately 1.1 million (2012) to 7.3 million (2050); in Singapore, from 33,000 to 567,000 over the same period.

The interplay of these two forces produces sustained TAM growth even as individual risk rates plateau.

### 2.7 Step 5: TAVI Adoption Modelling via Constrained Sigmoid

TAVI adoption is modelled as a proportion of TAM, reflecting the technology diffusion process by which TAVI progressively substitutes for SAVR.

#### 2.7.1 Historical TAVI Share Computation

For each historical year $t$, the TAVI share of the market is computed as:

$$
S_{\mathrm{TAVI}}(t) = \frac{\tilde{N}_{\mathrm{TAVI}}(t)}{\mathrm{TAM}(t)}
$$

where $\tilde{N}_{\mathrm{TAVI}}(t)$ is the redistribution-normalised TAVI volume (Section 2.4) and $\mathrm{TAM}(t)$ is the smoothed TAM derived from the projected risk rates (ensuring consistency with the forward projections).

#### 2.7.2 Sigmoid Curve Fitting

TAVI adoption is modelled using a logistic (sigmoid) function with a **fixed upper asymptote** $L_{\max}$, reflecting the assumption that TAVI will not fully replace SAVR but will converge to a maximum market share:

$$
\hat{S}_{\mathrm{TAVI}}(t) = \frac{L_{\max}}{1 + e^{-k(t - t_0)}}
$$

where:

- $L_{\max}$ is the maximum TAVI share of TAM (configurable; default 0.50),
- $k$ is the logistic growth rate (governing the steepness of adoption),
- $t_0$ is the inflection point (year at which adoption reaches $L_{\max}/2$).

Parameters $k$ and $t_0$ are estimated by nonlinear least-squares fitting to the historical share data $\{(t, S_{\mathrm{TAVI}}(t))\}$, restricted to years $t \geq 2016$. Data prior to 2016 are excluded from fitting to focus the curve on the period of established clinical adoption, avoiding early-phase noise from program initiation years.

**Bounds.** Parameter bounds are set as $k \in [0.01, 2.0]$ and $t_0 \in [2000, 2050]$ to ensure physiologically plausible adoption dynamics.

#### 2.7.3 Rationale for Fixed Upper Asymptote

The choice to fix $L_{\max}$ rather than the inflection point $t_0$ reflects several considerations:

1. **Clinical reality** — Certain patient subpopulations (e.g., those with bicuspid anatomy, complex coronary disease, or specific annular geometries) will continue to be preferentially treated with SAVR, precluding 100% TAVI adoption.
2. **Model stability** — Fixing the midpoint while allowing $L$ to be free can produce degenerate fits where the curve becomes nearly linear (infinite $L$ with small $k$), particularly when historical data lie entirely on the early, ascending limb of the sigmoid.
3. **Interpretability** — The maximum share is a clinically meaningful parameter that can be elicited from clinical expert opinion or benchmarked against international experience in mature TAVI markets.

### 2.8 Step 6: TAVI and SAVR Volume Decomposition

The final TAVI and SAVR volume forecasts are obtained by applying the projected adoption share to the TAM:

$$
\hat{N}_{\mathrm{TAVI}}(t) = \hat{S}_{\mathrm{TAVI}}(t) \cdot \mathrm{TAM}(t)
$$

$$
\hat{N}_{\mathrm{SAVR}}(t) = \max\left(0, \ \mathrm{TAM}(t) - \hat{N}_{\mathrm{TAVI}}(t)\right)
$$

This residual formulation captures the substitution dynamics between TAVI and SAVR. As TAVI share increases:

- If $\hat{S}_{\mathrm{TAVI}}(t)$ grows faster than $\mathrm{TAM}(t)$, SAVR volume declines.
- If TAVI share saturates at $L_{\max}$ while demographic forces continue to expand TAM, SAVR volume may stabilise or increase to absorb excess demand.

### 2.9 Downstream Application: Monte Carlo Simulation

The TAVI and SAVR volume forecasts from this framework serve as inputs to a patient-level Monte Carlo simulation (model v12) that estimates future valve-in-valve (ViV) TAVI demand. For each projected index procedure, the simulation draws individual patient-level trajectories including:

- **Prosthetic valve durability** — sampled from age- and valve-type-specific Normal mixture distributions.
- **Patient survival** — sampled from risk-category-specific Normal distributions with optional age-hazard adjustment.
- **Valve failure timing** — computed as the index year plus the drawn durability, discretised to calendar years.
- **ViV candidacy** — determined by requiring valve failure to precede patient death within the forecast horizon.
- **ViV realisation** — subject to a time-varying penetration rate reflecting clinical uptake of the ViV procedure.

Multiple independent Monte Carlo runs (default: 20) are executed and averaged to produce mean estimates with uncertainty bounds.

### 2.10 Software Implementation

The model is implemented in Python 3 using NumPy, pandas, SciPy (for curve fitting), and Matplotlib (for visualisation). The pipeline is orchestrated via a command-line interface that sequentially executes index projection (this framework) and Monte Carlo simulation, with configuration specified in YAML files. All model parameters — including the maximum TAVI share, fitting window, and redistribution settings — are exposed as configurable inputs, enabling sensitivity analysis and scenario exploration.

---

## 3. Results

The forecasting framework was applied to two countries — South Korea and Singapore — using historical registry data from 2012 to 2024 and UN population projections through 2050. Unless otherwise stated, the TAVI maximum adoption share was set to $L_{\max} = 0.50$ and the sigmoid was fitted to data from 2016 onwards.

### 3.1 Historical Data Profile

Table 1 summarises the historical procedure volumes for both countries at selected years.

**Table 1. Historical TAVI and SAVR Procedure Counts (Selected Years, 2012–2024)**

| Year | South Korea TAVI | South Korea SAVR | South Korea Total | Singapore TAVI | Singapore SAVR | Singapore Total |
| ---- | ---------------- | ---------------- | ----------------- | -------------- | -------------- | --------------- |
| 2012 | 0                | 1,531            | 1,531             | 30             | 14             | 44              |
| 2016 | 260              | 2,179            | 2,439             | 52             | 216            | 268             |
| 2020 | 763              | 2,195            | 2,958             | 95             | 208            | 303             |
| 2024 | 1,862            | 2,158            | 4,020             | 143            | 326            | 469             |

South Korea had substantially higher absolute volumes, reflecting its larger population (approximately 52 million vs 5.9 million). TAVI was introduced in Korea in 2015 (61 procedures) and grew rapidly to 1,862 by 2024 — a 31-fold increase in nine years. Singapore had a steady but smaller TAVI programme from 2012 (30 procedures), growing more gradually to 143 by 2024 — a 4.8-fold increase over twelve years.

### 3.2 Pandemic Redistribution (Step 2)

The volume-preserved redistribution corrected for COVID-19 disruptions in the 2020–2023 window.

**South Korea.** The scaling factor for combined volumes was $\kappa = 1.071$, indicating that total observed pandemic-era volume exceeded the pre-pandemic linear trend by 7.1%. This is consistent with pent-up demand and the expansion of TAVI indications during the recovery period.

**Singapore.** The scaling factor was $\kappa = 0.734$, indicating that pandemic-era combined volumes fell 26.6% below the pre-pandemic trend — a considerably larger disruption than in Korea. This is consistent with Singapore's stringent public health measures, including extended elective surgery deferrals during the circuit-breaker and heightened-alert periods.

### 3.3 Risk Rate Projections (Step 3)

Logarithmic saturation models were fitted independently for each of the seven age bands (50–54 through ≥80). In both countries, the ≥80 age band exhibited the highest absolute risk rates and the steepest historical growth. By construction, all age-band risk trajectories exhibit diminishing growth rates over the projection horizon, preventing biologically implausible extrapolation beyond the prevalence ceiling.

### 3.4 Total Addressable Market (Step 4)

Table 2 presents the projected TAM at selected time horizons, together with the contribution of the ≥80 cohort.

**Table 2. Projected Total Addressable Market (TAM) and ≥80 Cohort Contribution**

| Year                 | Korea TAM | Korea ≥80 TAM Contribution | Korea ≥80 Share | Singapore TAM | Singapore ≥80 TAM Contribution | Singapore ≥80 Share |
| -------------------- | --------- | -------------------------- | --------------- | ------------- | ------------------------------ | ------------------- |
| 2024                 | 3,768     | 1,269                      | 33.7%           | 413           | 176                            | 42.6%               |
| 2030                 | 5,182     | 1,948                      | 37.6%           | 606           | 286                            | 47.3%               |
| 2040                 | 7,935     | 3,908                      | 49.2%           | 1,004         | 578                            | 57.5%               |
| 2050                 | 10,189    | 6,178                      | 60.6%           | 1,389         | 835                            | 60.1%               |
| **Growth 2024→2050** | **×2.7**  | **×4.9**                   | —               | **×3.4**      | **×4.7**                       | —                   |

The TAM is projected to grow 2.7-fold in Korea and 3.4-fold in Singapore between 2024 and 2050. In both countries, the ≥80 cohort is the overwhelmingly dominant driver:

- **South Korea**: The ≥80 population is projected to grow from 2.4 million (2024) to 7.3 million (2050) — a 3.0-fold increase — driving the ≥80 TAM contribution from 33.7% to 60.6%.
- **Singapore**: The ≥80 population is projected to grow from 159,000 (2024) to 567,000 (2050) — a 3.6-fold increase — driving the ≥80 TAM contribution from 42.6% to 60.1%.

The convergence of both countries to approximately 60% ≥80 TAM share by 2050 reflects a shared demographic trajectory characteristic of developed Asian economies undergoing rapid population ageing.

### 3.5 TAVI Adoption Sigmoid Fit (Step 5)

The constrained sigmoid ($L_{\max} = 0.50$) was fitted to historical TAVI share of TAM for data from 2016 onwards. Table 3 presents the fitted parameters.

**Table 3. Sigmoid Curve Parameters for TAVI Share of TAM**

| Parameter               | South Korea | Singapore | Interpretation                |
| ----------------------- | ----------- | --------- | ----------------------------- |
| $L_{\max}$ (fixed)      | 0.50        | 0.50      | Maximum TAVI market share     |
| $k$ (growth rate)       | 0.502       | 0.096     | Speed of adoption transition  |
| $t_0$ (inflection year) | 2018.4      | 2015.1    | Year of half-maximum adoption |
| TAVI share at 2024      | 47.1%       | 35.1%     | Current adoption level        |
| Year to reach 45% share | ~2023       | ~2040     | Near-saturation timing        |

The two countries exhibit markedly different adoption dynamics:

- **South Korea** shows a steep sigmoid ($k = 0.502$) with an inflection point at 2018.4, indicating that TAVI adoption was already in its rapid-growth phase during the observation period. By 2024, the fitted model estimates a TAVI share of 47.1% — approaching the 50% ceiling — suggesting that Korea is nearing adoption saturation under this parameterisation.

- **Singapore** exhibits a much more gradual sigmoid ($k = 0.096$) with an earlier inflection point (2015.1). The shallower curve reflects the slower absolute pace of TAVI adoption in Singapore, where TAVI share stood at approximately 35.1% in 2024. The model projects that Singapore will not approach the 50% ceiling until approximately 2040.

### 3.6 TAVI and SAVR Volume Forecasts (Step 6)

Table 4 presents the final decomposition of TAM into TAVI and SAVR volumes at selected horizons.

**Table 4. Projected TAVI and SAVR Procedure Volumes**

| Year | Korea TAVI | Korea SAVR | Korea Total | Singapore TAVI | Singapore SAVR | Singapore Total |
| ---- | ---------- | ---------- | ----------- | -------------- | -------------- | --------------- |
| 2024 | 1,776      | 1,992      | 3,768       | 145            | 268            | 413             |
| 2030 | 2,583      | 2,599      | 5,182       | 244            | 362            | 606             |
| 2035 | 3,257      | 3,259      | 6,516       | 348            | 451            | 799             |
| 2040 | 3,967      | 3,967      | 7,935       | 460            | 544            | 1,004           |
| 2050 | 5,095      | 5,095      | 10,189      | 670            | 718            | 1,389           |

Several notable findings emerge from the volume projections:

1. **Korea reaches TAVI–SAVR volume parity by ~2030.** The steep Korean sigmoid results in TAVI and SAVR volumes converging at approximately 2,600 each by 2030. Beyond this point, both modalities grow in lockstep at half the TAM.

2. **Singapore approaches parity by ~2050.** The more gradual Singaporean adoption curve delays the crossover; by 2050, TAVI (670) approaches but does not exceed SAVR (718), with full parity projected shortly thereafter.

3. **Both TAVI and SAVR grow in absolute terms.** This is a critical finding: despite TAVI substituting for SAVR in terms of relative share, the demographic expansion of the elderly population is sufficiently strong that SAVR volumes are projected to increase — not decrease — in both countries. In Korea, SAVR grows from 1,992 (2024) to 5,095 (2050); in Singapore, from 268 to 718. This has important implications for surgical workforce planning and training, as demand for SAVR capacity will not diminish even as TAVI becomes the dominant modality.

4. **Asymptotic equilibrium.** As the TAVI adoption share approaches $L_{\max} = 0.50$, both TAVI and SAVR volumes converge to approximately half of the TAM. Korea reaches this equilibrium by the mid-2030s, while Singapore reaches it by ~2050. In the post-equilibrium phase, total volume growth is driven entirely by demographic forces rather than technology adoption.

### 3.7 Sensitivity to the Maximum Share Assumption

The $L_{\max}$ parameter exerts a first-order effect on the projected volume split. Under alternative assumptions:

- **$L_{\max} = 0.75$**: TAVI would constitute three-quarters of the market at maturity, producing considerably lower SAVR volumes. This scenario reflects potential expansion of TAVI to lower-risk populations currently treated with SAVR.
- **$L_{\max} = 0.30$**: TAVI adoption would plateau at 30%, preserving SAVR as the dominant approach. This scenario might apply in settings with limited catheterisation laboratory infrastructure or restrictive reimbursement policies.

The model architecture allows $L_{\max}$ to be adjusted as a single parameter, enabling rapid scenario-based capacity planning without re-fitting the underlying demographic or risk models.

---

## 4. Discussion

### 4.1 Summary of Principal Findings

This study presents a demographically anchored forecasting framework for TAVI and SAVR procedure volumes and applies it to two Asian healthcare systems — South Korea and Singapore — that are at different stages of the TAVI adoption curve. Three principal findings emerge:

First, **demographic forces will drive sustained growth in total aortic valve procedure demand regardless of TAVI adoption dynamics.** The TAM — representing combined TAVI + SAVR demand — is projected to grow 2.7-fold in Korea and 3.4-fold in Singapore between 2024 and 2050, driven overwhelmingly by the ≥80 population expansion. Both countries converge to approximately 60% of TAM originating from the ≥80 cohort by 2050, reflecting the universal demographic transition of developed Asian economies.

Second, **SAVR volumes are projected to grow in absolute terms despite TAVI substitution.** This counterintuitive finding arises because demographic expansion outpaces the substitution effect under most reasonable adoption ceilings ($L_{\max} \leq 0.50$). Even as TAVI captures half the market, the doubling or tripling of total demand means that SAVR volumes in 2050 are projected to exceed current levels — a finding with important implications for surgical workforce planning.

Third, **South Korea and Singapore exhibit markedly different adoption dynamics.** Korea's steep sigmoid ($k = 0.502$) with near-saturation by 2024 contrasts sharply with Singapore's gradual curve ($k = 0.096$), which does not approach the adoption ceiling until approximately 2040. These differences likely reflect distinct healthcare system factors including reimbursement timing, regulatory frameworks, centre accreditation policies, and the relative scale of catheterisation laboratory infrastructure.

### 4.2 Interpretation in Context

#### 4.2.1 TAVI Adoption Patterns: Asia vs. the West

The adoption curves observed in Korea and Singapore represent two distinct paradigms within the broader Asian context. Korea's steep sigmoid is consistent with the pattern observed in Germany and other early-adopting European countries, where national reimbursement coverage triggered rapid TAVI uptake that quickly approached or exceeded 50% market share. The Korean curve, however, is compressed in time — achieving in under a decade what took European centres 12–15 years — suggesting that late adopters may benefit from accumulated clinical evidence, established training pathways, and second-generation device platforms that reduce barriers to uptake.

Singapore's more gradual curve is consistent with patterns in Japan and other Asian healthcare systems with more conservative regulatory or reimbursement positions. The lower growth rate ($k = 0.096$) may reflect a clinical culture that preferentially selects TAVI for high-risk patients while continuing to direct intermediate- and low-risk patients toward SAVR. This pattern is clinically defensible given the current absence of long-term (>10 year) randomised trial data in low-risk populations.

#### 4.2.2 The Demographic Multiplier

The finding that demographic forces dominate the long-term trajectory of total procedure demand — even as adoption dynamics govern the TAVI/SAVR split — has not been sufficiently emphasised in the existing literature. Most forecasting studies focus on the adoption curve in isolation, implicitly assuming a stable denominator. Our framework makes explicit the interaction between two multiplicative forces:

$$
\hat{N}_{\mathrm{TAVI}}(t) = \underbrace{\hat{S}_{\mathrm{TAVI}}(t)}_{\text{Adoption (saturates)}} \times \underbrace{\mathrm{TAM}(t)}_{\text{Demographics (grows)}}
$$

Because the adoption share saturates while the TAM continues to grow, the long-run growth rate of TAVI volumes is governed by the demographic growth rate, not the adoption rate. This has a practical consequence: even if TAVI adoption were to freeze at current levels, TAVI volumes would still grow by approximately 2–3× by 2050 solely due to population ageing. Adoption is a transient phenomenon; demographics are structural.

#### 4.2.3 Implications for SAVR and Surgical Workforce

The projection that SAVR volumes will grow in absolute terms despite losing relative market share is a critical planning insight. In many healthcare systems, workforce planning has assumed that the growth of TAVI will produce a corresponding decline in SAVR, potentially reducing the need for cardiac surgical training positions and operating theatre capacity. Our results challenge this assumption:

- In Korea, SAVR volume is projected to grow from approximately 1,992 (2024) to 5,095 (2050) — a 2.6-fold increase.
- In Singapore, SAVR grows from 268 to 718 — a 2.7-fold increase.

These projections suggest that cardiac surgical capacity will need to expand, not contract, to meet future demand — even in a world where TAVI is the dominant modality. This finding is particularly relevant for training programme design, as the pipeline of cardiac surgeons competent in aortic valve surgery will need to grow proportionally.

#### 4.2.4 Implications for Valve-in-Valve Demand

The TAVI and SAVR volume projections generated by this framework serve as direct inputs to the downstream Monte Carlo simulation of ViV TAVI demand. The key insight for ViV planning is that the cumulative pool of implanted bioprostheses will grow substantially:

- **Cumulative TAVI implants (Korea, 2024–2050)**: approximately 94,000 valves.
- **Cumulative SAVR implants (Korea, 2024–2050)**: approximately 95,000 valves.
- **Combined**: nearly 190,000 bioprostheses requiring surveillance for SVD.

Given typical bioprosthetic durability of 10–20 years, the first significant wave of ViV demand from current-generation TAVI implants is projected to emerge in the early 2030s, peaking in the 2040s. The scale of this wave is directly proportional to the index procedure volumes projected by this framework, underscoring the importance of accurate upstream forecasting.

### 4.3 Methodological Considerations

#### 4.3.1 Pandemic Redistribution

The volume-preserved redistribution method offers a pragmatic solution to a ubiquitous challenge in pandemic-era epidemiological analysis. By preserving total observed volume while redistributing it to follow the pre-pandemic trend shape, the method avoids two common pitfalls: (1) including raw pandemic data that biases trend estimates, and (2) excluding pandemic years entirely and discarding volume information. The contrasting scaling factors — Korea $\kappa = 1.071$ (above trend) vs Singapore $\kappa = 0.734$ (below trend) — provide a quantitative measure of pandemic impact that is itself informative for health system resilience assessment.

#### 4.3.2 Logarithmic Risk Rate Saturation

The choice of a logarithmic functional form for risk rate projection, rather than linear or polynomial models, is motivated by the biological reality that treatment rates cannot grow indefinitely. The prevalence of severe AS sets an upper bound on the fraction of any age cohort that can benefit from intervention. The logarithmic form $R(t) = a + b \ln(t - t_0)$ naturally produces decelerating growth, asymptotically approaching but never exceeding the prevalence ceiling. This is more biologically plausible than linear models (which predict unbounded growth) or logistic models (which require specification of the upper asymptote, which is unknown).

#### 4.3.3 Fixed vs. Free Maximum Share

The decision to fix $L_{\max}$ rather than allow it to be freely estimated deserves particular attention. When historical data lie entirely on the ascending limb of the sigmoid — as is the case for most Asian TAVI programmes — the upper asymptote is poorly identified by the data, and free estimation can produce degenerate solutions where $L_{\max} \to \infty$ with $k \to 0$ (a linear fit masquerading as a sigmoid). Fixing $L_{\max}$ at a clinically justified value (0.50) stabilises the fit and ensures that the two remaining parameters ($k$, $t_0$) capture meaningful adoption dynamics. The sensitivity analysis (Section 3.7) demonstrates that the impact of alternative $L_{\max}$ choices is transparent and easily separable from the demographic forecast.

### 4.4 Limitations

Several limitations should be acknowledged:

1. **Fixed maximum share assumption.** The use of a fixed $L_{\max} = 0.50$ assumes that TAVI will capture at most half of the aortic valve market. In reality, the ceiling may evolve as clinical evidence accumulates, valve technology improves, and indications expand. If low-risk trial data continue to favour TAVI beyond 10-year follow-up, the true ceiling could exceed 0.50, and our SAVR projections would be overestimates. Conversely, if long-term structural valve deterioration proves more prevalent than anticipated, the ceiling could be lower.

2. **Deterministic TAM projection.** The TAM is computed as a deterministic product of projected populations and risk rates, without uncertainty quantification. In practice, population projections carry uncertainty (driven by fertility, mortality, and migration assumptions), and risk rate convergence to prevalence ceilings may be faster or slower than the logarithmic model predicts. A future extension could incorporate prediction intervals by propagating demographic uncertainty through the model.

3. **Country-level granularity.** The model operates at the national level and does not account for within-country geographic heterogeneity in access to TAVI. In large countries like South Korea, regional variation in centre availability, referral patterns, and patient willingness to travel may produce local demand patterns that differ substantially from the national aggregate.

4. **Static risk mix.** The model assumes that the age-specific risk rate captures both incidence and treatment propensity as a single quantity. It does not separately model changes in treatment thresholds, clinical guidelines, or patient selection criteria that could shift the risk profile of treated patients over time.

5. **Limited validation data.** With only 9–12 years of historical data, out-of-sample validation is constrained. Cross-validation against withheld years (e.g., fitting to 2012–2020 and validating against 2021–2024) could provide some assessment of predictive performance, but the short forecast horizon reduces the power of such tests.

6. **Absence of economic and policy dynamics.** The model does not explicitly account for policy changes such as reimbursement expansions, new device approvals, or changes in centre accreditation criteria — all of which could accelerate or decelerate adoption. These are implicitly captured to the extent that they are reflected in the historical data used for sigmoid fitting.

### 4.5 Future Directions

Several extensions of this framework are under development or consideration:

1. **Uncertainty quantification.** Incorporating bootstrap or Bayesian approaches to estimate confidence intervals on the sigmoid parameters, propagate demographic uncertainty, and produce probabilistic volume forecasts.

2. **Additional countries.** The framework is explicitly designed to be country-agnostic. Application to Japan, Taiwan, and other Asian healthcare systems would enable cross-country comparison of adoption dynamics and identification of system-level factors that accelerate or delay TAVI diffusion.

3. **Dynamic $L_{\max}$.** Replacing the fixed maximum share with a time-varying ceiling that responds to guideline changes, trial results, or policy interventions. This could be implemented as a piecewise function or as a second-level sigmoid.

4. **Integration with ViV forecasting.** The volume projections from this framework are already used as inputs to a patient-level Monte Carlo simulation of ViV demand. Tighter integration — where ViV demand feeds back to influence repeat procedure capacity — would create a fully closed-loop forecasting system.

5. **Sub-national modelling.** Extension to province- or city-level forecasting, incorporating geographic access models and centre-specific capacity data, would provide more actionable planning inputs for hospital administrators and health authorities.

---

## 5. Conclusion

We have presented a six-step, demographically anchored forecasting framework for TAVI and SAVR procedure volumes that explicitly separates technology adoption dynamics from the demographic forces driving total demand. Applied to South Korea and Singapore, the framework reveals three key insights: (1) total aortic valve procedure demand will grow 2.7–3.4× by 2050, driven primarily by the expansion of the ≥80 population; (2) SAVR volumes will grow in absolute terms despite declining market share, challenging assumptions that TAVI growth will reduce surgical demand; and (3) the two countries exhibit markedly different adoption trajectories, reflecting distinct healthcare system characteristics despite shared demographic trends.

The framework provides a transparent, parameterised platform for scenario-based capacity planning. By adjusting a single parameter — the maximum TAVI adoption share — planners can rapidly assess the implications of alternative technology diffusion scenarios for both interventional cardiology and cardiac surgical capacity. The resulting volume projections serve as essential upstream inputs for forecasting the emerging burden of valve-in-valve reintervention, a critical planning challenge for the coming decades.

---
