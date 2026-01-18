# Presentation Script: Forecasting ViV TAVI in Korea (2025-2035)

## Section 1: Introduction & Refresher
**Goal:** Re-orient the audience (Hyunjin and Prof) on the study's history and original direction.

*   **Opening**: "It’s been a while since we convened. Thank you for your patience while Hyunjin and I conducted this deep dive."
*   **The Original Goal**: To estimate the volume of Valve-in-Valve (ViV) TAVR procedures in Korea and Singapore.
*   **The Inspiration**:
    *   Modeled after **Genereux et al.** ("Predicting Treatment of Bioprosthetic Aortic Valve Failure in the US").
    *   Also **Ohno et al.** ("Predicting the Surge in ViV Volume: Japan vs US").
    *   *Reference*: These papers used 2023 TVT Registry data to generate linear forecast charts.
*   **Our First Pass**:
    *   We initially replicated this exact methodology using Korean registry data (OpenData HIRA, up to 2023).
    *   **Old Model Features**:
        *   Fixed SAVR rates post-2023.
        *   Extrapolated TAVR rates based on past uptake.
        *   **Speculative Penetration**: Assumed 10% uptake in 2022 scaling to 60% in 2035.
    *   *Result*: It produced a similar trend to the US/Japan papers, but we found errors (understated numbers) and, more importantly, methodological flaws.

## Section 2: The Critique & The Pivot
**Goal:** Explain why we abandoned the old method. It wasn't just "wrong numbers," it was "wrong philosophy."

*   **The Catalyst (Oct 7th Editorial)**:
    *   An editorial comment on Ohno’s paper ("The Past, Present, and Future") raised striking criticisms.
    *   **Key Critique**: "Blind extrapolation introduces bias." It ignores clinical factors (e.g., small annulus in Asians) and assumes infinite growth.
    *   *Quote*: "Interpretation of such predictions requires caution... crucial factors were not accounted for."
*   **Our Internal Discomfort**:
    *   **1. The 2023 Spike**: Korean data showed a massive spike in 2023.
        *   Likely due to **Post-COVID Backlog** (patients returning).
        *   **Insurance Changes**: 2024 policy shifts impacted decision-making.
        *   *Problem*: Linearly extrapolating from this 2023 outlier creates an "infinite growth" curve that is physically impossible.
    *   **2. The Aging Factor**: Korea has a unique, rapid aging velocity. A model that ignores this demographic reality is inappropriate.

## Section 3: The New Approach (Methodology)
**Goal:** Detail the three major changes in the new `model_v9`.

### Change 1: From "Extrapolation" to "Risk × Demography"
*   **Old Way**: "TAVI grew 10% last year, so it will grow 10% next year."
*   **New Way**: **Age-Sex Specific Risk Anchoring**.
    *   We anchor ONLY to **2023 and 2024**.
    *   We split registry counts by **Sex** and **5-Year Age Bands**.
    *   We calculate the **Per-Capita Risk** (Procedures per Person) for each band.
    *   *Logic*: We assume the *probability* of a 75-year-old man getting TAVI stays constant. What changes is the *number* of 75-year-old men.
    *   **[Display Figure 3: Nationally projected population trends]**
    *   **[Display Figure 5: Compiled risks for index SAVR/TAVI by age-band]**

### Change 2: Removing Speculative Penetration
*   **The Issue**: Genereux/Ohno assumed arbitrary curves (e.g., 0% to 40% uptake).
*   **Our Fix**: We removed this entirely.
    *   We do not have robust ViV data in Korea to calibrate such a curve.
    *   Importing US/Japan curves would be misleading.
*   **The New Metric**: **"ViV-Eligible Candidates"**.
    *   Patients who have a failed valve + are still alive + have NOT had a Redo-SAVR.
    *   We stop here. This is the "Demand Floor."

### Change 3: The Monte Carlo Engine
*   We still use the robust simulation engine:
    *   Allocate ages to index procedures.
    *   Sample **Durability** (Bimodal: Early vs Late failure).
    *   Sample **Survival** (Risk-dependent curves).
    *   **The Race**: Patients are dropped if they die before their valve fails.
    *   **Redo-SAVR**: We remove these from the pool, but project them using the same Risk × Demography framework, not ad-hoc guesses.

## Section 4: Results
**Goal:** Present the new numbers and compare them to the old "hype."

*   **The Forecast (2025-2035)**:
    *   **[Display Figure 1: Simulated ViV-TAVI candidates vs realised]**
    *   **Trend**: Steady increase.
    *   **Numbers**: From **791 candidates** in 2024 to **1,973 candidates** in 2035.
    *   **Growth**: A **2.5-fold increase**.
*   **Comparison vs Ohno/Genereux**:
    *   **[Display Figure 2: Predicted realised ViV patient volume]**
    *   If we *did* apply the same penetration rates just for comparison:
        *   Our volume grows **3-fold** (325 to 1078).
        *   Ohno/Genereux predicted **7-9 fold** increases.
    *   *Conclusion*: Our demographic anchor significantly tempers the "infinite scaling" seen in other papers. It is a more realistic, conservative baseline.

## Section 5: Supplementary Data
**Goal:** Show robustness of inputs.

*   **[Display Figure 4: Index procedures observed vs projected]**
    *   Shows how our "Risk × Demography" model projects index TAVI/SAVR volume forward. It smooths out the 2023 spike.
*   **[Display Figure 6: Waterfall plots]**
    *   (Optional) Breakdown of how we derive ViV-eligible patients from the total pool.

## Section 6: Moving Forward
**Goal:** Define the roadmap.

1.  **Singapore Data**: Inspecting now to see if this exact pipeline applies.
2.  **Publication Strategy**:
    *   Target: **JACC** (addressing the editorial directly?).
    *   Metrics: How do we best compare our "Candidates" vs their "Realized"?
3.  **Validation**: Gathering feedback on this non-conventional "Risk × Demography" approach.
4.  **Reproducibility**: Cleaning the code repo for reviewers.
