# Methodology: Volume-Preserved Trend Redistribution for Pandemic-Era Data Normalization

## 1. Abstract

This document details the statistical methodology used to normalize procedure volumes during the COVID-19 pandemic period (2020–2023). Standard trend extrapolation fails to account for the complex dynamics of the pandemic, which typically include a suppression of cases (dip) followed by a compensatory recovery (spike). Our approach, **Volume-Preserved Trend Redistribution**, normalizes the temporal distribution of these cases to match the underlying secular trend while rigorously preserving the total observed cumulative volume. This ensures that the forecasting model is trained on the "true" underlying demand signal without violating mass conservation principles.

## 2. Problem Statement

The raw procedure data $O(t)$ exhibits significant non-biological volatility during the window $W = [2020, 2021, 2022, 2023]$.

1.  **Suppression Phase (2020–2022)**: $O(t)$ falls below the expected trend due to hospital access restrictions.
2.  **Recovery Phase (2023)**: $O(t)$ spikes above the expected trend due to the treatment of backlog cases.

Using $O(t)$ directly for forecasting introduces noise. Simple trend replacement (ignoring actuals) violates the reality of the total disease burden processed. A robust normalization $N(t)$ is required.

## 3. Methodology Derivation

We define the normalized series $N(t)$ under two constraints:

### Constraint A: Shape Consistency

The normalized values for the pandemic window must follow the shape of the pre-pandemic secular trend. This assumes that without the external shock, adoption would have continued its established linear trajectory.
$$ N(t) \propto T(t) \quad \text{for } t \in W $$
Where $T(t)$ is the trend derived from the pre-COVID baseline (2012–2019).

### Constraint B: Conservation of Volume

The normalization process must not create or destroy patients. The total number of procedures performed in the window $W$ must remain unchanged. This acknowledges that the "spike" in 2023 was largely composed of "dip" patients from previous years.
$$ \sum*{t \in W} N(t) = \sum*{t \in W} O(t) $$

## 4. Algorithm Steps

The algorithm is implemented as follows:

### Step 1: Baseline Trend Fitting

We isolate the Pre-COVID history ($H = [2012, ..., 2019]$) and fit a linear regression model to established the "Natural Trend" $T(t)$.
$$ T(t) = \alpha + \beta t $$
Where $\alpha, \beta$ are minimized via Ordinary Least Squares on the domain $H$.

### Step 2: Volume Aggregation

We calculate the Total Observed Volume ($V_{obs}$) and the Total Expected Trend Volume ($V_{trend}$) for the window $W$.
$$ V*{obs} = \sum*{t=2020}^{2023} O(t) $$
$$ V*{trend} = \sum*{t=2020}^{2023} T(t) $$

### Step 3: Scaling Factor Derivation

We derive a uniform scalar $k$ that adjusts the trend magnitude to match the observed volume.
$$ k = \frac{V*{obs}}{V*{trend}} $$

### Step 4: Redistribution

We generate the normalized values $N(t)$ by applying the scalar to the trend projection.
$$ N(t) = k \cdot T(t) \quad \text{for } t \in W $$

## 5. Interpretation

If $k > 1$, it implies that the pandemic period actually saw _higher_ total intensity than the pre-2019 trend predicted (despite the dips), necessitating an upward shift of the trend line.
If $k < 1$, it implies the pandemic caused a permanent loss of cases that were not fully recovered in 2023, necessitating a downward shift.

In our specific case (TAVI+SAVR), we observed $k \approx 1.07$, indicating that even with the pandemic, the total volume in 2020-2023 slightly exceeded what the 2012-2019 linear trend predicted. This method allows us to feed a smooth, monotonically increasing curve into the forecasting model that faithfully represents 100% of the clinical activity that occurred.

## 6. Visualization

The plot below demonstrates the effect of this normalization. The Blue Line (Normalized) preserves the area under the curve of the Black Dashed Line (Raw) for the 2020-2023 period.

![Normalization Effect](step2_normalization_effect.png)
