# Methodology: Infrastructure-Constrained TAVI Adoption Model (Center-Driven)

## 1. Abstract

This document details the methodology for projecting TAVI adoption. Unlike traditional "diffusion of innovation" models that treat adoption as a purely demand-side phenomenon, we explicitly model the **Supply-Side Constraints**. We posit that the growth of Transcatheter Aortic Valve Implantation (TAVI) in South Korea is rate-limited by the physical infrastructure—specifically, the number of certified TAVI centers. Therefore, our projection is constructed as the product of _Infrastructure Capacity_ and _Center Efficiency_.

## 2. Problem Statement

We aim to project the total annual TAVI volume $T(t)$ for $t \in [2025, ..., 2050]$.
Directly fitting a sigmoid curve to historical TAVI volume risks overfitting to early exponential growth without accounting for physical saturation limits. A more robust approach couples volume to the growing network of hospitals.

## 3. Mathematical Model

We decompose TAVI volume into two independent components:

$$ T(t) = C(t) \times V\_{pc}(t) $$

Where:

- $C(t)$: The projected number of active TAVI Centers in year $t$.
- $V_{pc}(t)$: The projected "Volume Per Center" (Throughput) in year $t$.

### Component A: Center Capacity Projection ($C(t)$)

The expansion of hospital networks typically follows a **Logistic (Sigmoidal) Growth** pattern: slow initial certification, a period of rapid expansion, and final saturation as the market fills.

$$ C(t) = \frac{L}{1 + e^{-k(t - t_0)}} $$

- **$L$ (Carrying Capacity)**: Fixed at **94 Centers**. This represents the theoretical maximum number of institutions in South Korea capable of supporting a TAVI program.
- **$k$ (Growth Rate)** & **$t_0$ (Midpoint)**: Fitted to historical hospital registry data.

### Component B: Center Efficiency Projection ($V_{pc}(t)$)

We calculate the historical efficiency (average procedures per center) for the observed period 2012-2024:
$$ V*{pc}^{obs}(t) = \frac{T^{obs}(t)}{C^{obs}(t)} $$
We then project this efficiency forward using a **Linear Trends Analysis**.
$$ V*{pc}(t) = \alpha + \beta t $$
This captures clinical maturation—as centers gain experience, they become more efficient and can handle higher volumes, but this growth is linear rather than exponential.

## 4. Resulting Dynamics

This coupled inequality explains the shape of the final TAVI curve:

1.  **Growth Phase**: Both $C(t)$ (new centers opening) and $V_{pc}(t)$ (increasing efficiency) are rising, leading to multiplicative growth in total volume.
2.  **Saturation Phase**: As $C(t)$ approaches the cap of 94 centers (determined by the Green Line in the plot), the "infrastructure multiplier" stops growing.
3.  **Maturation Phase**: Future growth becomes solely dependent on $V_{pc}(t)$ (efficiency gains), resulting in a distinct "bend" or deceleration in the total TAVI curve (Blue Line).

## 5. Visualization

- **Blue Line**: Total Projected TAVI Volume.
- **Green Dashed Line**: Projected Number of Centers (The Driver).
- **Insight**: Note how the TAVI volume curve inflects and flattens exactly as the Center count hits its saturation plateau.

![Center-Driven TAVI Projection](step5_tavi_projection.png)
