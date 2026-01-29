
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.optimize import curve_fit
import sys

# Setup basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

def parse_age_band(band: str, open_width: int = 20):
    """
    Parse age-band labels like '50-54', '>=80'.
    Returns (lo, hi_exclusive).
    """
    s = str(band).strip().lower()
    s = s.replace("≥", ">=").replace("<", "").replace(">", "").replace("=", "") # simplistic cleanup
    
    if "-" in s:
        parts = s.split("-")
        return int(parts[0]), int(parts[1]) + 1
    
    # Handle single numbers like '80' (often >=80) or '5' (<5)
    # This is a bit brittle, relying on context, but let's assume raw numbers
    try:
        val = int(s)
        if val >= 70: # High assumption
            return val, val + open_width
        return 0, val
    except:
        return 0, 0

def sigmoid(x, L, k, x0):
    """
    Sigmoid function: L / (1 + exp(-k*(x-x0)))
    """
    return L / (1 + np.exp(-k * (x - x0)))

def fit_sigmoid_share(years, shares, max_share=0.95):
    """
    Fit a sigmoid to market share data.
    Constrain L (max share) to be reasonable (e.g. <= 0.95 or 1.0).
    """
    # p0 = [L, k, x0]
    # L ~ 0.9, k ~ 0.5, x0 ~ 2020
    p0 = [max(shares), 0.5, np.mean(years)]
    bounds = ([0.5, 0.01, 2000], [max_share, 2.0, 2050])
    
    try:
        popt, _ = curve_fit(sigmoid, years, shares, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except Exception as e:
        log.warning(f"Sigmoid fit failed: {e}. Fallback to linear extrapolation.")
        return None

def fit_linear(years, values):
    z = np.polyfit(years, values, 1)
    return z # slope, intercept

def load_population_korea_combined(csv_path):
    """
    Load the korea_population_combined.csv which has format:
    Index (Age Band), 2022, 2023 ...
    Rows include: 0-14, 15-24, 25-49, 50-64, >=65, >=70, >=75, >=85
    We need to derive non-overlapping counts for weighting.
    Units appear to be 10,000s (Sum ~5167 -> 51.6M).
    """
    df = pd.read_csv(csv_path, index_col=0)
    # Transpose: Index = Year, Cols = Age Bands
    df = df.T
    df.index.name = "year"
    df.index = df.index.astype(int)
    
    # Check available columns
    # Expected: '≥65', '≥70', '≥75', '≥85'
    # Need to normalize column names (remove unicode)
    df.columns = [str(c).replace("≥", ">=").strip() for c in df.columns]

    # Calculate buckets
    # 65-69 = (>=65) - (>=70)
    # 70-74 = (>=70) - (>=75)
    # 75-84 = (>=75) - (>=85)
    # 85+   = (>=85)
    # Under 65 = Total - (>=65)? We don't have Total.
    # We have 0-14, 15-24, 25-49, 50-64.
    # Sum of [0-14, 15-24, 25-49, 50-64] + [>=65] should be Total.
    
    # Let's trust the ">=65", ">=70", etc logic for the elderly weights
    
    # Helper to get (handle missing safely)
    def get_col(c): return df[c] if c in df.columns else  pd.Series(0, index=df.index)
    
    pop_65_plus = get_col(">=65")
    pop_70_plus = get_col(">=70")
    pop_75_plus = get_col(">=75")
    pop_85_plus = get_col(">=85")
    
    pop_65_69 = pop_65_plus - pop_70_plus
    pop_70_74 = pop_70_plus - pop_75_plus
    pop_75_84 = pop_75_plus - pop_85_plus
    pop_85_up = pop_85_plus
    
    # Under 65s (low weight)
    pop_u65 = get_col("0-14") + get_col("15-24") + get_col("25-49") + get_col("50-64")
    
    # Scale units (x 10,000)
    scale = 10000
    
    # Calculate Weighted Population directly here
    # Weights: <65 (0.01), 65-69 (0.1), 70-74 (0.4), 75-84 (0.8), 85+ (1.0)
    w_pop = (
        pop_u65 * 0.01 +
        pop_65_69 * 0.10 +
        pop_70_74 * 0.40 +
        pop_75_84 * 0.80 +
        pop_85_up * 1.00
    ) * scale
    
    return pd.DataFrame({"year": df.index, "weighted_pop": w_pop.values})

def load_population_un_flexible(male_path, female_path):
    # Load whatever is in there, assuming it's the "Korea" file we have
    # Wide format, read, melt
    def read_melt(path):
        df = pd.read_csv(path)
        # Find age columns: they are digits or ends in +
        # Columns 10 onwards?
        # Header: Index, Variant, ..., Year, 0-4, ...
        # Based on head output: Year is col 10 (0-indexed). 0-4 is 11.
        
        # Filter for Variant = 'Estimates' (History)
        # Assuming Variant is column 1 (based on previous inspections)
        # But using column name is safer. 
        # Column names: Index,Variant,...
        if "Variant" in df.columns:
            df = df[df["Variant"] == "Estimates"]
        
        # Melt
        id_vars = ["Year"]
        # Filter cols that are Age Bands
        age_cols = [c for c in df.columns if c[0].isdigit()]
        
        # Keep only relevant cols
        df = df[["Year"] + age_cols]
        # Strict deduplication before melt (keep first row per Year)
        df = df.groupby("Year").first().reset_index()
        
        df = df.melt(id_vars=["Year"], var_name="age_band", value_name="pop_str")
        
        # Clean pop_str
        df["population"] = pd.to_numeric(df["pop_str"].astype(str).str.replace(" ", ""), errors='coerce') * 1000
        return df
        
    m = read_melt(male_path)
    f = read_melt(female_path)
    
    df = pd.merge(m, f, on=["Year", "age_band"], suffixes=("_m", "_f"))
    df["total_pop"] = df["population_m"] + df["population_f"]
    
    # Parse age band to numeric start
    df["age_start"] = df["age_band"].str.extract(r'(\d+)').astype(int)
    
    return df

def get_age_weight_band(age_start):
    """
    Crude prevalence weights for Aortic Stenosis potential market, based on age band start.
    """
    if age_start < 65: return 0.01
    if 65 <= age_start < 70: return 0.1
    if 70 <= age_start < 75: return 0.4
    if 75 <= age_start < 80: return 0.8
    if age_start >= 80: return 1.0 # Highest risk
    return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/adoption_model")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Hardcoded paths based on project structure
    # Running from simulation_run_v1/
    tavi_path = Path("data/registry_tavi.csv") 
    savr_path = Path("data/registry_savr.csv")
    pop_un_m_path = Path("data/un_population_data/Population Projections - Male, Korea.csv")
    pop_un_f_path = Path("data/un_population_data/Population Projections - Female, Korea.csv")
    pop_korea_combined_path = Path("data/korea_population_combined.csv")

    if not tavi_path.exists():
        # Fallback if running from proper root
        tavi_path = Path("../data/registry_tavi.csv")
        savr_path = Path("../data/registry_savr.csv")
        pop_un_m_path = Path("../data/un_population_data/Population Projections - Male, Korea.csv")
        pop_un_f_path = Path("../data/un_population_data/Population Projections - Female, Korea.csv")
        pop_korea_combined_path = Path("../data/korea_population_combined.csv")
        
    log.info("Loading Data...")
    tavi_df = pd.read_csv(tavi_path)
    savr_df = pd.read_csv(savr_path)
    # Sum counts by year
    tavi_yearly = tavi_df.groupby("year")["count"].sum().sort_index()
    savr_yearly = savr_df.groupby("year")["count"].sum().sort_index()
    
    # Combined index
    years_hist = sorted(list(set(tavi_yearly.index) | set(savr_yearly.index)))
    # Align
    tavi_hist = tavi_yearly.reindex(years_hist, fill_value=0)
    savr_hist = savr_yearly.reindex(years_hist, fill_value=0)
    total_hist = tavi_hist + savr_hist
    tavi_share_hist = tavi_hist / total_hist
    
    # 2. Population & Market Size
    log.info("Loading & Processing Population...")
    
    # Load Korea combined data (Solid South Korea data 2022+)
    korea_wpop_df = load_population_korea_combined(pop_korea_combined_path)
    korea_wpop_yearly = korea_wpop_df.set_index("year")["weighted_pop"]
    
    # Heuristic Backfill for 2012-2021 (since UN data loading is proving fragile)
    # Assumption: Weighted population (elderly heavy) has been growing.
    # We estimate it was lower in the past.
    # CAGR approx 3-4% for elderly population in Korea.
    # We will backcast from 2022 value.
    
    wpop_dict = korea_wpop_yearly.to_dict()
    min_k_year = min(wpop_dict.keys())
    base_val = wpop_dict[min_k_year]
    
    # Backfill 2012 to min_k_year - 1
    growth_rate = 0.035 # 3.5% annual growth in weighted pop
    
    for y in range(min_k_year - 1, 2011, -1):
        # Value = NextYear / (1 + r)
        next_val = wpop_dict[y+1]
        wpop_dict[y] = next_val / (1 + growth_rate)
        
    wpop_yearly = pd.Series(wpop_dict).sort_index()
    wpop_yearly.name = "weighted_pop"
    wpop_yearly.index.name = "year"
    
    log.info(f"Combined WPop Years: {len(wpop_yearly)}")
    
    # 3. Derive Treatment Rates
    # Common years
    common_years = [y for y in years_hist if y in wpop_yearly.index]
    
    # Filter to analysis range (e.g. start from 2012 where data is robust)
    common_years = [y for y in common_years if y >= 2012]
    
    treatment_rates = []
    tavi_shares = []
    
    for y in common_years:
        vol = total_hist[y]
        wpop = wpop_yearly[y]
        rate = vol / wpop
        treatment_rates.append(rate)
        tavi_shares.append(tavi_share_hist[y])
        
    # 4. Fit Models
    X = np.array(common_years)
    Y_rate = np.array(treatment_rates)
    Y_share = np.array(tavi_shares)
    
    # Fit Treatment Rate (Sigmoid)
    # Assuming treatment rate saturates at some point
    # Current max is roughly Y_rate[-1]. Let's allow it to grow to say 1.5x current or fit it.
    popt_rate = fit_sigmoid_share(X, Y_rate, max_share=max(Y_rate)*2.0)
    
    # Fit TAVI Share (Sigmoid)
    # Max share 0.95 (SAVR never fully disappears)
    popt_share = fit_sigmoid_share(X, Y_share, max_share=0.95)
    
    # 5. Project
    future_years = np.arange(2012, 2051)
    
    def predict_sigmoid(x, p):
        return sigmoid(x, *p)
    
    if popt_rate is not None:
        proj_rate = predict_sigmoid(future_years, popt_rate)
    else:
        # Linear fallback
        z = np.polyfit(X, Y_rate, 1)
        p = np.poly1d(z)
        proj_rate = p(future_years)
        
    if popt_share is not None:
        proj_share = predict_sigmoid(future_years, popt_share)
    else:
        # Linear fallback
        z = np.polyfit(X, Y_share, 1)
        p = np.poly1d(z)
        proj_share = p(future_years)
        proj_share = np.clip(proj_share, 0, 0.95)
    
    # Construct Results
    results = []
    for i, y in enumerate(future_years):
        if y in wpop_yearly.index:
            wpop = wpop_yearly[y]
            t_rate = proj_rate[i]
            t_share = proj_share[i]
            
            est_total = wpop * t_rate
            est_tavi = est_total * t_share
            est_savr = est_total * (1 - t_share)
            
            # Record
            # Overwrite with observed if available?
            # User wants "new index predictions" - implies forecast.
            # But let's show history as history for comparison.
            is_hist = y in common_years
            
            results.append({
                "year": y,
                "wpop": wpop,
                "treatment_rate": t_rate,
                "tavi_share": t_share,
                "pred_total": est_total,
                "pred_tavi": est_tavi,
                "pred_savr": est_savr,
                "obs_tavi": tavi_hist.get(y, np.nan) if y <= max(common_years) else np.nan,
                "obs_savr": savr_hist.get(y, np.nan) if y <= max(common_years) else np.nan
            })
            
    res_df = pd.DataFrame(results)
    
    # Save Outputs
    res_df.to_csv(out_dir / "adoption_model_projections.csv", index=False)
    
    # Plots
    # 1. Fits
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rate
    ax[0].scatter(X, Y_rate, color='black', label='Observed')
    ax[0].plot(future_years, proj_rate, color='blue', label='Sigmoid Fit')
    ax[0].set_title("Population-Adjusted Treatment Rate")
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("Procedures per Weighted Person")
    ax[0].legend()
    
    # Share
    ax[1].scatter(X, Y_share, color='black', label='Observed')
    ax[1].plot(future_years, proj_share, color='red', label='Sigmoid Fit')
    ax[1].set_title("TAVI Market Share")
    ax[1].set_ylim(0, 1.0)
    ax[1].set_xlabel("Year")
    ax[1].set_ylabel("Share of Index Procedures")
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "adoption_fits.png")
    plt.close()
    
    # 2. Volume Projection vs History
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Stacked Area for projections
    ax.stackplot(res_df["year"], res_df["pred_tavi"], res_df["pred_savr"], 
                 labels=["Projected TAVI", "Projected SAVR"], alpha=0.3, colors=["blue", "red"])
    
    # Lines for observed
    ax.plot(res_df["year"], res_df["obs_tavi"], 'o-', color='blue', label='Observed TAVI')
    ax.plot(res_df["year"], res_df["obs_savr"], 's-', color='red', label='Observed SAVR')
    
    ax.set_title("Projected Index Volumes (Impact of Adoption & Demography)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Volume")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "volume_projection_comparison.png")
    plt.close()
    
    log.info(f"Done. Outputs in {out_dir}")

if __name__ == "__main__":
    main()
