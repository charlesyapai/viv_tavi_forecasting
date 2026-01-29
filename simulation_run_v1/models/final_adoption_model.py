import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import argparse
from pathlib import Path
from scipy.optimize import curve_fit
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


def load_raw_wide_data(base_path):
    # Reads the raw wide CSVs (Sex, Age group, 2012...2024) and melts them
    tavi_raw = base_path / "data/raw/registry_tavi_raw.csv"
    savr_raw = base_path / "data/raw/registry_savr_raw.csv"
    
    def process_wide(file_path):
        df = pd.read_csv(file_path)
        # Melt
        # Columns: Sex, Age group, 2012, ...
        # Identify year columns (all digits)
        year_cols = [c for c in df.columns if c.strip().isdigit()]
        
        melted = df.melt(id_vars=["Sex", "Age group"], value_vars=year_cols, 
                         var_name="year", value_name="count")
        
        # Clean count: replace '-', remove ','
        def clean_num(x):
            if isinstance(x, str):
                x = x.replace(",", "").replace("-", "0").strip()
                if x == "": return 0
            if pd.isna(x): return 0
            return float(x)
            
        melted["count"] = melted["count"].apply(clean_num)
        melted["year"] = melted["year"].astype(int)
        
        # Aggregate to [year, age_band]
        melted = melted.rename(columns={"Age group": "age_band"})
        
        # Group by year, age_band (summing over Sex)
        grouped = melted.groupby(["year", "age_band"])["count"].sum().reset_index()
        return grouped

    tavi_df = process_wide(tavi_raw)
    savr_df = process_wide(savr_raw)
    return tavi_df, savr_df


def load_data():
    base = Path(".")
    # Fallback logic for base path
    if not (base / "data/raw/registry_tavi_raw.csv").exists():
        base = Path("/Users/charles/Desktop/viv_tavi_forecasting")
        
    log.info(f"Loading raw wide data from base: {base}")
    tavi_df, savr_df = load_raw_wide_data(base)
    

    # Use absolute path for hospital data to be safe, or relative to base
    hosp_path = Path("/Users/charles/Desktop/viv_tavi_forecasting/data/raw/TAVI_adoption_hospital_numbers_korea")
    
    # Fallback if that fails (e.g. diff machine, but here we are on user machine)
    if not hosp_path.exists():
        hosp_path = base / "data/raw/TAVI_adoption_hospital_numbers_korea"

    hosp_df = pd.read_csv(hosp_path)
    
    un_m_path = Path("/Users/charles/Desktop/viv_tavi_forecasting/simulation_run_v1/data/un_population_data/Population Projections - Male, Korea.csv")
    un_f_path = Path("/Users/charles/Desktop/viv_tavi_forecasting/simulation_run_v1/data/un_population_data/Population Projections - Female, Korea.csv")


    
    # Load UN
    def process_un(p):
        df = pd.read_csv(p)
        df = df[df["Region, subregion, country or area *"] == "Republic of Korea"]
        age_cols = [c for c in df.columns if c[0].isdigit()]
        for c in age_cols:
            if df[c].dtype == object: df[c] = df[c].str.replace(" ", "").astype(float)
        # Unique years
        df = df.drop_duplicates(subset=["Year"]).set_index("Year")[age_cols] * 1000
        return df

    pop = process_un(un_m_path).add(process_un(un_f_path), fill_value=0)
    return tavi_df, savr_df, hosp_df, pop

def align_ages(pop_df):
    # Map UN to Registry Bands: 50-54... >=80
    mapped = pd.DataFrame(index=pop_df.index)
    bands = ['50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '>=80']
    for b in bands:
        if b == '>=80':
            cols = [c for c in pop_df.columns if c.split('-')[0].isdigit() and int(c.split('-')[0]) >= 80]
            if '100+' in pop_df.columns: cols.append('100+')
            mapped[b] = pop_df[cols].sum(axis=1)
        else:
            if b in pop_df.columns: mapped[b] = pop_df[b]
    return mapped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/Users/charles/Desktop/viv_tavi_forecasting/writeups-thoughts/projecting_index_procedures/official_report")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    

def project_risk_logarithmic(years, past_rates):
    # Fit a logarithmic curve to valid historical rates to capture "diminishing returns" growth
    # Rate = a + b * ln(year - offset)
    # This naturally slows down growth, preventing explosion to infinity.
    valid_y, valid_r = [], []
    for y, r in zip(years, past_rates):
        if not np.isnan(r) and r > 0:
            valid_y.append(y)
            valid_r.append(r)
            
    if len(valid_y) < 3: return [np.mean(valid_r)] * len(years) # Fallback
    
    def log_func(t, a, b):
        return a + b * np.log(t - 2010)

    try:
        popt, _ = curve_fit(log_func, valid_y, valid_r, maxfev=10000)
        # Project forward
        future_y = np.arange(2012, 2051)
        future_r = log_func(future_y, *popt)
        return future_r
    except:
        return [np.mean(valid_r)] * 39 # Fallback


def normalize_series_redistribution(years, values, label="Data"):
    # Redistribution Normalization:
    # 1. Fit Trend to Pre-COVID (2012-2019).
    # 2. Redistribute the TOTAL observed volume of 2020-2023 across 2020-2023 
    #    to match the SHAPE of the trend, but keeping the SUM constant.
    
    data_map = dict(zip(years, values))
    
    pre_covid_years = [y for y in years if y <= 2019]
    pre_covid_vals = [data_map[y] for y in pre_covid_years]
    
    covid_window = [2020, 2021, 2022, 2023]
    
    # Only proceed if we have enough pre-covid data AND the covid window exists in data
    has_window = all(y in data_map for y in covid_window)
    
    if len(pre_covid_years) > 3 and has_window:
        # 1. Fit Linear Trend to Pre-COVID
        z = np.polyfit(pre_covid_years, pre_covid_vals, 1)
        trend_func = np.poly1d(z)
        
        # 2. Get Total Volume to Redistribute
        total_obs_vol = sum(data_map[y] for y in covid_window)
        
        # 3. Get Trend Expected Volume
        trend_vals = [trend_func(y) for y in covid_window]
        total_trend_vol = sum(trend_vals)
        
        # 4. Calculate Scaling Factor (to preserve total volume)
        # If trend predicts 1000 but we have 800, we scale trend down by 0.8
        if total_trend_vol > 0:
            scale = total_obs_vol / total_trend_vol
        else:
            scale = 1.0
            
        norm_map = data_map.copy()
        
        log.info(f"Redistribution {label}: ObsSum={total_obs_vol:.1f}, TrendSum={total_trend_vol:.1f}, Scale={scale:.3f}")
        
        # Apply Redistribution
        for i, y in enumerate(covid_window):
            norm_map[y] = trend_vals[i] * scale
            
        return [norm_map[y] for y in years], norm_map
        
    return values, data_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/adoption_model")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    

def plot_normalization_effect(years, raw, norm, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, raw, 'k--', label="Raw Data (Dip & Spike)", alpha=0.5)
    ax.scatter(years, raw, color='black', alpha=0.3)
    
    ax.plot(years, norm, 'b-', label="Normalized Trend (Redistributed)", linewidth=2)
    ax.scatter(years, norm, color='blue')
    
    # Highlight 2020-2023
    mask = [2020 <= y <= 2023 for y in years]
    y_range = [y for y, m in zip(years, mask) if m]
    if y_range:
        ax.axvspan(2019.5, 2023.5, color='yellow', alpha=0.1, label="COVID/Recovery Window")
        
    ax.set_title("Step 2: Normalization & Redistribution")
    ax.set_ylabel("Procedure Volume")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(out_dir / "step2_normalization_effect.png")
    plt.close()




def main():
    parser = argparse.ArgumentParser()
    # UPDATE: New default output directory
    parser.add_argument("--out-dir", default="/Users/charles/Desktop/viv_tavi_forecasting/writeups-thoughts/projecting_index_procedures/official_report")
    parser.add_argument("--center-cap", type=int, default=94, help="Max number of centers")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tavi_df, savr_df, hosp_df, pop_raw = load_data()
    pop_mapped = align_ages(pop_raw)
    
    years = sorted(list(set(tavi_df['year']) | set(savr_df['year'])))
    years = [y for y in years if y <= 2024] 
    

    # DEBUG: Print Raw Sums to check 2015 and 2024
    log.info("--- Raw Data Check ---")
    tavi_sums = tavi_df.groupby("year")["count"].sum()
    savr_sums = savr_df.groupby("year")["count"].sum()
    combined_sums = tavi_sums.add(savr_sums, fill_value=0)
    for y in years:
        log.info(f"Year {y}: TAVI={tavi_sums.get(y,0)}, SAVR={savr_sums.get(y,0)}, Total={combined_sums.get(y,0)}")
    
    # ==========================================
    # === STEP 1 & 2: RAW DATA & NORMALIZATION ===
    # ==========================================
    # User Request: "This plot should be for COMBINED SAVR AND TAVR"
    X_comb = sorted([y for y in combined_sums.index if y <= 2024])
    Y_comb_raw = [combined_sums.get(y, 0) for y in X_comb]
    
    # ==========================================
    # === STEP 1: DATA OVERVIEW ===
    # ==========================================
    
    # 1. Bar Chart Data (TAVI/SAVR)
    # Ensure we cover up to 2024
    X_bar = sorted([y for y in combined_sums.index if 2012 <= y <= 2024])
    Y_tavi_bar = [tavi_sums.get(y, 0) for y in X_bar]
    Y_savr_bar = [savr_sums.get(y, 0) for y in X_bar]
    
    # 2. Population Data (50-79 vs 80+)
    # We need to sum the columns in pop_mapped
    # Bands: 50-54, 55-59, 60-64, 65-69, 70-74, 75-79 -> "50-79"
    # Bands: >=80 -> "80+"
    
    bands_mid = ['50-54', '55-59', '60-64', '65-69', '70-74', '75-79']
    bands_high = ['>=80']
    
    pop_years = sorted([y for y in pop_mapped.index if 2012 <= y <= 2050])
    pop_mid_series = []
    pop_high_series = []
    
    for y in pop_years:
        row = pop_mapped.loc[y]
        sum_mid = sum(row.get(b, 0) for b in bands_mid)
        sum_high = sum(row.get(b, 0) for b in bands_high)
        pop_mid_series.append(sum_mid)
        pop_high_series.append(sum_high)
        
    # Debug Logging for Population
    log.info(f"Pop 80+ (High) in {pop_years[0]}: {pop_high_series[0]:,.0f}")
    log.info(f"Pop 80+ (High) in {pop_years[-1]}: {pop_high_series[-1]:,.0f}")
    log.info(f"Pop 50-79 (Mid) in {pop_years[0]}: {pop_mid_series[0]:,.0f}")
    log.info(f"Pop 50-79 (Mid) in {pop_years[-1]}: {pop_mid_series[-1]:,.0f}")

    # Plotting
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Stacked Bar
    ax1.bar(X_bar, Y_tavi_bar, label='TAVI', alpha=0.7)
    ax1.bar(X_bar, Y_savr_bar, bottom=Y_tavi_bar, label='SAVR', alpha=0.7)
    ax1.set_title("Input 1: Index Procedure Volumes (Korea)", fontsize=12)
    ax1.set_ylabel("Annual Procedures")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Population Trends
    ax2.plot(pop_years, pop_high_series, 'o-', ms=4, label='Population 80+', linewidth=2, color='tab:blue')
    ax2.plot(pop_years, pop_mid_series, 's-', ms=4, label='Population 50-79', linewidth=2, color='tab:orange')
    ax2.set_title("Input 2: Key Age Cohorts (UN Projection)", fontsize=12)
    ax2.set_ylabel("Population Count (Millions)")
    
    # Fix Y-axis to start at 0
    ax2.set_ylim(bottom=0)
    
    # Format Y-axis in Millions (e.g. 1.5M)
    def millions(x, pos):
        return f'{x*1e-6:.1f}M'
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(millions))
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "step1_data_overview.png", dpi=150)
    plt.close()

    # Normalize the COMBINED Series
    Y_comb_norm, _ = normalize_series_redistribution(X_comb, Y_comb_raw, label="Combined Total")
    
    # Refined Step 2 Plot
    fig_norm, ax_norm = plt.subplots(figsize=(10, 6))
    ax_norm.plot(X_comb, Y_comb_raw, 'k--', label="Raw Observed Data (2012-2024)", alpha=0.6)
    ax_norm.scatter(X_comb, Y_comb_raw, color='black', alpha=0.4)
    ax_norm.plot(X_comb, Y_comb_norm, 'b-', label="Normalized Trend (Volume Preserved)", linewidth=2.5)
    ax_norm.scatter(X_comb, Y_comb_norm, color='blue', zorder=5)
    
    # Highlight 2020-2023
    ax_norm.axvspan(2019.5, 2023.5, color='orange', alpha=0.1, label="Pandemic Redistribution Window")
        
    ax_norm.set_title("Step 2: Normalization (Volume-Preserved Redistribution)", fontsize=14)
    ax_norm.set_ylabel("Total Procedures (TAVI + SAVR)", fontsize=12)
    ax_norm.legend(loc='upper left')
    ax_norm.grid(True, alpha=0.3)
    plt.savefig(out_dir / "step2_normalization_effect.png", dpi=150)
    plt.close()
    
    # ==========================================
    # === STEP 3: RISK TRENDS ===
    # ==========================================
    bands = ['50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '>=80']
    risk_rates_raw = {y: {} for y in years}
    for y in years:
        p_sub = pop_mapped.loc[y]
        for b in bands:
            t = tavi_df[(tavi_df['year']==y) & (tavi_df['age_band']==b)]['count'].sum()
            s = savr_df[(savr_df['year']==y) & (savr_df['age_band']==b)]['count'].sum()
            pop = p_sub.get(b, 1)
            risk_rates_raw[y][b] = (t+s)/pop 

    risk_rates_norm = {y: {} for y in years}
    future_risks_dict = {} 
    proj_years_full = np.arange(2012, 2051)
    
    fig_risk, ax_risk = plt.subplots(figsize=(10, 6))
    for b in bands:
        y_series = years
        val_series = [risk_rates_raw[y][b] for y in years]
        val_norm, _ = normalize_series_redistribution(y_series, val_series, label=f"Risk {b}")
        for i, y in enumerate(years): risk_rates_norm[y][b] = val_norm[i]
        
        future_r_curve = project_risk_logarithmic(y_series, val_norm)
        future_risks_dict[b] = future_r_curve
        
        p = ax_risk.plot(proj_years_full, future_r_curve * 100000, label=f"Age {b}")
        c = p[0].get_color()
        ax_risk.scatter(years, [v*100000 for v in val_norm], color=c, marker='o', s=15, alpha=0.4)

    ax_risk.set_title("Step 3: Age-Stratified Risk Projections (Logarithmic Model)", fontsize=14)
    ax_risk.set_ylabel("Procedures per 100,000 Population", fontsize=12)
    ax_risk.legend(fontsize='small', loc='upper left')
    ax_risk.grid(True, alpha=0.3)
    plt.savefig(out_dir / "step3_projected_risk_trends.png", dpi=150)
    
    # ==========================================
    # === STEP 4: TOTAL ADDRESSABLE MARKET (TAM) ===
    # ==========================================
    tam_series = []
    tam_breakdown = []
    for i, y in enumerate(proj_years_full):
        if y in pop_mapped.index:
            tam_total = 0
            row_det = {"year": y}
            for b in bands:
                risk_val = future_risks_dict[b][i] 
                pop_val = pop_mapped.loc[y].get(b, 0)
                seg_tam = pop_val * risk_val
                tam_total += seg_tam
                if y in [2023, 2024, 2030, 2040, 2050]:
                    row_det[f"Pop_{b}"] = pop_val
                    row_det[f"Risk_{b}"] = risk_val
                    row_det[f"TAM_{b}"] = seg_tam
            tam_series.append(tam_total)
            if "Pop_>=80" in row_det: tam_breakdown.append(row_det)
        else:
            tam_series.append(tam_series[-1])
    
    pd.DataFrame(tam_breakdown).to_csv(out_dir / "tam_granular_breakdown.csv", index=False)
            
    fig_tam, ax_tam = plt.subplots(figsize=(10, 6))
    ax_tam.plot(proj_years_full, tam_series, 'k-', lw=3, label="Total Addressable Market (TAM)")
    ax_pop = ax_tam.twinx()
    pop_80 = pop_mapped[">=80"].loc[2012:2050]
    ax_pop.plot(pop_80.index, pop_80.values, 'g--', linewidth=2, label="Population (80+ Cohort)")
    ax_pop.set_ylabel("Population Count (80+)", color='green', fontsize=12)
    
    ax_tam.set_title("Step 4: TAM vs Demographic Driver", fontsize=14)
    ax_tam.set_ylabel("Total Procedures (TAM)", fontsize=12)
    
    # Combine legends
    lines_1, labels_1 = ax_tam.get_legend_handles_labels()
    lines_2, labels_2 = ax_pop.get_legend_handles_labels()
    ax_tam.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    ax_tam.grid(True, alpha=0.3)
    plt.savefig(out_dir / "step4_projected_tam.png", dpi=150)
    
    # ==========================================
    # === STEP 5: TAVI ADOPTION (VARIANT: DEMOGRAPHICS + SIGMOID) ===
    # ==========================================
    # 1. Prepare TAVI Data & Apply Redistribution (Step 1 of Variant)
    # ---------------------------------------------------------
    tavi_tot = tavi_df.groupby("year")["count"].sum()
    X_tavi = sorted([y for y in tavi_tot.index if y <= 2024])
    Y_tavi_raw_vals = [tavi_tot[y] for y in X_tavi]
    
    Y_tavi_norm, norm_map = normalize_series_redistribution(X_tavi, Y_tavi_raw_vals, label="TAVI History")
    
    # Plot Variant Step 1: Redistribution
    fig_v1, ax_v1 = plt.subplots(figsize=(10, 6))
    ax_v1.plot(X_tavi, Y_tavi_raw_vals, 'k--', label="Raw TAVI", alpha=0.5)
    ax_v1.scatter(X_tavi, Y_tavi_raw_vals, color='black', alpha=0.3)
    ax_v1.plot(X_tavi, Y_tavi_norm, 'b-', label="Redistributed TAVI (2020-2023)", lw=2)
    ax_v1.axvspan(2019.5, 2023.5, color='orange', alpha=0.1, label="Redistribution Window")
    ax_v1.set_title("Step 5.1: Volume-Preserved Redistribution (TAVI Only)")
    ax_v1.legend()
    ax_v1.grid(True, alpha=0.3)
    plt.savefig(out_dir / "step5_tavi_redistribution.png", dpi=150)
    plt.close()
    
    # 2. Demographic Normalization (Step 2 of Variant)
    # ---------------------------------------------------------
    # Goal: Calculate "Adjusted Volume" as if population structure was fixed at 2024
    scale_factors = {}
    for y, count in zip(X_tavi, Y_tavi_raw_vals):
        norm_val = norm_map.get(y, count)
        scale_factors[y] = norm_val / count if count > 0 else 1.0
        
    ref_pop_year = 2024
    ref_pop = pop_mapped.loc[ref_pop_year]
    
    adj_vol_series = []
    
    for y in X_tavi:
        # Get Raw Age Counts
        y_rows = tavi_df[tavi_df['year'] == y]
        row_pop = pop_mapped.loc[y]
        
        adj_sum = 0
        for b in bands:
            raw_c = y_rows[y_rows['age_band'] == b]['count'].sum()
            # Apply redistribution scale
            norm_c = raw_c * scale_factors.get(y, 1.0)
            pop_c = row_pop.get(b, 1) # Avoid div by zero
            
            rate = norm_c / pop_c
            # Apply to Ref Population
            adj_sum += rate * ref_pop.get(b, 0)
            
        adj_vol_series.append(adj_sum)
        
    # Plot Variant Step 2: Demographic Adjustment
    fig_v2, ax_v2 = plt.subplots(figsize=(10, 6))
    ax_v2.plot(X_tavi, Y_tavi_norm, 'b--', label="Redistributed Volume (Actual Pop)", alpha=0.6)
    ax_v2.plot(X_tavi, adj_vol_series, 'g-', label="Demographically Adjusted (Fixed 2024 Pop)", lw=2)
    ax_v2.set_title("Step 5.2: Correction for Population Growth/Aging")
    ax_v2.legend()
    ax_v2.grid(True, alpha=0.3)
    plt.savefig(out_dir / "step5_demographic_adjustment.png", dpi=150)
    plt.close()

    # 3. Sigmoid Projection (Step 3 of Variant) - Midpoint 2023
    # ---------------------------------------------------------
    # Fit Sigmoid to AdjVol, x0 fixed at 2023
    def sigmoid_constrained(x, L, k):
        return L / (1 + np.exp(-k * (x - 2023)))
    
    max_vol = max(adj_vol_series)
    try:
        popt_s, _ = curve_fit(sigmoid_constrained, X_tavi, adj_vol_series, 
                            p0=[max_vol*2, 0.2], 
                            bounds=([max_vol, 0.01], [max_vol*10, 2.0]),
                            maxfev=5000)
    except Exception as e:
        log.error(f"Curve fit failed: {e}")
        popt_s = [max_vol * 1.5, 0.1]
        
    L_fit, k_fit = popt_s
    log.info(f"Sigmoid Fit: L={L_fit:.1f}, k={k_fit:.3f}, x0=2023 (Fixed)")
    
    pred_adj_vol = sigmoid_constrained(proj_years_full, L_fit, k_fit)
    
    # Plot Variant Step 3: Sigmoid Fit
    fig_v3, ax_v3 = plt.subplots(figsize=(10, 6))
    ax_v3.plot(X_tavi, adj_vol_series, 'go', label="Historical Adjusted Vol")
    ax_v3.plot(proj_years_full, pred_adj_vol, 'g--', label=f"Sigmoid Projection (Mid=2023, k={k_fit:.2f})")
    ax_v3.axvline(2023, color='orange', linestyle=':', label="Midpoint (2023)")
    ax_v3.set_title("Step 5.3: Sigmoid Projection of Demographically Adjusted Volume")
    ax_v3.legend()
    ax_v3.grid(True, alpha=0.3)
    plt.savefig(out_dir / "step5_tavi_sigmoid_fit.png", dpi=150)
    plt.close()

    # 4. Re-apply Demographic Growth (Forecast)
    # ---------------------------------------------------------
    # Calculate 2024 Age-Specific Rates 
    rates_2024 = {}
    if 2024 in X_tavi:
        y24_rows = tavi_df[tavi_df['year'] == 2024]
        row24_pop = pop_mapped.loc[2024]
        last_scale = scale_factors.get(2024, 1.0)
    else:
        y24_rows = tavi_df[tavi_df['year'] == X_tavi[-1]]
        row24_pop = pop_mapped.loc[X_tavi[-1]]
        last_scale = scale_factors.get(X_tavi[-1], 1.0)
        
    for b in bands:
        c = y24_rows[y24_rows['age_band']==b]['count'].sum()
        c_norm = c * last_scale
        p = row24_pop.get(b, 1)
        rates_2024[b] = c_norm / p

    # Calculate Demographic Multiplier
    demo_mult = []
    base_idx_val = sum(rates_2024[b] * ref_pop.get(b, 0) for b in bands)
    
    for y in proj_years_full:
        if y in pop_mapped.index:
            p_row = pop_mapped.loc[y]
            idx_val = sum(rates_2024[b] * p_row.get(b, 0) for b in bands)
            mult = idx_val / base_idx_val if base_idx_val > 0 else 1.0
            demo_mult.append(mult)
        else:
            demo_mult.append(demo_mult[-1])
            
    # FINAL TAVI FORECAST
    pred_tavi = pred_adj_vol * np.array(demo_mult)
    
    # NEW COMPARISON PLOT: Fixed vs With-Demo + Pop Distribution 
    # ---------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(12, 7))
    
    # 1. Curves
    ax5.plot(proj_years_full, pred_adj_vol, 'g--', linewidth=2, label="Sigmoid Only (Fixed 2024 Pop)")
    ax5.plot(proj_years_full, pred_tavi, 'b-', linewidth=3, label="Final Forecast (Sigmoid + Demographic Shift)")
    
    ax5.set_ylabel("TAVI Volume", fontsize=12)
    ax5.set_title("Step 5.4: Impact of Demographics on TAVI Adoption", fontsize=14)
    ax5.grid(True, alpha=0.3)
    
    # 2. Population on Twin Axis
    ax5_pop = ax5.twinx()
    
    # Calculate Pop Groups (Split at 80)
    pop_under_80 = []
    pop_over_80 = []
    
    bands_u80 = ['50-54', '55-59', '60-64', '65-69', '70-74', '75-79']
    bands_o80 = ['>=80']
    
    for y in proj_years_full:
        if y in pop_mapped.index:
            row = pop_mapped.loc[y]
            u80 = sum(row.get(b, 0) for b in bands_u80)
            o80 = sum(row.get(b, 0) for b in bands_o80)
            pop_under_80.append(u80)
            pop_over_80.append(o80)
        else:
            pop_under_80.append(pop_under_80[-1] if pop_under_80 else 0)
            pop_over_80.append(pop_over_80[-1] if pop_over_80 else 0)
            
    # Plot Pop
    ax5_pop.plot(proj_years_full, pop_under_80, color='orange', linestyle=':', alpha=0.7, label="Pop < 80 (50-79)")
    ax5_pop.plot(proj_years_full, pop_over_80, color='purple', linestyle=':', alpha=0.7, label="Pop >= 80")
    
    ax5_pop.set_ylabel("Population Count", fontsize=12)
    
    # Format Y-axis in Millions
    def millions_fmt(x, pos): return f'{x*1e-6:.1f}M'
    ax5_pop.yaxis.set_major_formatter(mtick.FuncFormatter(millions_fmt))
    
    # Combine Legends
    lines, labels = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_pop.get_legend_handles_labels()
    ax5.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.savefig(out_dir / "step5_demographic_impact.png", dpi=150)
    plt.close()
    
    # ==========================================
    # === STEP 6: FINAL RESIDUAL CALCULATION ===
    # ==========================================
    final_data = []
    for i, y in enumerate(proj_years_full):
        final_data.append({
            "year": y, "TAM": tam_series[i], "pred_TAVI": pred_tavi[i], 
            "pred_SAVR": max(0, tam_series[i] - pred_tavi[i])
        })
    res_df = pd.DataFrame(final_data)
    res_df.to_csv(out_dir / "final_calculation_report.csv", index=False)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(res_df['year'], res_df['TAM'], 'k--', linewidth=2, label='Total Addressable Market (TAM)')
    ax.plot(res_df['year'], res_df['pred_TAVI'], 'b-', linewidth=3, label='TAVI Forecast')
    ax.plot(res_df['year'], res_df['pred_SAVR'], 'r-', linewidth=3, label='SAVR Forecast (Residual)')
    ax.scatter(X_tavi, Y_tavi_norm, color='blue', alpha=0.3, label="Historical TAVI (Normalized)")
    
    ax.set_title("Step 6: Final Market Projection (TAVI vs SAVR)", fontsize=16)
    ax.set_ylabel("Annual Procedures", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.savefig(out_dir / "step6_final_projection.png", dpi=150)
    
    log.info(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
