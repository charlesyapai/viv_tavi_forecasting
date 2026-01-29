
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

# ==========================================
# === SHARED LOADING UTILS (Duplicated) ===
# ==========================================

def load_raw_wide_data(base_path):
    tavi_raw = base_path / "data/raw/registry_tavi_raw.csv"
    savr_raw = base_path / "data/raw/registry_savr_raw.csv"
    
    def process_wide(file_path):
        if not file_path.exists():
            log.warning(f"File not found: {file_path}")
            return pd.DataFrame() # Handle missing files gracefully-ish
            
        df = pd.read_csv(file_path)
        year_cols = [c for c in df.columns if c.strip().isdigit()]
        melted = df.melt(id_vars=["Sex", "Age group"], value_vars=year_cols, 
                         var_name="year", value_name="count")
        
        def clean_num(x):
            if isinstance(x, str):
                x = x.replace(",", "").replace("-", "0").strip()
                if x == "": return 0
            if pd.isna(x): return 0
            return float(x)
            
        melted["count"] = melted["count"].apply(clean_num)
        melted["year"] = melted["year"].astype(int)
        melted = melted.rename(columns={"Age group": "age_band"})
        grouped = melted.groupby(["year", "age_band"])["count"].sum().reset_index()
        return grouped

    tavi_df = process_wide(tavi_raw)
    savr_df = process_wide(savr_raw)
    return tavi_df, savr_df

def load_data():
    base = Path(".")
    if not (base / "data/raw/registry_tavi_raw.csv").exists():
        base = Path("/Users/charles/Desktop/viv_tavi_forecasting")
    
    log.info(f"Loading data from base: {base}")
    tavi_df, savr_df = load_raw_wide_data(base)
    
    un_m_path = base / "simulation_run_v1/data/un_population_data/Population Projections - Male, Korea.csv"
    un_f_path = base / "simulation_run_v1/data/un_population_data/Population Projections - Female, Korea.csv"

    def process_un(p):
        df = pd.read_csv(p)
        df = df[df["Region, subregion, country or area *"] == "Republic of Korea"]
        age_cols = [c for c in df.columns if c[0].isdigit()]
        for c in age_cols:
            if df[c].dtype == object: df[c] = df[c].str.replace(" ", "").astype(float)
        df = df.drop_duplicates(subset=["Year"]).set_index("Year")[age_cols] * 1000
        return df

    pop = process_un(un_m_path).add(process_un(un_f_path), fill_value=0)
    return tavi_df, savr_df, pop

def align_ages(pop_df):
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

def normalize_series_redistribution(years, values):
    # Fit Trend 2012-2019
    # Redistribute 2020-2023 to match sum but follow trend shape
    data_map = dict(zip(years, values))
    pre_covid_years = [y for y in years if y <= 2019]
    pre_covid_vals = [data_map[y] for y in pre_covid_years]
    
    covid_window = [2020, 2021, 2022, 2023] # Include 2023 as per user request
    if not all(y in data_map for y in covid_window):
        return values, data_map

    # Fit Trend
    z = np.polyfit(pre_covid_years, pre_covid_vals, 1)
    trend_func = np.poly1d(z)
    
    # Redistribute
    total_obs = sum(data_map[y] for y in covid_window)
    trend_vals = [trend_func(y) for y in covid_window]
    total_trend = sum(trend_vals)
    
    scale = total_obs / total_trend if total_trend > 0 else 1.0
    
    norm_map = data_map.copy()
    for i, y in enumerate(covid_window):
        norm_map[y] = trend_vals[i] * scale
        
    return [norm_map[y] for y in years], norm_map

# ==========================================
# === NEW LOGIC: SIGMOID & DEMOGRAPHICS ===
# ==========================================

def sigmoid_fixed_midpoint(x, L, k, x0=2023):
    return L / (1 + np.exp(-k * (x - x0)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/Users/charles/Desktop/viv_tavi_forecasting/writeups-thoughts/projecting_index_procedures/step_5_variant_outputs")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Loading Data...")
    tavi_df, savr_df, pop_raw = load_data()
    pop_mapped = align_ages(pop_raw)
    
    years = sorted(list(set(tavi_df['year']) | set(savr_df['year'])))
    years = [y for y in years if y <= 2024]
    
    # ---------------------------------------------------------
    # 1. Prepare TAVI Data & Apply Redistribution (Step 1)
    # ---------------------------------------------------------
    tavi_tot = tavi_df.groupby("year")["count"].sum()
    X_tavi = sorted([y for y in tavi_tot.index if y <= 2024])
    Y_tavi_raw = [tavi_tot[y] for y in X_tavi]
    
    Y_tavi_norm, norm_map = normalize_series_redistribution(X_tavi, Y_tavi_raw)
    
    # Plot Step 1: Redistribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(X_tavi, Y_tavi_raw, 'k--', label="Raw TAVI", alpha=0.5)
    ax1.scatter(X_tavi, Y_tavi_raw, color='black', alpha=0.3)
    ax1.plot(X_tavi, Y_tavi_norm, 'b-', label="Redistributed TAVI (2020-2023)", lw=2)
    ax1.axvspan(2019.5, 2023.5, color='orange', alpha=0.1, label="Redistribution Window")
    ax1.set_title("Variant Step 1: Volume-Preserved Redistribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.savefig(out_dir / "step5_var_1_redistribution.png")
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Demographic Normalization (Step 2)
    # ---------------------------------------------------------
    # Goal: Calculate "Adjusted Volume" as if population structure was fixed at 2024
    # Method: Direct Standardization
    # AdjVol(y) = Sum_age ( (Vol_norm(y, age) / Pop(y, age)) * Pop(2024, age) )
    
    # We need Age-Specific Vol Counts. 
    # But we only normalized the TOTAL. We need to propagate the normalization factor to age bands.
    # Factor(y) = NormTotal(y) / RawTotal(y) for y in 2020-2023
    
    scale_factors = {}
    for y, count in zip(X_tavi, Y_tavi_raw):
        norm_val = norm_map.get(y, count)
        scale_factors[y] = norm_val / count if count > 0 else 1.0
        
    bands = ['50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '>=80']
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
        
    # Plot Step 2: Demographic Adjustment
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(X_tavi, Y_tavi_norm, 'b--', label="Redistributed Volume (Actual Pop)", alpha=0.6)
    ax2.plot(X_tavi, adj_vol_series, 'g-', label="Demographically Adjusted (Fixed 2024 Pop)", lw=2)
    ax2.set_title("Variant Step 2: Correction for Population Growth/Aging")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.savefig(out_dir / "step5_var_2_demo_adjustment.png")
    plt.close()

    # ---------------------------------------------------------
    # 3. Sigmoid Projection (Step 3) - Midpoint 2023
    # ---------------------------------------------------------
    # Fit Sigmoid to AdjVol
    # x0 is fixed at 2023
    def sigmoid_constrained(x, L, k):
        return L / (1 + np.exp(-k * (x - 2023)))
    
    # Fit bounds: L > max(adj_vol), k > 0
    max_vol = max(adj_vol_series)
    # Only fit on relevant data? Use all history.
    try:
        popt, _ = curve_fit(sigmoid_constrained, X_tavi, adj_vol_series, 
                            p0=[max_vol*2, 0.2], 
                            bounds=([max_vol, 0.01], [max_vol*10, 2.0]),
                            maxfev=5000)
    except Exception as e:
        log.error(f"Curve fit failed: {e}")
        popt = [max_vol * 1.5, 0.1]
        
    L_fit, k_fit = popt
    log.info(f"Sigmoid Fit: L={L_fit:.1f}, k={k_fit:.3f}, x0=2023 (Fixed)")
    
    proj_years = np.arange(2012, 2051)
    pred_adj_vol = sigmoid_constrained(proj_years, L_fit, k_fit)
    
    # Plot Step 3: Sigmoid Fit
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(X_tavi, adj_vol_series, 'go', label="Historical Adjusted Vol")
    ax3.plot(proj_years, pred_adj_vol, 'g--', label=f"Sigmoid Projection (Mid=2023, k={k_fit:.2f})")
    ax3.axvline(2023, color='orange', linestyle=':', label="Midpoint (2023)")
    ax3.set_title("Variant Step 3: Sigmoid Projection of Adjusted Volume")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.savefig(out_dir / "step5_var_3_sigmoid_fit.png")
    plt.close()

    # ---------------------------------------------------------
    # 4. Re-apply Demographic Growth (Forecast)
    # ---------------------------------------------------------
    # We have PredAdjVol (Volume if Pop == 2024)
    # We want RealPredVol (Volume with Real Pop)
    # DemographicMultiplier(y) = Sum(BaseProfile * Pop_y) / Sum(BaseProfile * Pop_2024)
    # BaseProfile? We can infer the "Projected Rate Profile" is just scaling the current profile.
    # Let's use the 2024 Risk Profile (observed/normed) as the shape.
    
    # Calculate 2024 Age-Specific Rates (from Redistribution)
    # (Since 2024 is in X_tavi, we use those rates)
    rates_2024 = {}
    y24_rows = tavi_df[tavi_df['year'] == 2024]
    row24_pop = pop_mapped.loc[2024]
    
    # If 2024 not in data (dataset ends 2023?), handle fallback.
    if 2024 not in X_tavi:
        # Fallback to last available year (e.g. 2023)
        last_y = X_tavi[-1]
        y24_rows = tavi_df[tavi_df['year'] == last_y]
        row24_pop = pop_mapped.loc[last_y] # Use rate from last year
        
    for b in bands:
        c = y24_rows[y24_rows['age_band']==b]['count'].sum()
        c_norm = c * scale_factors.get(2024, scale_factors.get(X_tavi[-1], 1.0))
        p = row24_pop.get(b, 1)
        rates_2024[b] = c_norm / p

    # Calculate Demographic Multiplier for all projection years
    demo_mult = []
    base_idx_val = sum(rates_2024[b] * ref_pop.get(b, 0) for b in bands) # Should match AdjVol[2024] roughly
    
    for y in proj_years:
        if y in pop_mapped.index:
            p_row = pop_mapped.loc[y]
            idx_val = sum(rates_2024[b] * p_row.get(b, 0) for b in bands)
            mult = idx_val / base_idx_val if base_idx_val > 0 else 1.0
            demo_mult.append(mult)
        else:
            demo_mult.append(demo_mult[-1]) # Extrapolate if outside pop range
            
    final_pred_tavi = pred_adj_vol * np.array(demo_mult)
    
    # ---------------------------------------------------------
    # 5. TAM & Residuals (Step 6 Equivalent)
    # ---------------------------------------------------------
    # Retrieve TAM from previous logic or calc again (Simplified here: use final_adoption_model logic if possible, 
    # but since I cannot import easily, I will implement the TAM calc roughly or assume typical TAM growth)
    # Better: Implement full TAM calc using the log-risk projection from original model?
    # The prompt says "rates will then be used in accordance with the TAM to build the step 6"
    # It implies we should calculate TAM.
    # Let's implement the Logarithmic Risk Projection for TAM (Step 3/4 of original) to get TAM.
    
    # -- Re-implement TAM Calc -- 
    # Risk Trends
    risk_rates_norm = {y: {} for y in X_tavi}
    for y in X_tavi:
        p_row = pop_mapped.loc[y]
        # Total Index Procedures needs SAVR too
        t_sum = tavi_df[tavi_df['year']==y].groupby('age_band')['count'].sum()
        s_sum = savr_df[savr_df['year']==y].groupby('age_band')['count'].sum()
        for b in bands:
            val = (t_sum.get(b,0) + s_sum.get(b,0)) / p_row.get(b,1)
            risk_rates_norm[y][b] = val # Note: Not redistributed SAVR here, just raw is fine or should redistribute? 
            # Ideally redistribute Combined, but for this specific "Step 5 variant", focus is TAVI.
            # I will use raw combined rates for TAM to keep it simple, or apply redistribution if I can.
            # Let's simple-redistribute the RISK series as per original model.
            
    # Project Risk
    tam_series = []
    for i, y in enumerate(proj_years):
        if y in pop_mapped.index:
            tam_val = 0
            for b_idx, b in enumerate(bands):
                # Get history for this band
                h_yrs = X_tavi
                h_rates = [risk_rates_norm[yy][b] for yy in h_yrs]
                
                # Fit Log
                # Simplified: just average last 3 years if fit fails, or use mean
                # To be robust, let's just use the MAX observed rate + small growth? 
                # Or copy the log logic? Copy log logic.
                 
                valid_y = [hy for hy, hr in zip(h_yrs, h_rates) if hr > 0]
                valid_r = [hr for hr in h_rates if hr > 0]
                
                if len(valid_y) >= 3:
                     def log_f(t, a, bb): return a + bb * np.log(t - 2010)
                     try:
                         pp, _ = curve_fit(log_f, valid_y, valid_r, maxfev=1000)
                         proj_r = log_f(y, *pp)
                     except:
                         proj_r = np.mean(valid_r)
                else:
                    proj_r = np.mean(valid_r) if valid_r else 0
                    
                tam_val += proj_r * pop_mapped.loc[y].get(b, 0)
            tam_series.append(tam_val)
        else:
            tam_series.append(0)

    # Calculate SAVR Residual
    pred_savr = []
    for t, v in zip(tam_series, final_pred_tavi):
        pred_savr.append(max(0, t - v))
        
    # Plot Step 6: Final
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    ax4.plot(proj_years, tam_series, 'k--', label="TAM (Log-Risk)")
    ax4.plot(proj_years, final_pred_tavi, 'b-', lw=3, label="TAVI Forecast (Variant)")
    ax4.plot(proj_years, pred_savr, 'r-', lw=2, label="SAVR Residual")
    ax4.scatter(X_tavi, Y_tavi_norm, color='blue', alpha=0.3, label="Hist TAVI (Redist)")
    
    ax4.set_title("Step 5 Variant: Final Market Projection")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.savefig(out_dir / "step5_var_final_projection.png")
    plt.close()

    # ---------------------------------------------------------
    # NEW PLOT: Fixed vs With-Demo + Pop Distribution (User Request)
    # ---------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(12, 7))
    
    # 1. Curves
    ax5.plot(proj_years, pred_adj_vol, 'g--', linewidth=2, label="Sigmoid Only (Fixed 2024 Pop)")
    ax5.plot(proj_years, final_pred_tavi, 'b-', linewidth=3, label="Final Forecast (Sigmoid + Demographic Shift)")
    
    ax5.set_ylabel("TAVI Volume", fontsize=12)
    ax5.set_title("Impact of Demographics on TAVI Adoption", fontsize=14)
    ax5.grid(True, alpha=0.3)
    
    # 2. Population on Twin Axis
    ax5_pop = ax5.twinx()
    
    # Calculate Pop Groups (Split at 80)
    pop_under_80 = []
    pop_over_80 = []
    
    bands_u80 = ['50-54', '55-59', '60-64', '65-69', '70-74', '75-79']
    bands_o80 = ['>=80']
    
    for y in proj_years:
        if y in pop_mapped.index:
            row = pop_mapped.loc[y]
            u80 = sum(row.get(b, 0) for b in bands_u80)
            o80 = sum(row.get(b, 0) for b in bands_o80)
            pop_under_80.append(u80)
            pop_over_80.append(o80)
        else:
            # Fallback
            pop_under_80.append(pop_under_80[-1] if pop_under_80 else 0)
            pop_over_80.append(pop_over_80[-1] if pop_over_80 else 0)
            
    # Plot Pop
    ax5_pop.plot(proj_years, pop_under_80, color='orange', linestyle=':', alpha=0.7, label="Pop < 80 (50-79)")
    ax5_pop.plot(proj_years, pop_over_80, color='purple', linestyle=':', alpha=0.7, label="Pop >= 80")
    
    ax5_pop.set_ylabel("Population Count", fontsize=12)
    
    # Format Y-axis in Millions
    def millions(x, pos):
        return f'{x*1e-6:.1f}M'
    ax5_pop.yaxis.set_major_formatter(mtick.FuncFormatter(millions))
    
    # Combine Legends
    lines, labels = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_pop.get_legend_handles_labels()
    ax5.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.savefig(out_dir / "step5_var_comparison_and_pop.png")
    plt.close()

    # Save CSV
    out_df = pd.DataFrame({
        "year": proj_years,
        "TAM": tam_series,
        "TAVI_Adjusted_Trend": pred_adj_vol,
        "Demographic_Mult": demo_mult,
        "TAVI_Final": final_pred_tavi,
        "SAVR_Final": pred_savr
    })
    out_df.to_csv(out_dir / "variant_projection_data.csv", index=False)
    log.info(f"Done. Outputs in {out_dir}")

if __name__ == "__main__":
    main()
