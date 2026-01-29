
import pandas as pd
import numpy as np
from pathlib import Path

def load_population_korea_combined(csv_path):
    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    df = df.T
    df.index.name = "year"
    # Ensure index is int
    df.index = df.index.astype(int)
    print(f"Korea Combined Index: {df.index}")
    print(f"Duplicates? {df.index.duplicated().any()}")
    
    # ... logic skipped for brevity, just returning dummy series with index ...
    # Wait, error might be in the logic I skipped?
    # But error happens later.
    return df.index

def load_population_un_flexible(male_path, female_path):
    print("Loading UN Data")
    def read_melt(path):
        df = pd.read_csv(path)
        if "Variant" in df.columns:
            df = df[df["Variant"] == "Estimates"]
        # Melt
        age_cols = [c for c in df.columns if c[0].isdigit()]
        df = df[["Year"] + age_cols]
        # Strict dedupe
        df = df.groupby("Year").first().reset_index()
        
        df = df.melt(id_vars=["Year"], var_name="age_band", value_name="pop_str")
        df["population"] = pd.to_numeric(df["pop_str"].astype(str).str.replace(" ", ""), errors='coerce') * 1000
        return df
        
    m = read_melt(male_path)
    f = read_melt(female_path)
    
    df = pd.merge(m, f, on=["Year", "age_band"], suffixes=("_m", "_f"))
    df["total_pop"] = df["population_m"] + df["population_f"]
    df["age_start"] = df["age_band"].str.extract(r'(\d+)').astype(int)
    
    # Calculate weighted
    # Just return the groupby result
    wpop = df.groupby("Year")["total_pop"].sum().sort_index()
    print(f"UN Weighted Pop Index: {wpop.index}")
    print(f"Duplicates? {wpop.index.duplicated().any()}")
    return wpop

def main():
    tavi_path = Path("simulation_run_v1/data/registry_tavi.csv") 
    savr_path = Path("simulation_run_v1/data/registry_savr.csv")
    pop_un_m_path = Path("simulation_run_v1/data/un_population_data/Population Projections - Male, Korea.csv")
    pop_un_f_path = Path("simulation_run_v1/data/un_population_data/Population Projections - Female, Korea.csv")
    pop_korea_combined_path = Path("simulation_run_v1/data/korea_population_combined.csv")

    k_idx = load_population_korea_combined(pop_korea_combined_path)
    un_series = load_population_un_flexible(pop_un_m_path, pop_un_f_path)
    
    overlap = 2022
    print(f"Overlap {overlap} in Korea? {overlap in k_idx}")
    print(f"Overlap {overlap} in UN? {overlap in un_series.index}")

if __name__ == "__main__":
    main()
