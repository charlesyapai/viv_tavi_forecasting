
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_population_korea_combined(csv_path):
    log.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    df = df.T
    df.index.name = "year"
    df.index = df.index.astype(int)
    
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
    
    scale = 10000
    w_pop = (
        pop_u65 * 0.01 +
        pop_65_69 * 0.10 +
        pop_70_74 * 0.40 +
        pop_75_84 * 0.80 +
        pop_85_up * 1.00
    ) * scale
    
    return pd.DataFrame({"year": df.index, "weighted_pop": w_pop.values})

def main():
    pop_korea_combined_path = Path("simulation_run_v1/data/korea_population_combined.csv")

    korea_wpop_df = load_population_korea_combined(pop_korea_combined_path)
    korea_wpop_yearly = korea_wpop_df.set_index("year")["weighted_pop"]
    
    log.info(f"Korea Index Unique? {korea_wpop_yearly.index.is_unique}")
    
    wpop_dict = korea_wpop_yearly.to_dict()
    min_k_year = min(wpop_dict.keys())
    
    growth_rate = 0.035
    for y in range(min_k_year - 1, 2011, -1):
        next_val = wpop_dict[y+1]
        wpop_dict[y] = next_val / (1 + growth_rate)
        
    wpop_yearly = pd.Series(wpop_dict).sort_index()
    log.info(f"Final WPop Index Unique? {wpop_yearly.index.is_unique}")

if __name__ == "__main__":
    main()
