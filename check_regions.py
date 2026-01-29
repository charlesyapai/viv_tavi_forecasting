
import pandas as pd
from pathlib import Path

def check_file(path):
    print(f"--- Checking {path} ---")
    try:
        # Skip first row because it might be metadata, but based on head it looked like header is line 1
        # Actually in the head output: Index,Variant,"Region..." is the first line.
        df = pd.read_csv(path)
        col = "Region, subregion, country or area *"
        if col in df.columns:
            uniques = df[col].unique()
            print("Unique Regions found:")
            for u in uniques:
                print(f" - {u}")
        else:
            print(f"Column '{col}' not found. Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading {path}: {e}")

check_file("simulation_run_v1/data/un_population_data/Population Projections - Male, Korea.csv")
check_file("simulation_run_v1/data/un_population_data/Population Projections - Female, Korea.csv")
