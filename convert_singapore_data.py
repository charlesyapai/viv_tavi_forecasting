
import pandas as pd
import glob
import os
import numpy as np

# Configuration
input_dir = "data/raw/singapore_raw"
output_dir = "simulation_run_v1/data"
sheet_mapping = {
    "TAVI volume": "singapore_registry_tavi.csv",
    "SAVR volume": "singapore_registry_savr.csv",
    "Redo SAVR volume": "singapore_registry_redo_savr.csv"
}

files = glob.glob(os.path.join(input_dir, "*.xlsx"))
print(f"Found files: {files}")

# Container for aggregated data
# key: output_filename, value: list of dataframes
aggregated_data = {v: [] for v in sheet_mapping.values()}

for f in files:
    print(f"Processing file: {f}")
    xls = pd.ExcelFile(f)
    
    for sheet_name, output_filename in sheet_mapping.items():
        if sheet_name not in xls.sheet_names:
            print(f"  Warning: Sheet '{sheet_name}' not found in {f}")
            continue
            
        print(f"  Processing sheet: {sheet_name}")
        
        # Read the header row for years (row 2, 0-indexed)
        # We read the whole file to be safe, but we know structure
        # Actually easier to read with header=None and parse manually
        df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        
        # Years are in row index 2
        # Columns 0 (Sex) and 1 (Age Group) are identifiers
        year_row = df_raw.iloc[2]
        years = year_row.iloc[2:].tolist()
        
        # Clean years: remove NaN, convert floats to ints if possible
        # We expect 2012, 2013, etc.
        valid_year_indices = []
        cleaned_years = []
        
        for i, y in enumerate(years):
            col_idx = i + 2
            if pd.isna(y):
                continue
            try:
                # Handle "2013.0" -> 2013
                y_int = int(float(y))
                cleaned_years.append(y_int)
                valid_year_indices.append(col_idx)
            except ValueError:
                continue
                
        # Data starts at row index 4
        data_start_row = 4
        df_data = df_raw.iloc[data_start_row:].copy()
        
        # Rename columns
        # We only keep Sex, Age Group, and valid year columns
        cols_to_keep = [0, 1] + valid_year_indices
        df_data = df_data[cols_to_keep]
        
        # Rename columns for melting
        new_col_names = ["sex", "age_band"] + cleaned_years
        df_data.columns = new_col_names
        
        # Forward fill Sex if needed (handling merged cells)
        df_data["sex"] = df_data["sex"].ffill()
        
        # Filter out rows where Sex or Age Band is NaN
        df_data = df_data.dropna(subset=["sex", "age_band"])
        
        # Filter out "Subtotal" or "Total" or similar in age_band
        # Convert to string first
        df_data = df_data[~df_data["age_band"].astype(str).str.contains("Subtotal|Total|Sum|Missing", case=False, regex=True)]
        
        # Melt
        df_melted = df_data.melt(id_vars=["sex", "age_band"], var_name="year", value_name="count")
        
        # Clean data
        df_melted["count"] = pd.to_numeric(df_melted["count"], errors='coerce').fillna(0).astype(int)
        df_melted["year"] = df_melted["year"].astype(int)
        
        # Append to list
        aggregated_data[output_filename].append(df_melted)

# Concatenate and sum
for output_filename, dfs in aggregated_data.items():
    if not dfs:
        print(f"No data found for {output_filename}")
        continue
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Clean up Sex and Age Band values
    # Standardize Sex: "Men" -> "Men", "Women" -> "Women" (already correct likely)
    # Standardize Age Band: "<5yr" -> "<5yr", "5-9" -> "5-9"
    # Ensure they match the expected format in registry_*.csv
    # The existing registry files use "Men", "Women", and bands like "10-14", ">=80"
    
    # Group by sex, age_band, year and sum counts
    grouped = full_df.groupby(["sex", "age_band", "year"])["count"].sum().reset_index()
    
    # Sort
    grouped = grouped.sort_values(by=["year", "sex", "age_band"])
    
    # Output path
    out_path = os.path.join(output_dir, output_filename)
    grouped.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(grouped)} rows.")
