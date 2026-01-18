
import pandas as pd
import glob

files = glob.glob("data/raw/singapore_raw/*.xlsx")
sheets = ["TAVI volume", "SAVR volume", "Redo SAVR volume"]

for f in files:
    print(f"--- File: {f} ---")
    xls = pd.ExcelFile(f)
    print("Sheets:", xls.sheet_names)
    
    for s in sheets:
        if s not in xls.sheet_names:
            print(f"Sheet {s} not found!")
            continue
        
        print(f"  --- Sheet: {s} ---")
        df = pd.read_excel(xls, sheet_name=s, header=None)
        
        # Determine where the years are
        # Look for a row that contains integers like 2012, 2024, etc.
        # We can iterate first few rows
        found_year_row = -1
        for i in range(10):
            row_vals = df.iloc[i].astype(str).tolist()
            # simple heuristic: looks for 20xx
            years = [v for v in row_vals if v.startswith('20') and len(v) >= 4]
            if len(years) > 2:
                print(f"    Possible header row {i}: {row_vals}")
                found_year_row = i
                break
        
        if found_year_row != -1:
             print("    First few data rows:")
             print(df.iloc[found_year_row+1:found_year_row+5])
