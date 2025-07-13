#!/usr/bin/env python
"""
Convert the raw Registry TAVI table (wide format) into the long format
required by model.py:

    year,age_band,sex,count

USAGE:
    python processing/convert_registry.py data/raw/registry_tavi_raw.csv data/registry_tavi.csv


    OR


    python processing/convert_registry.py data/raw/registry_savr_raw.csv data/registry_savr.csv

    
    OR


    python processing/convert_registry.py data/raw/registry_redo_savr_raw.csv data/registry_redo_savr.csv
"""

import sys
import pandas as pd
from pathlib import Path


def main(infile: str, outfile: str):
    # ---------------------------------------------------------------------
    # 1. read ----------------------------------------------------------------
    df = pd.read_csv(
        infile,
        na_values=["-", ""],          # treat dash or blank as NaN
        keep_default_na=True,
    )

    # normalise column names
    df.rename(
        columns={"Sex": "sex", "Age group": "age_band"},
        inplace=True,
    )

    # ---------------------------------------------------------------------
    # 2. remove subtotal or empty rows -------------------------------------
    mask_subtotal = df["age_band"].str.contains("Subtotal", na=False)
    df = df.loc[~mask_subtotal].copy()

    # ---------------------------------------------------------------------
    # 3. melt into long form ----------------------------------------------
    year_cols = [c for c in df.columns if c.isdigit()]
    long = df.melt(
        id_vars=["sex", "age_band"],
        value_vars=year_cols,
        var_name="year",
        value_name="count",
    )

    # ---------------------------------------------------------------------
    # 4. clean numeric column ---------------------------------------------
    long["count"] = (
        long["count"]
        .astype(str)
        .str.replace(",", "", regex=False)   # remove thousands comma
        .astype(float)                       # convert to number
        .fillna(0)
        .astype(int)
    )

    # drop zero rows (optional, but tidy)
    long = long[long["count"] > 0]

    # ---------------------------------------------------------------------
    # 5. final typing & sort ----------------------------------------------
    long["year"] = long["year"].astype(int)
    long.sort_values(["year", "sex", "age_band"], inplace=True)

    # ---------------------------------------------------------------------
    # 6. write out ---------------------------------------------------------
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(outfile, index=False)
    print(f"✔️  Saved tidy file → {outfile}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python convert_registry.py <infile.csv> <outfile.csv>")
    main(sys.argv[1], sys.argv[2])
