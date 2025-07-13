# viv_tavi_forecasting
This is the repository used to conduct experiments related to the ViV TAVI study, using primarily a MC simulation model.


Project folder structure:
```bash
viv-tavi/
├─ data/
│  ├─ registry_redo_savr.csv        # yearly counts by age-band×sex
│  ├─ registry_tavi.csv             # same for index-TAVI
│  └─ redo_trends.csv               # SAVR-after-SAVR & SAVR-after-TAVR
├─ config.yaml                      # all model inputs & assumptions
├─ model.py                         # main simulation engine
├─ postprocess.py                   # plots & summary tables
# └─ tests/                         # For later validation
#    ├─ test_distributions.py
#    └─ test_mass_balance.py
```


To run,  

# 1. Install deps (conda or pip)
pip install numpy pandas scipy pyyaml matplotlib pydantic

# 2. Execute the model
python model.py --config config.yaml

# 3. Make the plot
python postprocess.py out/viv_forecast.csv --outdir out/figs




# Redo-SAVR subtraction

Local registry extracts that differentiate redo-SAVR (eg, SAVR-after-SAVR or SAVR-after-TAVI) are not yet available.

To avoid inflating the ViV forecast we apply literature-based fixed discounts—4 % of surgical valve failures are assumed to receive a second open surgical valve (savr_after_savr) and 1 % of TAVI failures undergo open redo surgery (savr_after_tavi).

These proportions are specified in config.yaml → redo_rates. 

When national data becomes available, we can replace the percentages with year-specific counts, or set the rates to 0 if we are looking to model an “all-ViV” scenario.


From the original paper, we are assuming:

| redo\_type        | meaning                                                    | typical 2022 volume (US paper) |
| ----------------- | ---------------------------------------------------------- | ------------------------------ |
| `savr_after_savr` | a second surgical valve in a failed surgical bioprosthesis | \~4 – 6 % of all SAVR failures |
| `savr_after_tavi` | a surgical valve in a failed TAVI prosthesis               | <1 % of all TAVI failures      |




This is reflected in our configuration where we state that:

```yaml

redo_rates:
  savr_after_savr: 0.04    # 4 % of SAVR failures each year
  savr_after_tavi: 0.01    # 1 % of TAVI failures



