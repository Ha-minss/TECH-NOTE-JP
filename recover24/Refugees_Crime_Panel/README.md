# Refugee Inflows → Crime Rates (Country-Year Panel, 2013–2022)

Refugee inflows are often claimed to affect public safety, but cross-country evidence is mixed. Using a country–year panel (2013–2022), we test whether higher refugees per 100k are associated with changes in crime rates. The results show no robust positive relationship with violent, narcotic, or sexual crime, and the baseline property-crime correlation weakens once country-specific trends are included.

This repository provides a reproducible panel-data analysis using two-way fixed effects (country + year) with country-clustered standard errors, plus dynamic lag/lead placebo checks and robustness tests (country-specific trends, year-drop sensitivity, first differences).
- **Study window:** 2013–2022  
- **Unit:** country × year  
- **Inference:** clustered SE by country  
- **Scope:** refugees-only specifications (logged per-100k rates)

---

## What you get (outputs)

Running the scripts regenerates:

- **Main TWFE table (5 outcomes):** `outputs/tables/main_twfe_refugees_only.csv`  
- **Dynamic checks (separate blocks):** `outputs/tables/dynamic_curr.csv`, `dynamic_lag1.csv`, `dynamic_lag2.csv`, `dynamic_lead1_placebo.csv`  
- **Dynamic checks (joint model):** `outputs/tables/dynamic_joint_t_lags_lead.csv`  
- **Robustness (country trends):** `outputs/tables/robustness_country_trends.csv`  
- **Robustness (drop-year sensitivity):** `outputs/tables/robustness_drop_years.csv`  
- **First difference (Year FE):** `outputs/tables/first_difference_year_fe.csv`  
- **Coefficient plot:** `outputs/figures/coefplot_main_twfe.png`

---

## Quickstart (single command)

```bash
pip install -r requirements.txt
PYTHONPATH=$PWD:$PYTHONPATH python scripts/run_all.py
```

This will:

- **Regenerate all CSV tables under outputs/tables/

- **Regenerate the coefficient plot under outputs/figures/

- **Print every table to the console for inspection

All outputs are saved under outputs/.

Repository layout
See scripts/ for the runnable pipeline and src/ref_crime/ for shared helpers.

---

## Citation
If you use this repository or its outputs, please cite:

Park, Dongha. Refugee Inflows → Crime Rates: Country-Year Panel Analysis (TWFE + Robustness Suite). (Year). Repository: https://github.com/Ha-minss/Portfolio/tree/main/Refugees_Crime_Panel

### BibTeX (optional)
```
@misc{park_refugees_crime_panel,
  author       = {Park, Dongha},
  title        = {Refugee Inflows → Crime Rates: Country-Year Panel Analysis (TWFE + Robustness Suite)},
  year         = {2026},
  howpublished = {\url{https://github.com/Ha-minss/Portfolio/tree/main/Refugees_Crime_Panel}},
  note         = {Accessed 2026-02-04}
}
```
---

## Contact
If you find a bug or want to reproduce a specific table/figure, open an issue (or message me) with:

* script name
* exact command you ran
* full error log
