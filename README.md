# From Capitol Hill to Capital Markets: Post-Disclosure Predictive Content of Congressional Trade Disclosures

**Author:** Kirill Sirik  
**Advisor:** Professor Jason Klusowski  
**Department:** Operations Research and Financial Engineering, Princeton University  
**Date:** April 2026  
**Submitted in partial fulfillment of the requirements for the degree of Bachelor of Science in Engineering.**

---

## Overview

This repository contains the complete data pipeline, analysis code, and output for the ORFE Junior Independent Work thesis investigating whether STOCK Act disclosure filings by members of the U.S. Congress predict post-disclosure stock returns. Using 25,770 trades by 158 legislators from February 2023 through January 2026, the paper finds that the average disclosure carries no predictive content, but that Senate purchase disclosures outperform comparable House purchases by approximately 1.1 percentage points (p < 0.05) — a signal that survives five risk-adjustment models, transaction costs, and out-of-sample validation.

---

## Repository Structure

```
ORFE_IW_KS_Spring_2026/
│
├── 00_raw_data/                          # Source data (not tracked — see Data Access)
│   ├── capitol_trades.csv                # Scraped from Capitol Trades (Feb 2023–Jan 2026)
│   ├── CRSP2022-2026.csv                 # CRSP Version 2 daily stock file (WRDS)
│   ├── F-F_Research_Data_5_Factors_2x3_daily.csv
│   ├── F-F_Momentum_Factor_daily.csv
│   └── F-F_Research_Data_Factors_daily.csv
│
├── 01_scripts/                           # Complete pipeline (10 scripts, ~4,100 lines)
│   ├── trim_crsp.py                      # Step 1:  Trim CRSP to 14 cols, filter to common stocks
│   ├── add_power_committee.py            # Step 1b: Augment trades with committee membership
│   ├── step2_merge_trades_crsp.py        # Step 2:  Clean Capitol Trades, merge to CRSP PERMNOs
│   ├── step2b_recover_unmatched.py       # Step 2b: Recover ticker-change and ShareType trades
│   ├── step3b_event_level_aggregation.py # Step 3b: Aggregate trades to disclosure events
│   ├── step4_event_windows_final.py      # Step 4:  Event dates, factor models, all ARs
│   ├── step5_6_descriptive_and_event_study.py  # Step 5-6: Descriptive tables, CARs, figures
│   ├── step7_regressions.py              # Step 7:  Trade-level cross-sectional regressions
│   ├── step7b_enhanced_regressions.py    # Step 7b: Winsorized, Khanna exclusion, large trades
│   ├── step7c_event_level_regressions.py # Step 7c: Event-level regressions, WLS
│   └── step8_trading_strategy.py         # Step 8:  Strategy backtest, walk-forward, costs
│
├── 02_intermediate_data/                 # Pipeline intermediates
│   ├── crsp_trimmed.csv                  # Trimmed CRSP (14 cols, 4.85M rows)
│   ├── capitol_trades_augmented.csv      # Trades + power committee flags
│   ├── factors_daily.csv                 # FF5 + MOM merged (920 trading days)
│   ├── trades_matched.csv                # 25,466 PERMNO-matched trades
│   ├── trades_unmatched.csv              # 2,499 unmatched trades (mostly ADRs)
│   ├── trades_recovered.csv              # 452 recovered ticker-change trades
│   └── trades_matched_final.csv          # 25,918 combined sample
│
├── 03_analysis_data/                     # Regression-ready datasets
│   ├── analysis_sample.csv               # MASTER trade-level file (25,770 obs, 100+ cols)
│   ├── analysis_sample_event_level.csv   # MASTER event-level file (1,081 obs, 104 cols)
│   └── event_study_daily.csv             # Daily CARs for event study plots (~3.6M rows)
│
├── 04_results_tables/                    # All CSV output tables
│   ├── table_3_1_summary_stats.csv
│   ├── table_6_1_party_activity.csv
│   ├── table_6_2_correlation_matrix.csv
│   ├── table_6_3_univariate_returns.csv
│   ├── regression_table_main.csv         # Trade-level Specs 1-5 (Table 6.1)
│   ├── regression_table_all_models.csv   # Spec 3 across 5 risk models (Table C.4)
│   ├── regression_table_all_horizons.csv # Spec 3 across 6 horizons
│   ├── regression_table_split_sample.csv # Buy vs Sell subsamples (Table 6.3)
│   ├── regression_vif.csv                # VIF diagnostics (Table 6.7)
│   ├── regression_table_winsorized.csv   # Winsorized Specs 1-5
│   ├── regression_table_no_khanna.csv    # Excluding Ro Khanna
│   ├── regression_table_large_trades.csv # >$15K subsample (Table 6.4)
│   ├── regression_event_main.csv         # Event-level Specs 1-5 (Table 6.5)
│   ├── regression_event_wls.csv          # OLS vs WLS (Table 6.6)
│   ├── regression_event_no_khanna.csv    # Event-level excl. Khanna
│   ├── regression_event_large.csv        # Large events >$30K
│   ├── regression_event_horizons.csv     # Event-level across horizons
│   ├── strategy_performance.csv          # Strategy backtest (Table 7.1)
│   ├── strategy_horizons.csv             # Strategy across horizons (Table 7.2)
│   ├── strategy_cost_sensitivity.csv     # Transaction cost analysis (Table 7.3)
│   └── strategy_walk_forward.csv         # Walk-forward test (Table 7.4)
│
├── 05_figures/
│   ├── figure_5_1_disclosure_lag_dist.png    # Disclosure lag histogram
│   ├── figure_6_1_car_dual_anchor.png        # Dual-anchor CAR comparison
│   ├── figure_7_1_strategy_comparison.png    # Strategy bar chart comparison
│   ├── figure_7_2_monthly_returns.png        # Monthly strategy returns
│   └── figure_ar_by_horizon.png              # AR by model and horizon
│
├── 06_diagnostics/                       # Pipeline logs
│   ├── merge_diagnostics.txt
│   ├── recovery_diagnostics.txt
│   ├── step3b_diagnostics.txt
│   ├── step4_diagnostics.txt
│   ├── step5_6_diagnostics.txt
│   ├── step7_diagnostics.txt
│   ├── step7b_diagnostics.txt
│   ├── step7c_diagnostics.txt
│   ├── step8_diagnostics.txt
│   └── sample_attrition.csv
│
├── 07_audit/
│   ├── step4_methodology_audit.txt       # Equation-by-equation audit of Step 4
│   ├── full_pipeline_audit.txt           # Complete pipeline audit
│   └── project_reference.txt            # Master reference document
│
├── paper/
│   ├── main.tex                          # LaTeX source
│   └── ORFE_Junior_IW_KS.pdf            # Compiled thesis
│
├── README.md                             # This file
└── requirements.txt                      # Python dependencies
```

---

## Pipeline Execution

### Prerequisites

- Python 3.11+
- Dependencies: `pip install -r requirements.txt`

### Data Access

The raw data is **not included** in this repository due to licensing restrictions:

- **Capitol Trades:** Scraped from [capitoltrades.com](https://www.capitoltrades.com) (February 2023 – January 2026). The scraper is not included; the raw CSV contains 35,016 trade records.
- **CRSP:** Daily stock file (Version 2) accessed through [WRDS](https://wrds-www.wharton.upenn.edu/) via Princeton University. Requires institutional credentials. Download: Stock → Version 2 → Daily → All variables → June 2022 – February 2026.
- **Fama-French Factors:** Downloaded from [Kenneth French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Files: "Fama/French 5 Factors (2x3) — Daily" and "Momentum Factor — Daily."

Place all raw data files in `00_raw_data/` before running the pipeline.

### Execution Order

Scripts must be run sequentially due to data dependencies. Total runtime is approximately 15 minutes on a modern laptop (Apple M-series or equivalent), with Step 4 accounting for ~10 minutes.

```bash
cd 01_scripts/

# Stage 1: Data preparation
python trim_crsp.py                         # 94 → 14 cols, 9.1M → 4.85M rows
python add_power_committee.py               # Append committee membership flags

# Stage 2: Trade-CRSP matching
python step2_merge_trades_crsp.py           # Clean trades, merge PERMNOs (91.1% match)
python step2b_recover_unmatched.py          # Recover ticker changes (+452 trades → 92.7%)

# Combine matched + recovered (one-liner)
python -c "
import pandas as pd
m = pd.read_csv('../02_intermediate_data/trades_matched.csv')
r = pd.read_csv('../02_intermediate_data/trades_recovered.csv')
pd.concat([m, r], ignore_index=True).to_csv('../02_intermediate_data/trades_matched_final.csv', index=False)
"

# Stage 3: Event windows and abnormal returns (~10 min)
python step4_event_windows_final.py         # 5 factor models × 6 horizons × 25,918 trades

# Stage 4: Analysis
python step5_6_descriptive_and_event_study.py   # Descriptive tables + CAR figures
python step7_regressions.py                     # Trade-level Specs 1-5
python step7b_enhanced_regressions.py           # Robustness variants
python step3b_event_level_aggregation.py        # Aggregate to 1,081 disclosure events
python step7c_event_level_regressions.py        # Event-level Specs 1-5 + WLS

# Stage 5: Strategy backtest
python step8_trading_strategy.py                # 5 strategies, costs, walk-forward
```

---

## Key Results

| Finding | Estimate | Significance | Robust? |
|---|---|---|---|
| Average post-disclosure AR | +0.04% | p = 0.517 | Yes — all models, all samples |
| Senate buy advantage (20d) | +1.13% | p < 0.05 | Yes — all models, costs, OOS |
| Disclosure lag (event-level) | +0.037%/day | p < 0.05 | Partial — loses sig. with legislator FE |
| Power committee (within-legislator) | +3.86% | p < 0.05 | Partial — strategy collapses OOS |
| Senate Buys strategy Sharpe | 1.51 (1.23 net) | — | Yes — 1.49 in walk-forward test |

---

## Methodology

- **Abnormal returns:** Market-adjusted (primary), CAPM, FF3, Carhart, FF5
- **Holding periods:** 5, 10, 20, 40, 60, 127 trading days
- **Standard errors:** Two-way clustered at legislator × disclosure-month (Cameron, Gelbach, and Miller, 2011)
- **Event-level aggregation:** Value-weighted signed ARs within (legislator, disclosure date) pairs
- **Robustness:** Winsorization, Ro Khanna exclusion, large-trade subsample, WLS, walk-forward OOS

---

## Software

| Package | Version | Purpose |
|---|---|---|
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.24 | Numerical computation |
| scipy | ≥ 1.10 | Statistical functions |
| statsmodels | ≥ 0.14 | Regression diagnostics |
| matplotlib | ≥ 3.7 | Figures |

All analyses conducted on Python 3.11, macOS (Apple Silicon).

---

## Citation

```
Sirik, K. (2026). From Capitol Hill to Capital Markets: Post-Disclosure Predictive
Content of Congressional Trade Disclosures. Junior Independent Work, Department
of Operations Research and Financial Engineering, Princeton University.
Advisor: J. Klusowski.
```

---

## License

This repository is made available for academic and research purposes. The code may be reused with attribution. The raw data is subject to the licensing terms of Capitol Trades, WRDS/CRSP, and Kenneth French's Data Library, respectively.
