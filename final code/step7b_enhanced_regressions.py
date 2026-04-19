"""
step7b_enhanced_regressions.py
===============================
Enhanced regression analysis adding three improvements:
  1. Winsorization of ARs at 1%/99% (all models, all horizons)
  2. Excluding Ro Khanna (42% of sample) as robustness
  3. Large-trade subsample (>$15K bracket)

All ORIGINAL results from step7_regressions.py are preserved.
This script produces ADDITIONAL tables.

Input:
  - analysis_sample.csv (from Step 4)

Output:
  - regression_table_winsorized.csv        (Specs 1-5, winsorized, 20d)
  - regression_table_no_khanna.csv         (Specs 1-5, excl. Ro Khanna, 20d)
  - regression_table_large_trades.csv      (Specs 1-5, trades >$15K, 20d)
  - regression_table_combined_robustness.csv (Spec 3 across all enhancements)
  - regression_table_winsorized_horizons.csv (Spec 3, winsorized, all horizons)
  - regression_table_winsorized_models.csv  (Spec 3, winsorized, all models)
  - regression_table_winsorized_split.csv   (Buy/Sell, winsorized)
  - step7b_diagnostics.txt

Usage:
    python step7b_enhanced_regressions.py
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import t as t_dist
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =====================================================================
# CONFIGURATION
# =====================================================================
ANALYSIS_PATH = "analysis_sample.csv"
OUTPUT_DIR = "."

HOLDING_PERIODS = [5, 10, 20, 40, 60, 127]
MODELS = ["market", "CAPM", "FF3", "Carhart", "FF5"]
PRIMARY_OMEGA = 20

WINSORIZE_LO = 0.01
WINSORIZE_HI = 0.99

MIN_TRADES_LEGISLATOR_FE = 5

LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


# =====================================================================
# CLUSTERING (same as step7)
# =====================================================================

def one_way_cluster_cov(X, resid, clusters):
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    meat = np.zeros((k, k))
    for g in unique_clusters:
        mask = clusters == g
        Xg = X[mask]
        eg = resid[mask]
        score_g = Xg.T @ eg
        meat += np.outer(score_g, score_g)

    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V = correction * XtX_inv @ meat @ XtX_inv
    return V


def twoway_cluster_cov(X, resid, cluster1, cluster2):
    cluster12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(cluster1, cluster2)])

    V1 = one_way_cluster_cov(X, resid, cluster1)
    V2 = one_way_cluster_cov(X, resid, cluster2)
    V12 = one_way_cluster_cov(X, resid, cluster12)

    V = V1 + V2 - V12

    eigvals_full, eigvecs = np.linalg.eigh(V)
    eigvals_full = np.maximum(eigvals_full, 0)
    V = eigvecs @ np.diag(eigvals_full) @ eigvecs.T

    return V


def run_ols_twoway(y, X, cluster1, cluster2, var_names):
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y_c = y[valid]
    X_c = X[valid]
    cl1 = cluster1[valid]
    cl2 = cluster2[valid]

    n, k = X_c.shape
    if n < k + 10:
        return None

    beta = np.linalg.lstsq(X_c, y_c, rcond=None)[0]
    resid = y_c - X_c @ beta

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_c - np.mean(y_c)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    V_two = twoway_cluster_cov(X_c, resid, cl1, cl2)
    se_two = np.sqrt(np.diag(V_two))

    V_one = one_way_cluster_cov(X_c, resid, cl1)
    se_one = np.sqrt(np.diag(V_one))

    t_stats = beta / se_two
    df = min(len(np.unique(cl1)), len(np.unique(cl2))) - 1
    df = max(df, 1)
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=df))

    return {
        "var_names": var_names,
        "beta": beta,
        "se_twoway": se_two,
        "se_oneway": se_one,
        "t_stats": t_stats,
        "p_values": p_values,
        "r2": r2,
        "adj_r2": adj_r2,
        "n_obs": n,
        "n_legislators": len(np.unique(cl1)),
        "n_months": len(np.unique(cl2)),
    }


def format_coef(beta, se, p):
    stars = ""
    if p < 0.01:
        stars = "***"
    elif p < 0.05:
        stars = "**"
    elif p < 0.10:
        stars = "*"
    return f"{beta:.5f}{stars}", f"({se:.5f})"


REPORT_VARS = [
    "Intercept", "Buy", "Senate", "Republican",
    "ln_Size", "ln_Size_resid",
    "Disc_Lag", "Disc_Lag_Sq",
    "Buy×Senate", "Buy×Republican",
    "ln_MktCap", "Volatility", "PowerCommittee",
]


def sic_to_sector(sic):
    try:
        s = int(str(sic)[:2])
    except (ValueError, TypeError):
        return "Other"
    if s <= 9: return "Agriculture"
    elif s <= 14: return "Mining"
    elif s <= 17: return "Construction"
    elif s <= 39: return "Manufacturing"
    elif s <= 49: return "Utilities/Transport"
    elif s <= 51: return "Wholesale"
    elif s <= 59: return "Retail"
    elif s <= 67: return "Finance"
    elif s <= 89: return "Services"
    else: return "Other"


# =====================================================================
# SPECIFICATION BUILDERS
# =====================================================================

def build_spec(d, spec_num, legs_with_enough):
    """Build X, var_names, cl1, cl2 for a given spec on dataframe d."""

    if spec_num == 5:
        d = d[d["politician_name"].isin(legs_with_enough)].copy()
        if len(d) < 100:
            return None, None, None, None, None

    var_names = ["Intercept"]
    X_list = [np.ones(len(d))]

    if spec_num <= 4:
        X_list.append(d["buy_indicator"].values)
        var_names.append("Buy")
        X_list.append(d["senate_indicator"].values)
        var_names.append("Senate")
        X_list.append(d["republican_indicator"].values)
        var_names.append("Republican")
    else:
        # Spec 5: Buy survives, Senate/Republican absorbed by legislator FE
        X_list.append(d["buy_indicator"].values)
        var_names.append("Buy")

    if spec_num <= 2:
        X_list.append(d["ln_size"].values)
        var_names.append("ln_Size")
    else:
        X_list.append(d["ln_size_resid"].values)
        var_names.append("ln_Size_resid")

    X_list.append(d["disclosure_lag"].values)
    var_names.append("Disc_Lag")
    X_list.append(d["disclosure_lag_sq"].values)
    var_names.append("Disc_Lag_Sq")

    if spec_num >= 2:
        X_list.append(d["buy_x_senate"].values)
        var_names.append("Buy×Senate")
        X_list.append(d["buy_x_republican"].values)
        var_names.append("Buy×Republican")

    if spec_num >= 3:
        X_list.append(d["ln_mktcap"].values)
        var_names.append("ln_MktCap")
        X_list.append(d["volatility"].values)
        var_names.append("Volatility")
        X_list.append(d["power_committee"].values.astype(float))
        var_names.append("PowerCommittee")

    if spec_num >= 4:
        sector_dummies = pd.get_dummies(d["sector"], prefix="Sec", drop_first=True)
        for col in sector_dummies.columns:
            X_list.append(sector_dummies[col].values.astype(float))
            var_names.append(col)
        my_dummies = pd.get_dummies(d["disc_month_year"], prefix="MY", drop_first=True)
        for col in my_dummies.columns:
            X_list.append(my_dummies[col].values.astype(float))
            var_names.append(col)

    if spec_num == 5:
        leg_dummies = pd.get_dummies(d["politician_name"], prefix="Leg", drop_first=True)
        for col in leg_dummies.columns:
            X_list.append(leg_dummies[col].values.astype(float))
            var_names.append(col)

    X = np.column_stack(X_list)
    cl1 = d["cluster_legislator"].values
    cl2 = d["cluster_month"].values

    return X, var_names, cl1, cl2, d


def build_split_spec(d, spec_num):
    """Build matrices for split-sample (no Buy, no interactions)."""
    var_names = ["Intercept"]
    X_list = [np.ones(len(d))]

    X_list.append(d["senate_indicator"].values)
    var_names.append("Senate")
    X_list.append(d["republican_indicator"].values)
    var_names.append("Republican")

    if spec_num <= 2:
        X_list.append(d["ln_size"].values)
        var_names.append("ln_Size")
    else:
        X_list.append(d["ln_size_resid"].values)
        var_names.append("ln_Size_resid")

    X_list.append(d["disclosure_lag"].values)
    var_names.append("Disc_Lag")
    X_list.append(d["disclosure_lag_sq"].values)
    var_names.append("Disc_Lag_Sq")

    if spec_num >= 3:
        X_list.append(d["ln_mktcap"].values)
        var_names.append("ln_MktCap")
        X_list.append(d["volatility"].values)
        var_names.append("Volatility")
        X_list.append(d["power_committee"].values.astype(float))
        var_names.append("PowerCommittee")

    if spec_num >= 4:
        sector_dummies = pd.get_dummies(d["sector"], prefix="Sec", drop_first=True)
        for col in sector_dummies.columns:
            X_list.append(sector_dummies[col].values.astype(float))
            var_names.append(col)
        my_dummies = pd.get_dummies(d["disc_month_year"], prefix="MY", drop_first=True)
        for col in my_dummies.columns:
            X_list.append(my_dummies[col].values.astype(float))
            var_names.append(col)

    X = np.column_stack(X_list)
    cl1 = d["cluster_legislator"].values
    cl2 = d["cluster_month"].values

    return X, var_names, cl1, cl2


def run_spec(df_sub, dep_col, spec_num, legs_with_enough):
    """Run one specification and return result dict."""
    X, var_names, cl1, cl2, d_used = build_spec(df_sub, spec_num, legs_with_enough)
    if X is None:
        return None
    y = d_used[dep_col].values.astype(float)
    return run_ols_twoway(y, X, cl1, cl2, var_names)


def format_table(results_dict, column_labels, specs_or_keys):
    """Format regression results into a readable table."""
    table_rows = []
    for var in REPORT_VARS:
        coef_row = {"Variable": var}
        se_row = {"Variable": ""}
        for key, col_label in zip(specs_or_keys, column_labels):
            if key in results_dict and results_dict[key] is not None:
                res = results_dict[key]
                if var in res["var_names"]:
                    idx = res["var_names"].index(var)
                    c, s = format_coef(res["beta"][idx], res["se_twoway"][idx], res["p_values"][idx])
                    coef_row[col_label] = c
                    se_row[col_label] = s
                else:
                    coef_row[col_label] = ""
                    se_row[col_label] = ""
            else:
                coef_row[col_label] = ""
                se_row[col_label] = ""
        table_rows.append(coef_row)
        table_rows.append(se_row)

    for stat_name, stat_key in [("N", "n_obs"), ("R²", "r2"), ("Adj R²", "adj_r2"),
                                 ("Legislators", "n_legislators"), ("Months", "n_months")]:
        row = {"Variable": stat_name}
        for key, col_label in zip(specs_or_keys, column_labels):
            if key in results_dict and results_dict[key] is not None:
                val = results_dict[key][stat_key]
                if stat_name in ("N", "Legislators", "Months"):
                    row[col_label] = f"{val:,}"
                else:
                    row[col_label] = f"{val:.4f}"
            else:
                row[col_label] = ""
        table_rows.append(row)

    for fe_name, specs_with in [("Sector FE", [4, 5]), ("Month-Year FE", [4, 5]),
                                  ("Legislator FE", [5])]:
        row = {"Variable": fe_name}
        for key, col_label in zip(specs_or_keys, column_labels):
            if isinstance(key, int):
                row[col_label] = "Yes" if key in specs_with else "No"
            else:
                row[col_label] = ""
        table_rows.append(row)

    return pd.DataFrame(table_rows)


def main():
    start_time = time.time()
    log("=" * 70)
    log("STEP 7B: ENHANCED REGRESSIONS")
    log("(Winsorization + Ro Khanna Exclusion + Large Trades)")
    log("=" * 70)

    # ==================================================================
    # LOAD AND PREPARE DATA
    # ==================================================================
    log("\n--- Loading data ---")
    df = pd.read_csv(ANALYSIS_PATH)
    log(f"Loaded {len(df):,} trades")

    for col in ["buy_indicator", "senate_indicator", "republican_indicator",
                "power_committee", "disclosure_lag", "disclosure_lag_sq",
                "ln_size", "ln_mktcap", "volatility", "ln_size_resid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["buy_x_senate"] = df["buy_indicator"] * df["senate_indicator"]
    df["buy_x_republican"] = df["buy_indicator"] * df["republican_indicator"]
    df["sector"] = df["siccd"].apply(sic_to_sector)
    df["cluster_legislator"] = df["politician_name"].astype(str)
    df["cluster_month"] = df["disc_month_year"].astype(str)

    leg_counts = df["politician_name"].value_counts()
    legs_with_enough = set(leg_counts[leg_counts >= MIN_TRADES_LEGISLATOR_FE].index)

    # ==================================================================
    # STEP 1: WINSORIZE ALL AR COLUMNS
    # ==================================================================
    log("\n--- Winsorizing abnormal returns at 1%/99% ---")

    ar_cols = []
    for omega in HOLDING_PERIODS:
        for model in MODELS:
            for prefix in [f"AR_{model}_{omega}d", f"AR_signed_{model}_{omega}d"]:
                if prefix in df.columns:
                    ar_cols.append(prefix)
        for prefix in [f"raw_ret_{omega}d", f"sp500_ret_{omega}d", f"AR_market_trade_{omega}d"]:
            if prefix in df.columns:
                ar_cols.append(prefix)

    ar_cols = list(set(ar_cols))
    log(f"Winsorizing {len(ar_cols)} AR columns")

    # Create winsorized copy
    dfw = df.copy()
    for col in ar_cols:
        vals = dfw[col].dropna()
        if len(vals) > 100:
            lo = vals.quantile(WINSORIZE_LO)
            hi = vals.quantile(WINSORIZE_HI)
            dfw[col] = dfw[col].clip(lo, hi)

    # Check impact
    sample_col = f"AR_signed_market_{PRIMARY_OMEGA}d"
    log(f"\n  {sample_col}:")
    log(f"    Before: mean={df[sample_col].mean():.5f}, std={df[sample_col].std():.5f}, "
        f"min={df[sample_col].min():.5f}, max={df[sample_col].max():.5f}")
    log(f"    After:  mean={dfw[sample_col].mean():.5f}, std={dfw[sample_col].std():.5f}, "
        f"min={dfw[sample_col].min():.5f}, max={dfw[sample_col].max():.5f}")

    # Recompute residualized size on winsorized data (same regression)
    mask_both = dfw["ln_size"].notna() & dfw["ln_mktcap"].notna()
    if mask_both.sum() > 100:
        coeffs = np.polyfit(dfw.loc[mask_both, "ln_mktcap"].values,
                            dfw.loc[mask_both, "ln_size"].values, 1)
        dfw["ln_size_resid"] = dfw["ln_size"] - (coeffs[0] * dfw["ln_mktcap"] + coeffs[1])

    # ==================================================================
    # STEP 2: CREATE SUBSAMPLES
    # ==================================================================
    log("\n--- Creating subsamples ---")

    # No Ro Khanna
    df_no_khanna = dfw[dfw["politician_name"] != "Ro Khanna"].copy()
    log(f"Excluding Ro Khanna: {len(df_no_khanna):,} trades "
        f"(dropped {len(dfw) - len(df_no_khanna):,})")

    # Recompute legs_with_enough for no-Khanna sample
    nk_leg_counts = df_no_khanna["politician_name"].value_counts()
    nk_legs_enough = set(nk_leg_counts[nk_leg_counts >= MIN_TRADES_LEGISLATOR_FE].index)

    # Recompute residualized size for no-Khanna
    mask_nk = df_no_khanna["ln_size"].notna() & df_no_khanna["ln_mktcap"].notna()
    if mask_nk.sum() > 100:
        coeffs_nk = np.polyfit(df_no_khanna.loc[mask_nk, "ln_mktcap"].values,
                               df_no_khanna.loc[mask_nk, "ln_size"].values, 1)
        df_no_khanna["ln_size_resid"] = (
            df_no_khanna["ln_size"] - (coeffs_nk[0] * df_no_khanna["ln_mktcap"] + coeffs_nk[1])
        )

    # Large trades (>$15K)
    df_large = dfw[dfw["size_midpoint"] > 15000].copy()
    log(f"Large trades (>$15K): {len(df_large):,} trades")

    # Recompute legs_with_enough for large trades
    lg_leg_counts = df_large["politician_name"].value_counts()
    lg_legs_enough = set(lg_leg_counts[lg_leg_counts >= MIN_TRADES_LEGISLATOR_FE].index)

    # Recompute residualized size for large trades
    mask_lg = df_large["ln_size"].notna() & df_large["ln_mktcap"].notna()
    if mask_lg.sum() > 100:
        coeffs_lg = np.polyfit(df_large.loc[mask_lg, "ln_mktcap"].values,
                               df_large.loc[mask_lg, "ln_size"].values, 1)
        df_large["ln_size_resid"] = (
            df_large["ln_size"] - (coeffs_lg[0] * df_large["ln_mktcap"] + coeffs_lg[1])
        )

    # ==================================================================
    # TABLE A: WINSORIZED — Specs 1-5, 20d, market-adjusted
    # ==================================================================
    log("\n" + "=" * 70)
    log("TABLE A: WINSORIZED REGRESSIONS (Specs 1-5, 20d)")
    log("=" * 70)

    dep_col = f"AR_signed_market_{PRIMARY_OMEGA}d"
    win_results = {}
    for spec in [1, 2, 3, 4, 5]:
        log(f"  Spec {spec}...")
        res = run_spec(dfw, dep_col, spec, legs_with_enough)
        if res:
            win_results[spec] = res
            log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}")

    table_a = format_table(win_results, [f"Spec {s}" for s in [1,2,3,4,5]], [1,2,3,4,5])
    table_a.to_csv(os.path.join(OUTPUT_DIR, "regression_table_winsorized.csv"), index=False)
    log(f"\n{table_a.to_string(index=False)}")

    # ==================================================================
    # TABLE B: NO RO KHANNA — Specs 1-5, 20d, winsorized
    # ==================================================================
    log("\n" + "=" * 70)
    log("TABLE B: EXCLUDING RO KHANNA (Specs 1-5, 20d, winsorized)")
    log("=" * 70)

    nk_results = {}
    for spec in [1, 2, 3, 4, 5]:
        log(f"  Spec {spec}...")
        res = run_spec(df_no_khanna, dep_col, spec, nk_legs_enough)
        if res:
            nk_results[spec] = res
            log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}")

    table_b = format_table(nk_results, [f"Spec {s}" for s in [1,2,3,4,5]], [1,2,3,4,5])
    table_b.to_csv(os.path.join(OUTPUT_DIR, "regression_table_no_khanna.csv"), index=False)
    log(f"\n{table_b.to_string(index=False)}")

    # ==================================================================
    # TABLE C: LARGE TRADES — Specs 1-5, 20d, winsorized
    # ==================================================================
    log("\n" + "=" * 70)
    log("TABLE C: LARGE TRADES >$15K (Specs 1-5, 20d, winsorized)")
    log("=" * 70)

    lg_results = {}
    for spec in [1, 2, 3, 4, 5]:
        log(f"  Spec {spec}...")
        res = run_spec(df_large, dep_col, spec, lg_legs_enough)
        if res:
            lg_results[spec] = res
            log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}")

    table_c = format_table(lg_results, [f"Spec {s}" for s in [1,2,3,4,5]], [1,2,3,4,5])
    table_c.to_csv(os.path.join(OUTPUT_DIR, "regression_table_large_trades.csv"), index=False)
    log(f"\n{table_c.to_string(index=False)}")

    # ==================================================================
    # TABLE D: COMBINED ROBUSTNESS — Spec 3 across all samples
    # ==================================================================
    log("\n" + "=" * 70)
    log("TABLE D: SPEC 3 ACROSS ALL SAMPLE VARIANTS (20d, market-adj)")
    log("=" * 70)

    combined = {}
    combined["baseline"] = run_spec(df, dep_col, 3, legs_with_enough)
    combined["winsorized"] = run_spec(dfw, dep_col, 3, legs_with_enough)
    combined["no_khanna"] = run_spec(df_no_khanna, dep_col, 3, nk_legs_enough)
    combined["large"] = run_spec(df_large, dep_col, 3, lg_legs_enough)

    labels = ["Baseline", "Winsorized", "Excl. Khanna", "Large (>$15K)"]
    keys = ["baseline", "winsorized", "no_khanna", "large"]

    table_d = format_table(combined, labels, keys)
    table_d.to_csv(os.path.join(OUTPUT_DIR, "regression_table_combined_robustness.csv"), index=False)
    log(f"\n{table_d.to_string(index=False)}")

    # ==================================================================
    # TABLE E: WINSORIZED — Spec 3 across all horizons
    # ==================================================================
    log("\n" + "=" * 70)
    log("TABLE E: WINSORIZED SPEC 3 ACROSS ALL HORIZONS")
    log("=" * 70)

    hz_results = {}
    for omega in HOLDING_PERIODS:
        dep = f"AR_signed_market_{omega}d"
        if dep in dfw.columns:
            log(f"  ω={omega}...")
            res = run_spec(dfw, dep, 3, legs_with_enough)
            if res:
                hz_results[omega] = res
                log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}")

    hz_labels = [f"{o}d" for o in HOLDING_PERIODS]
    table_e = format_table(hz_results, hz_labels, HOLDING_PERIODS)
    table_e.to_csv(os.path.join(OUTPUT_DIR, "regression_table_winsorized_horizons.csv"), index=False)
    log(f"\n{table_e.to_string(index=False)}")

    # ==================================================================
    # TABLE F: WINSORIZED — Spec 3 across all models (20d)
    # ==================================================================
    log("\n" + "=" * 70)
    log("TABLE F: WINSORIZED SPEC 3 ACROSS ALL MODELS (20d)")
    log("=" * 70)

    model_results = {}
    for model in MODELS:
        dep = f"AR_signed_{model}_{PRIMARY_OMEGA}d"
        if dep in dfw.columns:
            log(f"  {model}...")
            res = run_spec(dfw, dep, 3, legs_with_enough)
            if res:
                model_results[model] = res
                log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}")

    table_f = format_table(model_results, MODELS, MODELS)
    table_f.to_csv(os.path.join(OUTPUT_DIR, "regression_table_winsorized_models.csv"), index=False)
    log(f"\n{table_f.to_string(index=False)}")

    # ==================================================================
    # TABLE G: WINSORIZED SPLIT-SAMPLE (Spec 3, 20d)
    # ==================================================================
    log("\n" + "=" * 70)
    log("TABLE G: WINSORIZED SPLIT-SAMPLE (Spec 3, 20d)")
    log("=" * 70)

    split_results = {}

    # Buy subsample: DV = unsigned AR (Eq 4.36)
    df_buys = dfw[dfw["buy_indicator"] == 1].copy()
    dep_buy = f"AR_market_{PRIMARY_OMEGA}d"
    X_b, vn_b, cl1_b, cl2_b = build_split_spec(df_buys, 3)
    y_b = df_buys[dep_buy].values.astype(float)
    res_buy = run_ols_twoway(y_b, X_b, cl1_b, cl2_b, vn_b)
    if res_buy:
        split_results["Buys"] = res_buy
        log(f"  Buys: N={res_buy['n_obs']:,}, R²={res_buy['r2']:.4f}")

    # Sell subsample: DV = -AR (Eq 4.37)
    df_sells = dfw[dfw["buy_indicator"] == 0].copy()
    dep_sell = f"AR_market_{PRIMARY_OMEGA}d"
    df_sells["neg_ar"] = -df_sells[dep_sell]
    X_s, vn_s, cl1_s, cl2_s = build_split_spec(df_sells, 3)
    y_s = df_sells["neg_ar"].values.astype(float)
    res_sell = run_ols_twoway(y_s, X_s, cl1_s, cl2_s, vn_s)
    if res_sell:
        split_results["Sells"] = res_sell
        log(f"  Sells: N={res_sell['n_obs']:,}, R²={res_sell['r2']:.4f}")

    split_report = [v for v in REPORT_VARS if v not in ("Buy", "Buy×Senate", "Buy×Republican")]
    split_rows = []
    for var in split_report:
        coef_row = {"Variable": var}
        se_row = {"Variable": ""}
        for label in ["Buys", "Sells"]:
            if label in split_results:
                res = split_results[label]
                if var in res["var_names"]:
                    idx = res["var_names"].index(var)
                    c, s = format_coef(res["beta"][idx], res["se_twoway"][idx], res["p_values"][idx])
                    coef_row[label] = c
                    se_row[label] = s
                else:
                    coef_row[label] = ""
                    se_row[label] = ""
            else:
                coef_row[label] = ""
                se_row[label] = ""
        split_rows.append(coef_row)
        split_rows.append(se_row)
    for stat_name, stat_key in [("N", "n_obs"), ("R²", "r2")]:
        row = {"Variable": stat_name}
        for label in ["Buys", "Sells"]:
            if label in split_results:
                val = split_results[label][stat_key]
                row[label] = f"{val:,}" if stat_name == "N" else f"{val:.4f}"
        split_rows.append(row)

    table_g = pd.DataFrame(split_rows)
    table_g.to_csv(os.path.join(OUTPUT_DIR, "regression_table_winsorized_split.csv"), index=False)
    log(f"\n{table_g.to_string(index=False)}")

    # ==================================================================
    # COMPARISON SUMMARY
    # ==================================================================
    log(f"\n{'=' * 70}")
    log("KEY COEFFICIENT COMPARISON ACROSS SAMPLES (Spec 3, 20d)")
    log(f"{'=' * 70}")

    compare_vars = ["Buy", "Senate", "Buy×Senate", "Republican",
                     "Buy×Republican", "Volatility", "PowerCommittee"]

    header = f"{'Variable':<18} {'Baseline':>12} {'Winsorized':>12} {'No Khanna':>12} {'Large':>12}"
    log(f"\n{header}")
    log("-" * len(header))

    for var in compare_vars:
        vals = []
        for key in ["baseline", "winsorized", "no_khanna", "large"]:
            if key in combined and combined[key] is not None:
                res = combined[key]
                if var in res["var_names"]:
                    idx = res["var_names"].index(var)
                    c, _ = format_coef(res["beta"][idx], res["se_twoway"][idx], res["p_values"][idx])
                    vals.append(f"{c:>12}")
                else:
                    vals.append(f"{'':>12}")
            else:
                vals.append(f"{'':>12}")
        log(f"{var:<18} {'  '.join(vals)}")

    # ==================================================================
    # DONE
    # ==================================================================
    elapsed = time.time() - start_time
    log(f"\n{'=' * 70}")
    log(f"STEP 7B COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log(f"{'=' * 70}")

    outputs = [
        "regression_table_winsorized.csv",
        "regression_table_no_khanna.csv",
        "regression_table_large_trades.csv",
        "regression_table_combined_robustness.csv",
        "regression_table_winsorized_horizons.csv",
        "regression_table_winsorized_models.csv",
        "regression_table_winsorized_split.csv",
    ]
    log(f"\nOutput files:")
    for fname in outputs:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            log(f"  {fname}")

    with open(os.path.join(OUTPUT_DIR, "step7b_diagnostics.txt"), "w") as f:
        f.write("\n".join(LOG))
    log(f"Diagnostics saved to: step7b_diagnostics.txt")


if __name__ == "__main__":
    main()
