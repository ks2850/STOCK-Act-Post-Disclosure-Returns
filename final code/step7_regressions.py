"""
step7_regressions.py
====================
Step 7 of the JIW pipeline: Cross-sectional regressions (Specifications 1-5)
with two-way clustered standard errors.

Implements:
  - Specification 1: Core Predictors (Eq 4.26)
  - Specification 2: + Institutional Interactions (Eq 4.27)
  - Specification 3: + Firm-Level Controls (Eq 4.28)
  - Specification 4: + Sector & Month-Year FE (Eq 4.32)
  - Specification 5: + Legislator FE (Eq 4.34)
  - Split-Sample: Buy/Sell subsamples (Section 4.1.4)
  - Two-Way Clustering: Cameron-Gelbach-Miller (2011) (Eq 4.38)
  - VIF Diagnostics (Section 4.1.6)

Input:
  - analysis_sample.csv (from Step 4)

Output:
  - regression_table_main.csv       (Specs 1-5, 20d, market-adjusted)
  - regression_table_all_models.csv (Specs 1-5, 20d, all 5 risk models)
  - regression_table_all_horizons.csv (Spec 3, all 6 horizons, market-adjusted)
  - regression_table_split_sample.csv (Buy/Sell subsamples)
  - regression_vif.csv              (VIF diagnostics)
  - step7_diagnostics.txt

Usage:
    pip install statsmodels  (if not already installed)
    python step7_regressions.py
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =====================================================================
# CONFIGURATION
# =====================================================================
ANALYSIS_PATH = "analysis_sample.csv"
OUTPUT_DIR = "."

HOLDING_PERIODS = [5, 10, 20, 40, 60, 127]
MODELS = ["market", "CAPM", "FF3", "Carhart", "FF5"]
PRIMARY_OMEGA = 20
PRIMARY_MODEL = "market"

# Minimum trades per legislator for Spec 5 (Section 4.1.4)
MIN_TRADES_LEGISLATOR_FE = 5

LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


# =====================================================================
# TWO-WAY CLUSTERING: Cameron-Gelbach-Miller (2011) — Eq 4.38
# =====================================================================

def one_way_cluster_cov(X, resid, clusters):
    """
    Compute one-way cluster-robust covariance matrix.
    Implements Eq 4.39/4.40/4.41.
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    # Meat of the sandwich
    meat = np.zeros((k, k))
    for g in unique_clusters:
        mask = clusters == g
        Xg = X[mask]
        eg = resid[mask]
        score_g = Xg.T @ eg  # k x 1
        meat += np.outer(score_g, score_g)

    # Small sample correction: G/(G-1) * (n-1)/(n-k)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V = correction * XtX_inv @ meat @ XtX_inv

    return V


def twoway_cluster_cov(X, resid, cluster1, cluster2):
    """
    Cameron-Gelbach-Miller (2011) two-way clustered covariance (Eq 4.38).
    V_twoway = V_cluster1 + V_cluster2 - V_intersection
    """
    # Intersection cluster
    cluster12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(cluster1, cluster2)])

    V1 = one_way_cluster_cov(X, resid, cluster1)
    V2 = one_way_cluster_cov(X, resid, cluster2)
    V12 = one_way_cluster_cov(X, resid, cluster12)

    V = V1 + V2 - V12

    # Ensure positive semi-definite (can fail with small intersection cells)
    eigvals = np.linalg.eigvalsh(V)
    if np.any(eigvals < 0):
        # Fall back to max of V1 and V2 for the problematic elements
        V = V1 + V2 - V12
        # Zero out negative eigenvalues
        eigvals_full, eigvecs = np.linalg.eigh(V)
        eigvals_full = np.maximum(eigvals_full, 0)
        V = eigvecs @ np.diag(eigvals_full) @ eigvecs.T

    return V


def run_ols_twoway(y, X, cluster1, cluster2, var_names):
    """
    Run OLS with two-way clustered SEs.

    Returns dict with: coefficients, twoway_se, oneway_se (legislator only),
    t_stats, p_values, r2, adj_r2, n_obs, n_clusters1, n_clusters2.
    """
    # Drop missing
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y_c = y[valid]
    X_c = X[valid]
    cl1 = cluster1[valid]
    cl2 = cluster2[valid]

    n, k = X_c.shape

    if n < k + 10:
        return None

    # OLS
    beta = np.linalg.lstsq(X_c, y_c, rcond=None)[0]
    resid = y_c - X_c @ beta

    # R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_c - np.mean(y_c)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    # Two-way clustered SEs (Eq 4.38)
    V_two = twoway_cluster_cov(X_c, resid, cl1, cl2)
    se_two = np.sqrt(np.diag(V_two))

    # One-way legislator-clustered SEs (for comparison, as paper says)
    V_one = one_way_cluster_cov(X_c, resid, cl1)
    se_one = np.sqrt(np.diag(V_one))

    # T-stats and p-values (using two-way SEs)
    t_stats = beta / se_two
    from scipy.stats import t as t_dist
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
    """Format coefficient with stars for significance."""
    stars = ""
    if p < 0.01:
        stars = "***"
    elif p < 0.05:
        stars = "**"
    elif p < 0.10:
        stars = "*"
    return f"{beta:.5f}{stars}", f"({se:.5f})"


def compute_vif(X, var_names):
    """Compute VIF for each regressor (Eq 4.42). Excludes intercept."""
    vifs = {}
    # Find intercept column (all ones)
    intercept_idx = None
    for i in range(X.shape[1]):
        if np.all(X[:, i] == 1):
            intercept_idx = i
            break

    for i, name in enumerate(var_names):
        if i == intercept_idx:
            continue
        try:
            vif_val = variance_inflation_factor(X, i)
            vifs[name] = vif_val
        except:
            vifs[name] = np.nan

    return vifs


def main():
    start_time = time.time()
    log("=" * 70)
    log("STEP 7: CROSS-SECTIONAL REGRESSIONS")
    log("=" * 70)

    # ==================================================================
    # LOAD DATA
    # ==================================================================
    log("\n--- Loading data ---")
    df = pd.read_csv(ANALYSIS_PATH)
    log(f"Loaded {len(df):,} trades")

    # ==================================================================
    # PREPARE VARIABLES
    # ==================================================================
    log("\n--- Preparing variables ---")

    # Ensure numeric
    for col in ["buy_indicator", "senate_indicator", "republican_indicator",
                "power_committee", "disclosure_lag", "disclosure_lag_sq",
                "ln_size", "ln_mktcap", "volatility", "ln_size_resid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Interaction terms (Eqs 4.27-4.28)
    df["buy_x_senate"] = df["buy_indicator"] * df["senate_indicator"]
    df["buy_x_republican"] = df["buy_indicator"] * df["republican_indicator"]

    # Sector grouping from SIC codes (map to ~10 Fama-French sectors)
    def sic_to_sector(sic):
        """Map SIC code to broad sector for fixed effects."""
        try:
            s = int(str(sic)[:2])
        except (ValueError, TypeError):
            return "Other"
        if s <= 9:
            return "Agriculture"
        elif s <= 14:
            return "Mining"
        elif s <= 17:
            return "Construction"
        elif s <= 39:
            return "Manufacturing"
        elif s <= 49:
            return "Utilities/Transport"
        elif s <= 51:
            return "Wholesale"
        elif s <= 59:
            return "Retail"
        elif s <= 67:
            return "Finance"
        elif s <= 89:
            return "Services"
        else:
            return "Other"

    df["sector"] = df["siccd"].apply(sic_to_sector)

    # Cluster variables
    df["cluster_legislator"] = df["politician_name"].astype(str)
    df["cluster_month"] = df["disc_month_year"].astype(str)

    # Legislator FE: filter to legislators with >= MIN_TRADES trades
    leg_counts = df["politician_name"].value_counts()
    legs_with_enough = set(leg_counts[leg_counts >= MIN_TRADES_LEGISLATOR_FE].index)
    log(f"Legislators with >= {MIN_TRADES_LEGISLATOR_FE} trades: {len(legs_with_enough)} "
        f"({df['politician_name'].isin(legs_with_enough).sum():,} trades)")

    # ==================================================================
    # DEFINE SPECIFICATIONS (Section 4.1.4)
    # ==================================================================

    def build_spec_matrices(df_sub, spec_num):
        """
        Build y, X, variable names, cluster arrays for a given specification.
        Returns None if insufficient data.

        Specification mapping:
          1: Eq 4.26 — core predictors, RAW ln_size
          2: Eq 4.27 — + interactions
          3: Eq 4.28 — + firm controls, RESIDUALIZED ln_size
          4: Eq 4.32 — + sector FE + month-year FE
          5: Eq 4.34 — + legislator FE, drops Senate/Republican main effects
        """
        d = df_sub.copy()

        # Base variables present in all specs
        base_vars = {
            "Buy": d["buy_indicator"].values,
            "Senate": d["senate_indicator"].values,
            "Republican": d["republican_indicator"].values,
            "Disc_Lag": d["disclosure_lag"].values,
            "Disc_Lag_Sq": d["disclosure_lag_sq"].values,
        }

        if spec_num <= 2:
            # Specs 1-2: Use RAW ln(Size) — Section 4.1.4
            base_vars["ln_Size"] = d["ln_size"].values
        else:
            # Specs 3-5: Use RESIDUALIZED ln(Size)* — Eq 4.31
            base_vars["ln_Size_resid"] = d["ln_size_resid"].values

        var_names = ["Intercept"] + list(base_vars.keys())
        X_list = [np.ones(len(d))] + list(base_vars.values())

        # Spec 2+: Add interactions
        if spec_num >= 2:
            X_list.append(d["buy_x_senate"].values)
            var_names.append("Buy×Senate")
            X_list.append(d["buy_x_republican"].values)
            var_names.append("Buy×Republican")

        # Spec 3+: Add firm-level controls
        if spec_num >= 3:
            X_list.append(d["ln_mktcap"].values)
            var_names.append("ln_MktCap")
            X_list.append(d["volatility"].values)
            var_names.append("Volatility")
            X_list.append(d["power_committee"].values.astype(float))
            var_names.append("PowerCommittee")

        # Spec 4+: Add sector and month-year FE
        if spec_num >= 4:
            # Sector dummies (drop first for identification)
            sector_dummies = pd.get_dummies(d["sector"], prefix="Sec", drop_first=True)
            for col in sector_dummies.columns:
                X_list.append(sector_dummies[col].values.astype(float))
                var_names.append(col)

            # Month-year dummies (drop first)
            my_dummies = pd.get_dummies(d["disc_month_year"], prefix="MY", drop_first=True)
            for col in my_dummies.columns:
                X_list.append(my_dummies[col].values.astype(float))
                var_names.append(col)

        # Spec 5: Add legislator FE, drop Senate/Republican main effects
        if spec_num == 5:
            # Filter to legislators with enough trades
            d_mask = d["politician_name"].isin(legs_with_enough)
            d = d[d_mask].copy()

            if len(d) < 100:
                return None

            # Rebuild everything for filtered sample
            return build_spec5(d)

        X = np.column_stack(X_list)
        cl1 = d["cluster_legislator"].values
        cl2 = d["cluster_month"].values

        return X, var_names, cl1, cl2

    def build_spec5(d):
        """
        Specification 5 (Eq 4.34): Legislator FE.
        Drops Senate/Republican main effects (absorbed by μ_j).
        Keeps Buy×Senate, Buy×Republican (not absorbed — Eq 4.35).
        """
        var_names = ["Intercept"]
        X_list = [np.ones(len(d))]

        # Variables that survive legislator FE (Section 4.1.4)
        X_list.append(d["buy_indicator"].values)
        var_names.append("Buy")

        # Note: Senate and Republican DROPPED (absorbed by legislator FE)

        X_list.append(d["ln_size_resid"].values)
        var_names.append("ln_Size_resid")

        X_list.append(d["disclosure_lag"].values)
        var_names.append("Disc_Lag")

        X_list.append(d["disclosure_lag_sq"].values)
        var_names.append("Disc_Lag_Sq")

        # Interactions SURVIVE (Eq 4.35)
        X_list.append(d["buy_x_senate"].values)
        var_names.append("Buy×Senate")

        X_list.append(d["buy_x_republican"].values)
        var_names.append("Buy×Republican")

        # Firm controls
        X_list.append(d["ln_mktcap"].values)
        var_names.append("ln_MktCap")

        X_list.append(d["volatility"].values)
        var_names.append("Volatility")

        X_list.append(d["power_committee"].values.astype(float))
        var_names.append("PowerCommittee")

        # Sector FE
        sector_dummies = pd.get_dummies(d["sector"], prefix="Sec", drop_first=True)
        for col in sector_dummies.columns:
            X_list.append(sector_dummies[col].values.astype(float))
            var_names.append(col)

        # Month-year FE
        my_dummies = pd.get_dummies(d["disc_month_year"], prefix="MY", drop_first=True)
        for col in my_dummies.columns:
            X_list.append(my_dummies[col].values.astype(float))
            var_names.append(col)

        # Legislator FE (drop first for identification)
        leg_dummies = pd.get_dummies(d["politician_name"], prefix="Leg", drop_first=True)
        for col in leg_dummies.columns:
            X_list.append(leg_dummies[col].values.astype(float))
            var_names.append(col)

        X = np.column_stack(X_list)
        cl1 = d["cluster_legislator"].values
        cl2 = d["cluster_month"].values

        return X, var_names, cl1, cl2

    def build_split_sample_matrices(df_sub, spec_num, is_buy_sample):
        """
        Build matrices for split-sample regressions (Section 4.1.4 end).
        Drops Buy indicator and its interactions.
        """
        d = df_sub.copy()

        var_names = ["Intercept"]
        X_list = [np.ones(len(d))]

        # No Buy indicator in split sample
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

        # No Buy×Senate or Buy×Republican in split sample

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

    # ==================================================================
    # RUN REGRESSIONS
    # ==================================================================

    # Key covariates to report (exclude FE dummies)
    REPORT_VARS = [
        "Intercept", "Buy", "Senate", "Republican",
        "ln_Size", "ln_Size_resid",
        "Disc_Lag", "Disc_Lag_Sq",
        "Buy×Senate", "Buy×Republican",
        "ln_MktCap", "Volatility", "PowerCommittee",
    ]

    def run_and_store(df_sub, dep_var_col, spec_num, label, is_split=False, is_buy=True):
        """Run one regression and return formatted results."""
        y = df_sub[dep_var_col].values.astype(float)

        if is_split:
            result = build_split_sample_matrices(df_sub, spec_num, is_buy)
        elif spec_num == 5:
            # Spec 5 needs filtered data
            d_filt = df_sub[df_sub["politician_name"].isin(legs_with_enough)].copy()
            if len(d_filt) < 100:
                return None
            y = d_filt[dep_var_col].values.astype(float)
            result = build_spec5(d_filt)
        else:
            result = build_spec_matrices(df_sub, spec_num)

        if result is None:
            return None

        X, var_names, cl1, cl2 = result

        res = run_ols_twoway(y, X, cl1, cl2, var_names)
        if res is None:
            return None

        res["spec"] = spec_num
        res["label"] = label
        res["dep_var"] = dep_var_col
        return res

    # ==================================================================
    # TABLE 1: MAIN RESULTS — Specs 1-5, 20d, Market-Adjusted (Pooled)
    # ==================================================================
    log("\n" + "=" * 70)
    log("TABLE 1: MAIN REGRESSION RESULTS")
    log(f"Dependent Variable: AR_signed_market_{PRIMARY_OMEGA}d")
    log("=" * 70)

    main_results = {}
    for spec in [1, 2, 3, 4, 5]:
        log(f"\n  Running Specification {spec}...")
        dep_col = f"AR_signed_market_{PRIMARY_OMEGA}d"
        res = run_and_store(df, dep_col, spec, f"Spec{spec}")
        if res is not None:
            main_results[spec] = res
            log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}, "
                f"Adj R²={res['adj_r2']:.4f}, "
                f"Legislators={res['n_legislators']}, "
                f"Months={res['n_months']}")

    # Format main results table
    table_rows = []
    for var in REPORT_VARS:
        coef_row = {"Variable": var}
        se_row = {"Variable": ""}
        for spec in [1, 2, 3, 4, 5]:
            col = f"Spec {spec}"
            if spec in main_results:
                res = main_results[spec]
                if var in res["var_names"]:
                    idx = res["var_names"].index(var)
                    coef_str, se_str = format_coef(
                        res["beta"][idx], res["se_twoway"][idx], res["p_values"][idx])
                    coef_row[col] = coef_str
                    se_row[col] = se_str
                else:
                    coef_row[col] = ""
                    se_row[col] = ""
            else:
                coef_row[col] = ""
                se_row[col] = ""
        table_rows.append(coef_row)
        table_rows.append(se_row)

    # Add summary stats rows
    for stat_name, stat_key in [("N", "n_obs"), ("R²", "r2"),
                                 ("Adj R²", "adj_r2"),
                                 ("Legislators", "n_legislators"),
                                 ("Months", "n_months")]:
        row = {"Variable": stat_name}
        for spec in [1, 2, 3, 4, 5]:
            col = f"Spec {spec}"
            if spec in main_results:
                val = main_results[spec][stat_key]
                if stat_name in ("N", "Legislators", "Months"):
                    row[col] = f"{val:,}"
                else:
                    row[col] = f"{val:.4f}"
            else:
                row[col] = ""
        table_rows.append(row)

    # FE indicators
    for fe_name, specs_with in [("Sector FE", [4, 5]), ("Month-Year FE", [4, 5]),
                                  ("Legislator FE", [5])]:
        row = {"Variable": fe_name}
        for spec in [1, 2, 3, 4, 5]:
            row[f"Spec {spec}"] = "Yes" if spec in specs_with else "No"
        table_rows.append(row)

    main_table = pd.DataFrame(table_rows)
    main_table.to_csv(os.path.join(OUTPUT_DIR, "regression_table_main.csv"), index=False)
    log(f"\nSaved regression_table_main.csv")

    # Print the table
    log(f"\n{main_table.to_string(index=False)}")

    # ==================================================================
    # TABLE 2: ALL MODELS — Spec 3, 20d, all 5 risk models
    # ==================================================================
    log("\n" + "=" * 70)
    log("TABLE 2: SPECIFICATION 3 ACROSS ALL RISK MODELS (20d)")
    log("=" * 70)

    model_results = {}
    for model in MODELS:
        dep_col = f"AR_signed_{model}_{PRIMARY_OMEGA}d"
        if dep_col not in df.columns:
            continue
        log(f"\n  Running {model}...")
        res = run_and_store(df, dep_col, 3, f"Spec3_{model}")
        if res is not None:
            model_results[model] = res
            log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}")

    model_table_rows = []
    for var in REPORT_VARS:
        coef_row = {"Variable": var}
        se_row = {"Variable": ""}
        for model in MODELS:
            if model in model_results:
                res = model_results[model]
                if var in res["var_names"]:
                    idx = res["var_names"].index(var)
                    c, s = format_coef(res["beta"][idx], res["se_twoway"][idx], res["p_values"][idx])
                    coef_row[model] = c
                    se_row[model] = s
                else:
                    coef_row[model] = ""
                    se_row[model] = ""
            else:
                coef_row[model] = ""
                se_row[model] = ""
        model_table_rows.append(coef_row)
        model_table_rows.append(se_row)

    for stat_name, stat_key in [("N", "n_obs"), ("R²", "r2")]:
        row = {"Variable": stat_name}
        for model in MODELS:
            if model in model_results:
                val = model_results[model][stat_key]
                row[model] = f"{val:,}" if stat_name == "N" else f"{val:.4f}"
            else:
                row[model] = ""
        model_table_rows.append(row)

    model_table = pd.DataFrame(model_table_rows)
    model_table.to_csv(os.path.join(OUTPUT_DIR, "regression_table_all_models.csv"), index=False)
    log(f"\nSaved regression_table_all_models.csv")
    log(f"\n{model_table.to_string(index=False)}")

    # ==================================================================
    # TABLE 3: ALL HORIZONS — Spec 3, market-adjusted, all 6 horizons
    # ==================================================================
    log("\n" + "=" * 70)
    log("TABLE 3: SPECIFICATION 3 ACROSS ALL HOLDING PERIODS (Market-Adj)")
    log("=" * 70)

    horizon_results = {}
    for omega in HOLDING_PERIODS:
        dep_col = f"AR_signed_market_{omega}d"
        if dep_col not in df.columns:
            continue
        log(f"\n  Running ω={omega}...")
        res = run_and_store(df, dep_col, 3, f"Spec3_{omega}d")
        if res is not None:
            horizon_results[omega] = res
            log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}")

    hz_table_rows = []
    for var in REPORT_VARS:
        coef_row = {"Variable": var}
        se_row = {"Variable": ""}
        for omega in HOLDING_PERIODS:
            col = f"{omega}d"
            if omega in horizon_results:
                res = horizon_results[omega]
                if var in res["var_names"]:
                    idx = res["var_names"].index(var)
                    c, s = format_coef(res["beta"][idx], res["se_twoway"][idx], res["p_values"][idx])
                    coef_row[col] = c
                    se_row[col] = s
                else:
                    coef_row[col] = ""
                    se_row[col] = ""
            else:
                coef_row[col] = ""
                se_row[col] = ""
        hz_table_rows.append(coef_row)
        hz_table_rows.append(se_row)

    for stat_name, stat_key in [("N", "n_obs"), ("R²", "r2")]:
        row = {"Variable": stat_name}
        for omega in HOLDING_PERIODS:
            col = f"{omega}d"
            if omega in horizon_results:
                val = horizon_results[omega][stat_key]
                row[col] = f"{val:,}" if stat_name == "N" else f"{val:.4f}"
            else:
                row[col] = ""
        hz_table_rows.append(row)

    hz_table = pd.DataFrame(hz_table_rows)
    hz_table.to_csv(os.path.join(OUTPUT_DIR, "regression_table_all_horizons.csv"), index=False)
    log(f"\nSaved regression_table_all_horizons.csv")
    log(f"\n{hz_table.to_string(index=False)}")

    # ==================================================================
    # TABLE 4: SPLIT-SAMPLE — Buy and Sell subsamples
    # ==================================================================
    log("\n" + "=" * 70)
    log("TABLE 4: SPLIT-SAMPLE REGRESSIONS (Spec 3, 20d, Market-Adj)")
    log("=" * 70)

    split_results = {}

    # Buy subsample: DV = unsigned AR (Eq 4.36)
    df_buys = df[df["buy_indicator"] == 1].copy()
    dep_buy = f"AR_market_{PRIMARY_OMEGA}d"
    log(f"\n  Buy subsample: N={len(df_buys):,}, DV={dep_buy}")
    res_buy = run_and_store(df_buys, dep_buy, 3, "Buys", is_split=True, is_buy=True)
    if res_buy:
        split_results["Buys"] = res_buy
        log(f"    N={res_buy['n_obs']:,}, R²={res_buy['r2']:.4f}")

    # Sell subsample: DV = negated AR (Eq 4.37)
    df_sells = df[df["buy_indicator"] == 0].copy()
    # Negate: positive means legislator avoided losses
    neg_col = f"neg_AR_market_{PRIMARY_OMEGA}d"
    df_sells[neg_col] = -df_sells[f"AR_market_{PRIMARY_OMEGA}d"]
    log(f"\n  Sell subsample: N={len(df_sells):,}, DV=-AR_market_{PRIMARY_OMEGA}d")
    res_sell = run_and_store(df_sells, neg_col, 3, "Sells", is_split=True, is_buy=False)
    if res_sell:
        split_results["Sells"] = res_sell
        log(f"    N={res_sell['n_obs']:,}, R²={res_sell['r2']:.4f}")

    # Format split-sample table
    split_report_vars = [v for v in REPORT_VARS if v not in ("Buy", "Buy×Senate", "Buy×Republican")]
    split_table_rows = []
    for var in split_report_vars:
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
        split_table_rows.append(coef_row)
        split_table_rows.append(se_row)

    for stat_name, stat_key in [("N", "n_obs"), ("R²", "r2")]:
        row = {"Variable": stat_name}
        for label in ["Buys", "Sells"]:
            if label in split_results:
                val = split_results[label][stat_key]
                row[label] = f"{val:,}" if stat_name == "N" else f"{val:.4f}"
            else:
                row[label] = ""
        split_table_rows.append(row)

    split_table = pd.DataFrame(split_table_rows)
    split_table.to_csv(os.path.join(OUTPUT_DIR, "regression_table_split_sample.csv"), index=False)
    log(f"\nSaved regression_table_split_sample.csv")
    log(f"\n{split_table.to_string(index=False)}")

    # ==================================================================
    # VIF DIAGNOSTICS (Section 4.1.6)
    # ==================================================================
    log("\n" + "=" * 70)
    log("VIF DIAGNOSTICS (Section 4.1.6)")
    log("=" * 70)

    vif_results = []
    for spec in [1, 2, 3, 4]:  # Skip 5 (too many dummies)
        dep_col = f"AR_signed_market_{PRIMARY_OMEGA}d"
        if spec == 5:
            continue

        # Build X matrix
        if spec <= 3:
            result = build_spec_matrices(df, spec)
        else:
            result = build_spec_matrices(df, spec)

        if result is None:
            continue

        X, var_names, _, _ = result

        # Drop rows with NaN
        valid = np.all(np.isfinite(X), axis=1)
        X_clean = X[valid]

        vifs = compute_vif(X_clean, var_names)
        for var, vif_val in vifs.items():
            if var.startswith("Sec_") or var.startswith("MY_"):
                continue  # skip FE dummies
            vif_results.append({
                "Specification": spec,
                "Variable": var,
                "VIF": vif_val,
            })

    vif_df = pd.DataFrame(vif_results)
    vif_df.to_csv(os.path.join(OUTPUT_DIR, "regression_vif.csv"), index=False)
    log(f"\nSaved regression_vif.csv")

    # Print key VIFs
    for spec in [1, 2, 3, 4]:
        spec_vifs = vif_df[vif_df["Specification"] == spec]
        if len(spec_vifs) > 0:
            log(f"\n  Spec {spec}:")
            for _, row in spec_vifs.iterrows():
                flag = " ⚠ HIGH" if row["VIF"] > 5 else ""
                log(f"    {row['Variable']:<20} VIF={row['VIF']:>8.2f}{flag}")

    # ==================================================================
    # DONE
    # ==================================================================
    elapsed = time.time() - start_time
    log(f"\n{'=' * 70}")
    log(f"STEP 7 COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log(f"{'=' * 70}")

    log(f"\nOutput files:")
    for fname in ["regression_table_main.csv", "regression_table_all_models.csv",
                   "regression_table_all_horizons.csv", "regression_table_split_sample.csv",
                   "regression_vif.csv"]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            log(f"  {fname}")

    with open(os.path.join(OUTPUT_DIR, "step7_diagnostics.txt"), "w") as f:
        f.write("\n".join(LOG))
    log(f"Diagnostics saved to: step7_diagnostics.txt")


if __name__ == "__main__":
    main()
