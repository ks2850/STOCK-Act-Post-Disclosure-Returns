"""
step7c_event_level_regressions.py (UPDATED)
=============================================
Runs Specifications 1-5 on the disclosure-event-level dataset.

UPDATES from prior version:
  1. BuyFraction (continuous [0,1]) replaces binary Buy indicator
     — 45% of events are mixed buy/sell; continuous variable preserves info
  2. Interactions: BuyFrac×Senate, BuyFrac×Republican
  3. Sector FE dropped in Spec 4 (events span multiple sectors; mode is
     a poor characterization of diversified filings). Month-year FE retained.
  4. WLS variant with sqrt(n_trades) weights added as robustness
     — accounts for mechanical variance reduction in large events

Input:
  - analysis_sample_event_level.csv (from step3b)

Output:
  - regression_event_main.csv          (Specs 1-5, 20d)
  - regression_event_no_khanna.csv     (Excl. Khanna)
  - regression_event_large.csv         (Large events)
  - regression_event_horizons.csv      (Spec 3, all horizons)
  - regression_event_comparison.csv    (Spec 3 side-by-side)
  - regression_event_wls.csv           (WLS robustness)
  - step7c_diagnostics.txt

Usage:
    python step7c_event_level_regressions.py
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import t as t_dist

# =====================================================================
# CONFIGURATION
# =====================================================================
EVENT_PATH = "analysis_sample_event_level.csv"
OUTPUT_DIR = "."

HOLDING_PERIODS = [5, 10, 20, 40, 60, 127]
MODELS = ["market", "CAPM", "FF3", "Carhart", "FF5"]
PRIMARY_OMEGA = 20

MIN_TRADES_LEGISLATOR_FE = 3

LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


# =====================================================================
# CLUSTERING
# =====================================================================

def one_way_cluster_cov(X, resid, clusters):
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    meat = np.zeros((k, k))
    for g in unique_clusters:
        mask = clusters == g
        score_g = X[mask].T @ resid[mask]
        meat += np.outer(score_g, score_g)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    return correction * XtX_inv @ meat @ XtX_inv


def twoway_cluster_cov(X, resid, cluster1, cluster2):
    cluster12 = np.array([f"{c1}_{c2}" for c1, c2 in zip(cluster1, cluster2)])
    V1 = one_way_cluster_cov(X, resid, cluster1)
    V2 = one_way_cluster_cov(X, resid, cluster2)
    V12 = one_way_cluster_cov(X, resid, cluster12)
    V = V1 + V2 - V12
    eigvals, eigvecs = np.linalg.eigh(V)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def run_ols_twoway(y, X, cluster1, cluster2, var_names, weights=None):
    """
    OLS with two-way clustered SEs.
    If weights is provided, runs WLS: transform y and X by sqrt(weights).
    """
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if weights is not None:
        valid = valid & np.isfinite(weights) & (weights > 0)

    y_c, X_c = y[valid], X[valid]
    cl1, cl2 = cluster1[valid], cluster2[valid]

    n, k = X_c.shape
    if n < k + 10:
        return None

    if weights is not None:
        w = np.sqrt(weights[valid])
        y_c = y_c * w
        X_c = X_c * w[:, np.newaxis]

    beta = np.linalg.lstsq(X_c, y_c, rcond=None)[0]
    resid = y_c - X_c @ beta

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_c - np.mean(y_c)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    V_two = twoway_cluster_cov(X_c, resid, cl1, cl2)
    se_two = np.sqrt(np.diag(V_two))

    t_stats = beta / se_two
    df = max(min(len(np.unique(cl1)), len(np.unique(cl2))) - 1, 1)
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=df))

    return {
        "var_names": var_names, "beta": beta,
        "se_twoway": se_two, "t_stats": t_stats, "p_values": p_values,
        "r2": r2, "adj_r2": adj_r2, "n_obs": n,
        "n_legislators": len(np.unique(cl1)),
        "n_months": len(np.unique(cl2)),
    }


def format_coef(beta, se, p):
    stars = ""
    if p < 0.01: stars = "***"
    elif p < 0.05: stars = "**"
    elif p < 0.10: stars = "*"
    return f"{beta:.5f}{stars}", f"({se:.5f})"


REPORT_VARS = [
    "Intercept", "BuyFrac", "Senate", "Republican",
    "ln_Volume", "ln_Volume_resid",
    "Disc_Lag", "Disc_Lag_Sq",
    "BuyFrac×Senate", "BuyFrac×Republican",
    "ln_MktCap", "Volatility", "PowerCommittee",
    "n_trades",
]


# =====================================================================
# SPECIFICATION BUILDER
# =====================================================================

def build_spec(d, spec_num, legs_enough):
    """
    Build X matrix for event-level regression.
    Uses BuyFraction (continuous) instead of binary Buy indicator.
    Spec 4: Month-year FE only (no sector FE — events are multi-sector).
    Spec 5: + Legislator FE, drops Senate/Republican.
    """
    if spec_num == 5:
        d = d[d["politician_name"].isin(legs_enough)].copy()
        if len(d) < 50:
            return None, None, None, None, None

    var_names = ["Intercept"]
    X_list = [np.ones(len(d))]

    # BuyFraction: continuous [0, 1] — replaces binary Buy
    X_list.append(d["buy_fraction"].values.astype(float))
    var_names.append("BuyFrac")

    if spec_num <= 4:
        X_list.append(d["senate_indicator"].values.astype(float))
        var_names.append("Senate")
        X_list.append(d["republican_indicator"].values.astype(float))
        var_names.append("Republican")

    if spec_num <= 2:
        X_list.append(d["ln_total_volume"].values.astype(float))
        var_names.append("ln_Volume")
    else:
        X_list.append(d["ln_volume_resid"].values.astype(float))
        var_names.append("ln_Volume_resid")

    X_list.append(d["disclosure_lag"].values.astype(float))
    var_names.append("Disc_Lag")
    X_list.append(d["disclosure_lag_sq"].values.astype(float))
    var_names.append("Disc_Lag_Sq")

    if spec_num >= 2:
        # Continuous interactions: BuyFrac × Senate, BuyFrac × Republican
        X_list.append((d["buy_fraction"] * d["senate_indicator"]).values.astype(float))
        var_names.append("BuyFrac×Senate")
        X_list.append((d["buy_fraction"] * d["republican_indicator"]).values.astype(float))
        var_names.append("BuyFrac×Republican")

    if spec_num >= 3:
        X_list.append(d["ln_mktcap"].values.astype(float))
        var_names.append("ln_MktCap")
        X_list.append(d["volatility"].values.astype(float))
        var_names.append("Volatility")
        X_list.append(d["power_committee"].values.astype(float))
        var_names.append("PowerCommittee")
        X_list.append(d["n_trades"].values.astype(float))
        var_names.append("n_trades")

    if spec_num >= 4:
        # Month-year FE only (no sector FE — events span multiple sectors)
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


def run_spec(df_sub, dep_col, spec_num, legs_enough, weights=None):
    """Run one specification. Optional WLS via weights array."""
    X, var_names, cl1, cl2, d_used = build_spec(df_sub, spec_num, legs_enough)
    if X is None:
        return None
    y = d_used[dep_col].values.astype(float)
    w = d_used[weights].values.astype(float) if weights and weights in d_used.columns else None
    return run_ols_twoway(y, X, cl1, cl2, var_names, weights=w)


def format_table(results_dict, col_labels, keys):
    rows = []
    for var in REPORT_VARS:
        cr, sr = {"Variable": var}, {"Variable": ""}
        for key, label in zip(keys, col_labels):
            if key in results_dict and results_dict[key]:
                res = results_dict[key]
                if var in res["var_names"]:
                    idx = res["var_names"].index(var)
                    c, s = format_coef(res["beta"][idx], res["se_twoway"][idx], res["p_values"][idx])
                    cr[label], sr[label] = c, s
                else:
                    cr[label], sr[label] = "", ""
            else:
                cr[label], sr[label] = "", ""
        rows.extend([cr, sr])

    for stat, key_name in [("N", "n_obs"), ("R²", "r2"), ("Adj R²", "adj_r2"),
                            ("Legislators", "n_legislators"), ("Months", "n_months")]:
        row = {"Variable": stat}
        for key, label in zip(keys, col_labels):
            if key in results_dict and results_dict[key]:
                val = results_dict[key][key_name]
                row[label] = f"{val:,}" if stat in ("N", "Legislators", "Months") else f"{val:.4f}"
            else:
                row[label] = ""
        rows.append(row)

    for fe, specs_with in [("Month-Year FE", [4, 5]), ("Legislator FE", [5])]:
        row = {"Variable": fe}
        for key, label in zip(keys, col_labels):
            row[label] = "Yes" if isinstance(key, int) and key in specs_with else "No" if isinstance(key, int) else ""
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    start_time = time.time()
    log("=" * 70)
    log("STEP 7C: EVENT-LEVEL REGRESSIONS (UPDATED)")
    log("  - BuyFraction (continuous) replaces binary Buy")
    log("  - Sector FE dropped (events are multi-sector portfolios)")
    log("  - WLS robustness with sqrt(n_trades) weights")
    log("=" * 70)

    # ==================================================================
    # LOAD
    # ==================================================================
    log("\n--- Loading event-level data ---")
    df = pd.read_csv(EVENT_PATH)
    log(f"Events loaded: {len(df):,}")

    for col in ["buy_fraction", "buy_indicator", "senate_indicator",
                "republican_indicator", "power_committee", "disclosure_lag",
                "disclosure_lag_sq", "ln_total_volume", "ln_mktcap",
                "volatility", "ln_volume_resid", "n_trades", "total_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["cluster_legislator"] = df["politician_name"].astype(str)
    df["cluster_month"] = df["disc_month_year"].astype(str)

    # WLS weights: n_trades (run_ols_twoway takes sqrt internally)
    # Rationale: Var(AR_event) ≈ σ²/n_trades, so GLS weight = n_trades
    df["wls_weight"] = df["n_trades"]

    leg_counts = df["politician_name"].value_counts()
    legs_enough = set(leg_counts[leg_counts >= MIN_TRADES_LEGISLATOR_FE].index)
    log(f"Legislators with >= {MIN_TRADES_LEGISLATOR_FE} events: {len(legs_enough)}")

    log(f"\nBuy fraction distribution:")
    log(f"  Pure buy (=1.0):  {(df['buy_fraction'] == 1.0).sum()}")
    log(f"  Pure sell (=0.0): {(df['buy_fraction'] == 0.0).sum()}")
    log(f"  Mixed:            {((df['buy_fraction'] > 0) & (df['buy_fraction'] < 1)).sum()}")
    log(f"  Mean: {df['buy_fraction'].mean():.3f}")

    dep = f"AR_signed_market_{PRIMARY_OMEGA}d"

    # ==================================================================
    # TABLE 1: MAIN — Specs 1-5 (OLS)
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"TABLE 1: EVENT-LEVEL MAIN RESULTS (20d, BuyFrac, OLS)")
    log(f"{'='*70}")

    main_res = {}
    for spec in [1, 2, 3, 4, 5]:
        log(f"  Spec {spec}...")
        res = run_spec(df, dep, spec, legs_enough)
        if res:
            main_res[spec] = res
            log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}")

    t1 = format_table(main_res, [f"Spec {s}" for s in [1,2,3,4,5]], [1,2,3,4,5])
    t1.to_csv(os.path.join(OUTPUT_DIR, "regression_event_main.csv"), index=False)
    log(f"\n{t1.to_string(index=False)}")

    # ==================================================================
    # TABLE 2: NO RO KHANNA
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"TABLE 2: EXCL. RO KHANNA (20d, BuyFrac)")
    log(f"{'='*70}")

    df_nk = df[df["politician_name"] != "Ro Khanna"].copy()
    nk_counts = df_nk["politician_name"].value_counts()
    nk_legs = set(nk_counts[nk_counts >= MIN_TRADES_LEGISLATOR_FE].index)

    mask_nk = df_nk["ln_total_volume"].notna() & df_nk["ln_mktcap"].notna()
    if mask_nk.sum() > 50:
        c_nk = np.polyfit(df_nk.loc[mask_nk, "ln_mktcap"].values,
                           df_nk.loc[mask_nk, "ln_total_volume"].values, 1)
        df_nk["ln_volume_resid"] = df_nk["ln_total_volume"] - (c_nk[0] * df_nk["ln_mktcap"] + c_nk[1])

    log(f"  Events: {len(df_nk):,} (dropped {len(df)-len(df_nk):,})")

    nk_res = {}
    for spec in [1, 2, 3, 4, 5]:
        log(f"  Spec {spec}...")
        res = run_spec(df_nk, dep, spec, nk_legs)
        if res:
            nk_res[spec] = res
            log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}")

    t2 = format_table(nk_res, [f"Spec {s}" for s in [1,2,3,4,5]], [1,2,3,4,5])
    t2.to_csv(os.path.join(OUTPUT_DIR, "regression_event_no_khanna.csv"), index=False)
    log(f"\n{t2.to_string(index=False)}")

    # ==================================================================
    # TABLE 3: LARGE EVENTS (total_volume > $30K)
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"TABLE 3: LARGE EVENTS >$30K (20d, BuyFrac)")
    log(f"{'='*70}")

    df_lg = df[df["total_volume"] > 30000].copy()
    lg_counts = df_lg["politician_name"].value_counts()
    lg_legs = set(lg_counts[lg_counts >= MIN_TRADES_LEGISLATOR_FE].index)

    mask_lg = df_lg["ln_total_volume"].notna() & df_lg["ln_mktcap"].notna()
    if mask_lg.sum() > 50:
        c_lg = np.polyfit(df_lg.loc[mask_lg, "ln_mktcap"].values,
                           df_lg.loc[mask_lg, "ln_total_volume"].values, 1)
        df_lg["ln_volume_resid"] = df_lg["ln_total_volume"] - (c_lg[0] * df_lg["ln_mktcap"] + c_lg[1])

    log(f"  Events: {len(df_lg):,}")

    lg_res = {}
    for spec in [1, 2, 3, 4, 5]:
        log(f"  Spec {spec}...")
        res = run_spec(df_lg, dep, spec, lg_legs)
        if res:
            lg_res[spec] = res
            log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}")

    t3 = format_table(lg_res, [f"Spec {s}" for s in [1,2,3,4,5]], [1,2,3,4,5])
    t3.to_csv(os.path.join(OUTPUT_DIR, "regression_event_large.csv"), index=False)
    log(f"\n{t3.to_string(index=False)}")

    # ==================================================================
    # TABLE 4: SPEC 3 ACROSS ALL HORIZONS
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"TABLE 4: SPEC 3 ALL HORIZONS (BuyFrac)")
    log(f"{'='*70}")

    hz_res = {}
    for omega in HOLDING_PERIODS:
        dep_hz = f"AR_signed_market_{omega}d"
        if dep_hz in df.columns:
            log(f"  ω={omega}...")
            res = run_spec(df, dep_hz, 3, legs_enough)
            if res:
                hz_res[omega] = res
                log(f"    N={res['n_obs']:,}, R²={res['r2']:.4f}")

    t4 = format_table(hz_res, [f"{o}d" for o in HOLDING_PERIODS], HOLDING_PERIODS)
    t4.to_csv(os.path.join(OUTPUT_DIR, "regression_event_horizons.csv"), index=False)
    log(f"\n{t4.to_string(index=False)}")

    # ==================================================================
    # TABLE 5: COMPARISON — Spec 3, full vs no-khanna vs large
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"TABLE 5: SPEC 3 COMPARISON (20d, BuyFrac)")
    log(f"{'='*70}")

    comp = {
        "full": run_spec(df, dep, 3, legs_enough),
        "no_khanna": run_spec(df_nk, dep, 3, nk_legs),
        "large": run_spec(df_lg, dep, 3, lg_legs),
    }

    t5 = format_table(comp, ["Full", "Excl. Khanna", "Large (>$30K)"],
                       ["full", "no_khanna", "large"])
    t5.to_csv(os.path.join(OUTPUT_DIR, "regression_event_comparison.csv"), index=False)
    log(f"\n{t5.to_string(index=False)}")

    # ==================================================================
    # TABLE 6: WLS ROBUSTNESS — Spec 3, weighted by sqrt(n_trades)
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"TABLE 6: WLS ROBUSTNESS — sqrt(n_trades) weights")
    log(f"{'='*70}")
    log(f"  Accounts for mechanical variance reduction in large events")

    wls_results = {}

    # OLS (same as main Spec 3)
    wls_results["OLS"] = run_spec(df, dep, 3, legs_enough)

    # WLS
    X, var_names, cl1, cl2, d_used = build_spec(df, 3, legs_enough)
    if X is not None:
        y = d_used[dep].values.astype(float)
        w = d_used["wls_weight"].values.astype(float)
        wls_results["WLS"] = run_ols_twoway(y, X, cl1, cl2, var_names, weights=w)

    if wls_results.get("OLS") and wls_results.get("WLS"):
        log(f"  OLS: N={wls_results['OLS']['n_obs']:,}, R²={wls_results['OLS']['r2']:.4f}")
        log(f"  WLS: N={wls_results['WLS']['n_obs']:,}, R²={wls_results['WLS']['r2']:.4f}")

    wls_table_rows = []
    for var in REPORT_VARS:
        cr, sr = {"Variable": var}, {"Variable": ""}
        for method in ["OLS", "WLS"]:
            if method in wls_results and wls_results[method]:
                res = wls_results[method]
                if var in res["var_names"]:
                    idx = res["var_names"].index(var)
                    c, s = format_coef(res["beta"][idx], res["se_twoway"][idx], res["p_values"][idx])
                    cr[method], sr[method] = c, s
                else:
                    cr[method], sr[method] = "", ""
            else:
                cr[method], sr[method] = "", ""
        wls_table_rows.extend([cr, sr])

    for stat, key_name in [("N", "n_obs"), ("R²", "r2")]:
        row = {"Variable": stat}
        for method in ["OLS", "WLS"]:
            if method in wls_results and wls_results[method]:
                val = wls_results[method][key_name]
                row[method] = f"{val:,}" if stat == "N" else f"{val:.4f}"
        wls_table_rows.append(row)

    t6 = pd.DataFrame(wls_table_rows)
    t6.to_csv(os.path.join(OUTPUT_DIR, "regression_event_wls.csv"), index=False)
    log(f"\n{t6.to_string(index=False)}")

    # ==================================================================
    # KEY COMPARISON
    # ==================================================================
    log(f"\n{'='*70}")
    log("KEY COEFFICIENTS: EVENT-LEVEL Spec 3 (BuyFrac, OLS)")
    log(f"{'='*70}")
    if 3 in main_res:
        res = main_res[3]
        for var in ["BuyFrac", "Senate", "BuyFrac×Senate", "Republican",
                     "BuyFrac×Republican", "Disc_Lag", "Disc_Lag_Sq",
                     "Volatility", "PowerCommittee", "n_trades"]:
            if var in res["var_names"]:
                idx = res["var_names"].index(var)
                c, s = format_coef(res["beta"][idx], res["se_twoway"][idx], res["p_values"][idx])
                log(f"  {var:<22} {c:>15}  {s}")

    elapsed = time.time() - start_time
    log(f"\n{'='*70}")
    log(f"STEP 7C COMPLETE in {elapsed:.1f}s")
    log(f"{'='*70}")

    outputs = [
        "regression_event_main.csv", "regression_event_no_khanna.csv",
        "regression_event_large.csv", "regression_event_horizons.csv",
        "regression_event_comparison.csv", "regression_event_wls.csv",
    ]
    log(f"\nOutput files:")
    for fname in outputs:
        if os.path.exists(os.path.join(OUTPUT_DIR, fname)):
            log(f"  {fname}")

    with open(os.path.join(OUTPUT_DIR, "step7c_diagnostics.txt"), "w") as f:
        f.write("\n".join(LOG))
    log(f"Diagnostics saved to: step7c_diagnostics.txt")


if __name__ == "__main__":
    main()
