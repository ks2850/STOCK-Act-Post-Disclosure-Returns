"""
step5_6_descriptive_and_event_study.py
======================================
Generates all descriptive evidence tables (Chapter 6) and
event study plots (Chapter 7 / Figure 7.1) from the analysis sample.

Input files:
  - analysis_sample.csv         (25,770 rows from Step 4)
  - event_study_daily.csv       (3.6M rows from Step 4)

Output files:
  Tables (CSV):
    - table_3_1_summary_stats.csv           (Table 3.1)
    - table_6_1_party_activity.csv          (Table 6.1)
    - table_6_2_correlation_matrix.csv      (Table 6.2)
    - table_6_3_univariate_returns.csv      (Table 6.3)
    - table_chamber_activity.csv            (Section 6.1.3)
    - table_trade_clustering.csv            (Section 6.1.4)
  Figures (PNG):
    - figure_6_1_disclosure_lag_dist.png    (Figure 6.1)
    - figure_7_1_car_dual_anchor.png        (Figure 7.1)
    - figure_ar_by_horizon.png              (supplementary)
  Log:
    - step5_6_diagnostics.txt

Usage:
    pip install matplotlib   (if not already installed)
    python step5_6_descriptive_and_event_study.py
"""

import pandas as pd
import numpy as np
import os
import time

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for servers
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Figures will be skipped.")
    print("Install with: pip install matplotlib")

# =====================================================================
# CONFIGURATION
# =====================================================================
ANALYSIS_PATH = "analysis_sample.csv"
EVENT_STUDY_PATH = "event_study_daily.csv"
OUTPUT_DIR = "."  # change to a subfolder if desired

HOLDING_PERIODS = [5, 10, 20, 40, 60, 127]
MODELS = ["market", "CAPM", "FF3", "Carhart", "FF5"]

LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


def main():
    start_time = time.time()
    log("=" * 70)
    log("STEPS 5-6: DESCRIPTIVE EVIDENCE & EVENT STUDY")
    log("=" * 70)

    # ==================================================================
    # LOAD DATA
    # ==================================================================
    log("\n--- Loading analysis sample ---")
    df = pd.read_csv(ANALYSIS_PATH)
    df["t0_disc"] = pd.to_datetime(df["t0_disc"])
    df["t0_trade"] = pd.to_datetime(df["t0_trade"])
    df["published_date"] = pd.to_datetime(df["published_date"])
    df["traded_date"] = pd.to_datetime(df["traded_date"])
    log(f"Loaded {len(df):,} trades")

    # ==================================================================
    # TABLE 3.1: SUMMARY STATISTICS (Section 3.1.4)
    # ==================================================================
    log("\n--- Table 3.1: Summary Statistics ---")

    summary_vars = {
        "disclosure_lag": "Disclosure Lag (days)",
        "size_midpoint": "Trade Size Midpoint (USD)",
        "ln_size": "ln(Trade Size)",
        "ln_mktcap": "ln(Market Cap)",
        "volatility": "Annualized Volatility",
        "buy_indicator": "Buy Indicator",
        "senate_indicator": "Senate Indicator",
        "republican_indicator": "Republican Indicator",
        "power_committee": "Power Committee Indicator",
    }

    # Add AR columns
    for omega in HOLDING_PERIODS:
        summary_vars[f"AR_market_{omega}d"] = f"Market-Adj AR ({omega}d)"
        summary_vars[f"AR_signed_market_{omega}d"] = f"Signed Market-Adj AR ({omega}d)"

    rows = []
    for col, label in summary_vars.items():
        if col in df.columns:
            s = df[col].dropna()
            rows.append({
                "Variable": label,
                "N": len(s),
                "Mean": s.mean(),
                "Std Dev": s.std(),
                "Min": s.min(),
                "P25": s.quantile(0.25),
                "Median": s.median(),
                "P75": s.quantile(0.75),
                "Max": s.max(),
            })

    table_3_1 = pd.DataFrame(rows)
    table_3_1.to_csv(os.path.join(OUTPUT_DIR, "table_3_1_summary_stats.csv"), index=False)
    log(f"Saved table_3_1_summary_stats.csv")
    log(f"\n{table_3_1.to_string(index=False, float_format='%.4f')}")

    # ==================================================================
    # FIGURE 6.1: DISCLOSURE LAG DISTRIBUTION (Section 6.1.1)
    # ==================================================================
    log("\n--- Figure 6.1: Disclosure Lag Distribution ---")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 6))

        lag = df["disclosure_lag"].dropna()

        ax.hist(lag, bins=range(0, int(lag.max()) + 5, 5), color="#4472C4",
                edgecolor="white", alpha=0.85)

        # Mark the 45-day STOCK Act deadline
        ax.axvline(x=45, color="#C00000", linestyle="--", linewidth=2,
                    label="STOCK Act 45-day deadline")

        ax.set_xlabel("Disclosure Lag (calendar days)", fontsize=12)
        ax.set_ylabel("Number of Trades", fontsize=12)
        ax.set_title("Distribution of Disclosure Lags (Trade Date to Filing Date)",
                      fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)

        # Add summary stats text box
        stats_text = (f"N = {len(lag):,}\n"
                      f"Mean = {lag.mean():.1f} days\n"
                      f"Median = {lag.median():.0f} days\n"
                      f"% within 45 days = {(lag <= 45).mean()*100:.1f}%")
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))

        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "figure_6_1_disclosure_lag_dist.png"),
                    dpi=200)
        plt.close(fig)
        log("Saved figure_6_1_disclosure_lag_dist.png")
    else:
        log("SKIPPED figure (matplotlib not available)")

    # ==================================================================
    # TABLE 6.1: TRADING ACTIVITY BY PARTY (Section 6.1.2)
    # ==================================================================
    log("\n--- Table 6.1: Trading Activity by Party ---")

    party_groups = []
    for party_label, mask in [
        ("Democrat", df["republican_indicator"] == 0),
        ("Republican", df["republican_indicator"] == 1),
    ]:
        sub = df[mask]
        party_groups.append({
            "Party": party_label,
            "Number of Trades": len(sub),
            "Number of Unique Traders": sub["politician_name"].nunique(),
            "Mean Trade Size (USD)": sub["size_midpoint"].mean(),
            "Median Trade Size (USD)": sub["size_midpoint"].median(),
            "Buy/Sell Ratio": sub["buy_indicator"].mean() / (1 - sub["buy_indicator"].mean())
                if sub["buy_indicator"].mean() < 1 else np.inf,
            "Mean Disclosure Lag": sub["disclosure_lag"].mean(),
            "Mean Signed AR (20d)": sub["AR_signed_market_20d"].mean(),
            "Pct Senate": sub["senate_indicator"].mean() * 100,
            "Pct Power Committee": sub["power_committee"].mean() * 100,
        })

    # Total row
    party_groups.append({
        "Party": "Total",
        "Number of Trades": len(df),
        "Number of Unique Traders": df["politician_name"].nunique(),
        "Mean Trade Size (USD)": df["size_midpoint"].mean(),
        "Median Trade Size (USD)": df["size_midpoint"].median(),
        "Buy/Sell Ratio": df["buy_indicator"].mean() / (1 - df["buy_indicator"].mean())
            if df["buy_indicator"].mean() < 1 else np.inf,
        "Mean Disclosure Lag": df["disclosure_lag"].mean(),
        "Mean Signed AR (20d)": df["AR_signed_market_20d"].mean(),
        "Pct Senate": df["senate_indicator"].mean() * 100,
        "Pct Power Committee": df["power_committee"].mean() * 100,
    })

    table_6_1 = pd.DataFrame(party_groups)
    table_6_1.to_csv(os.path.join(OUTPUT_DIR, "table_6_1_party_activity.csv"), index=False)
    log(f"Saved table_6_1_party_activity.csv")
    log(f"\n{table_6_1.to_string(index=False, float_format='%.4f')}")

    # ==================================================================
    # TRADING PATTERNS BY CHAMBER (Section 6.1.3)
    # ==================================================================
    log("\n--- Trading Patterns by Chamber ---")

    chamber_groups = []
    for chamber_label, mask in [
        ("House", df["senate_indicator"] == 0),
        ("Senate", df["senate_indicator"] == 1),
    ]:
        sub = df[mask]
        chamber_groups.append({
            "Chamber": chamber_label,
            "Number of Trades": len(sub),
            "Number of Unique Traders": sub["politician_name"].nunique(),
            "Mean Trade Size (USD)": sub["size_midpoint"].mean(),
            "Buy/Sell Ratio": sub["buy_indicator"].mean() / max(1 - sub["buy_indicator"].mean(), 0.001),
            "Mean Disclosure Lag": sub["disclosure_lag"].mean(),
            "Mean Signed AR (20d)": sub["AR_signed_market_20d"].mean(),
            "Pct Republican": sub["republican_indicator"].mean() * 100,
            "Pct Power Committee": sub["power_committee"].mean() * 100,
        })

    table_chamber = pd.DataFrame(chamber_groups)
    table_chamber.to_csv(os.path.join(OUTPUT_DIR, "table_chamber_activity.csv"), index=False)
    log(f"Saved table_chamber_activity.csv")
    log(f"\n{table_chamber.to_string(index=False, float_format='%.4f')}")

    # ==================================================================
    # TRADE CLUSTERING ANALYSIS (Section 6.1.4)
    # ==================================================================
    log("\n--- Trade Clustering Analysis ---")

    # Top traders by volume
    trader_counts = df.groupby("politician_name").agg(
        n_trades=("PERMNO", "size"),
        n_unique_stocks=("PERMNO", "nunique"),
        mean_signed_ar_20d=("AR_signed_market_20d", "mean"),
        pct_buy=("buy_indicator", "mean"),
        chamber=("chamber", "first"),
        party=("party", "first"),
    ).sort_values("n_trades", ascending=False)

    table_clustering = trader_counts.head(20).reset_index()
    table_clustering.to_csv(os.path.join(OUTPUT_DIR, "table_trade_clustering.csv"), index=False)
    log(f"Saved table_trade_clustering.csv (top 20 traders)")
    log(f"\n{table_clustering.to_string(index=False, float_format='%.4f')}")

    # ==================================================================
    # TABLE 6.2: CORRELATION MATRIX (Section 6.1.5)
    # ==================================================================
    log("\n--- Table 6.2: Correlation Matrix ---")

    corr_vars = [
        "buy_indicator", "senate_indicator", "republican_indicator",
        "ln_size", "disclosure_lag", "ln_mktcap", "volatility",
        "power_committee", "AR_signed_market_20d",
    ]
    corr_labels = [
        "Buy", "Senate", "Republican",
        "ln(Size)", "Disc. Lag", "ln(MktCap)", "Volatility",
        "Power Comm.", "Signed AR(20d)",
    ]

    corr_data = df[corr_vars].dropna()
    corr_matrix = corr_data.corr()
    corr_matrix.index = corr_labels
    corr_matrix.columns = corr_labels

    corr_matrix.to_csv(os.path.join(OUTPUT_DIR, "table_6_2_correlation_matrix.csv"))
    log(f"Saved table_6_2_correlation_matrix.csv (N={len(corr_data):,})")
    log(f"\n{corr_matrix.to_string(float_format='%.3f')}")

    # ==================================================================
    # TABLE 6.3: UNIVARIATE RETURN ANALYSIS (Section 6.1.6)
    # ==================================================================
    log("\n--- Table 6.3: Average Abnormal Returns by Trade Characteristics ---")

    univariate_rows = []

    # By trade direction
    for label, mask in [("Buys", df["buy_indicator"] == 1),
                         ("Sells", df["buy_indicator"] == 0)]:
        sub = df[mask]
        row = {"Group": label, "N": len(sub)}
        for omega in HOLDING_PERIODS:
            col_signed = f"AR_signed_market_{omega}d"
            col_unsigned = f"AR_market_{omega}d"
            row[f"Signed AR {omega}d"] = sub[col_signed].mean()
            row[f"Unsigned AR {omega}d"] = sub[col_unsigned].mean()
        univariate_rows.append(row)

    # By chamber
    for label, mask in [("Senate", df["senate_indicator"] == 1),
                         ("House", df["senate_indicator"] == 0)]:
        sub = df[mask]
        row = {"Group": label, "N": len(sub)}
        for omega in HOLDING_PERIODS:
            row[f"Signed AR {omega}d"] = sub[f"AR_signed_market_{omega}d"].mean()
            row[f"Unsigned AR {omega}d"] = sub[f"AR_market_{omega}d"].mean()
        univariate_rows.append(row)

    # By party
    for label, mask in [("Republican", df["republican_indicator"] == 1),
                         ("Democrat", df["republican_indicator"] == 0)]:
        sub = df[mask]
        row = {"Group": label, "N": len(sub)}
        for omega in HOLDING_PERIODS:
            row[f"Signed AR {omega}d"] = sub[f"AR_signed_market_{omega}d"].mean()
            row[f"Unsigned AR {omega}d"] = sub[f"AR_market_{omega}d"].mean()
        univariate_rows.append(row)

    # By power committee
    for label, mask in [("Power Committee", df["power_committee"] == 1),
                         ("Non-Power Committee", df["power_committee"] == 0)]:
        sub = df[mask]
        row = {"Group": label, "N": len(sub)}
        for omega in HOLDING_PERIODS:
            row[f"Signed AR {omega}d"] = sub[f"AR_signed_market_{omega}d"].mean()
            row[f"Unsigned AR {omega}d"] = sub[f"AR_market_{omega}d"].mean()
        univariate_rows.append(row)

    # By disclosure lag quartile
    df["lag_quartile"] = pd.qcut(df["disclosure_lag"], q=4,
                                  labels=["Q1 (fastest)", "Q2", "Q3", "Q4 (slowest)"],
                                  duplicates="drop")
    for label in df["lag_quartile"].dropna().unique():
        sub = df[df["lag_quartile"] == label]
        row = {"Group": f"Lag {label}", "N": len(sub)}
        for omega in HOLDING_PERIODS:
            row[f"Signed AR {omega}d"] = sub[f"AR_signed_market_{omega}d"].mean()
            row[f"Unsigned AR {omega}d"] = sub[f"AR_market_{omega}d"].mean()
        univariate_rows.append(row)

    # By trade size quartile
    # By trade size bracket (using original size categories since 76% of
    # trades fall in the 1K-15K bracket, making quartile cuts infeasible)
    size_order = ["< 1K", "1K–15K", "15K–50K", "50K–100K", "100K–250K",
                  "250K–500K", "500K–1M", "1M–5M", "5M–25M", "25M–50M"]
    for label in size_order:
        sub = df[df["size"] == label]
        if len(sub) < 5:
            continue
        row = {"Group": f"Size {label}", "N": len(sub)}
        for omega in HOLDING_PERIODS:
            row[f"Signed AR {omega}d"] = sub[f"AR_signed_market_{omega}d"].mean()
            row[f"Unsigned AR {omega}d"] = sub[f"AR_market_{omega}d"].mean()
        univariate_rows.append(row)

    table_6_3 = pd.DataFrame(univariate_rows)
    table_6_3.to_csv(os.path.join(OUTPUT_DIR, "table_6_3_univariate_returns.csv"), index=False)
    log(f"Saved table_6_3_univariate_returns.csv")

    # Print a clean subset
    print_cols = ["Group", "N", "Signed AR 5d", "Signed AR 20d", "Signed AR 60d"]
    log(f"\n{table_6_3[print_cols].to_string(index=False, float_format='%.5f')}")

    # ==================================================================
    # SUPPLEMENTARY: AR BY HOLDING PERIOD AND MODEL
    # ==================================================================
    log("\n--- Supplementary: Mean Signed AR by Model and Horizon ---")

    model_horizon = []
    for model in MODELS:
        row = {"Model": model}
        for omega in HOLDING_PERIODS:
            col = f"AR_signed_{model}_{omega}d"
            if col in df.columns:
                row[f"{omega}d"] = df[col].mean()
                row[f"{omega}d_N"] = df[col].notna().sum()
        model_horizon.append(row)

    table_model = pd.DataFrame(model_horizon)
    log(f"\n{table_model.to_string(index=False, float_format='%.5f')}")

    # ==================================================================
    # FIGURE 7.1: DUAL-ANCHOR CAR PLOT (Section 4.1.3)
    # ==================================================================
    log("\n--- Figure 7.1: Dual-Anchor CAR Plot ---")

    if HAS_MPL:
        log("Loading event study daily data (this may take a moment)...")

        # Read in chunks to handle 3.6M rows
        car_agg = {"disclosure": {}, "trade": {}}

        for chunk in pd.read_csv(EVENT_STUDY_PATH, chunksize=500_000):
            for anchor in ["disclosure", "trade"]:
                sub = chunk[chunk["anchor"] == anchor]
                if len(sub) == 0:
                    continue
                grouped = sub.groupby("k")["daily_ar"].agg(["sum", "count"])
                for k_val, row_data in grouped.iterrows():
                    k_int = int(k_val)
                    if k_int not in car_agg[anchor]:
                        car_agg[anchor][k_int] = {"sum": 0.0, "count": 0}
                    car_agg[anchor][k_int]["sum"] += row_data["sum"]
                    car_agg[anchor][k_int]["count"] += int(row_data["count"])

        # Compute average daily AR and cumulative CAR for each anchor
        def build_car_series(agg_dict):
            k_vals = sorted(agg_dict.keys())
            avg_ar = {}
            for k in k_vals:
                avg_ar[k] = agg_dict[k]["sum"] / agg_dict[k]["count"]

            # Cumulate from k=0
            car = {}
            cumulative = 0.0
            for k in k_vals:
                if k >= 0:
                    cumulative += avg_ar[k]
                    car[k] = cumulative
                else:
                    # Pre-event: cumulate backward from 0
                    pass

            # For pre-event, compute reverse cumulation
            pre_keys = sorted([k for k in k_vals if k < 0])
            pre_cum = 0.0
            pre_car = {}
            for k in reversed(pre_keys):
                pre_cum -= avg_ar[k]  # subtract because we're going backward
                pre_car[k] = -pre_cum  # flip sign

            # Actually, for the plot we want CAR from some start to each k
            # Standard: CAR(-10, K) for the pre-event, CAR(0, K) for post
            # Simpler: just cumulate from the earliest k
            all_car = {}
            cum = 0.0
            for k in k_vals:
                cum += avg_ar[k]
                all_car[k] = cum

            return k_vals, [avg_ar[k] for k in k_vals], [all_car[k] for k in k_vals]

        disc_k, disc_ar, disc_car = build_car_series(car_agg["disclosure"])
        trade_k, trade_ar, trade_car = build_car_series(car_agg["trade"])

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # Panel A: Trade-date anchor
        ax1.plot(trade_k, [c * 100 for c in trade_car], color="#4472C4",
                 linewidth=2)
        ax1.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")
        ax1.axvline(x=0, color="#C00000", linewidth=1, linestyle="--",
                     alpha=0.7, label="Event date (t=0)")
        ax1.set_xlabel("Trading Days Relative to Trade Date", fontsize=11)
        ax1.set_ylabel("Cumulative Abnormal Return (%)", fontsize=11)
        ax1.set_title("Panel A: Anchored at Trade Execution Date",
                       fontsize=12, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Panel B: Disclosure-date anchor
        ax2.plot(disc_k, [c * 100 for c in disc_car], color="#ED7D31",
                 linewidth=2)
        ax2.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")
        ax2.axvline(x=0, color="#C00000", linewidth=1, linestyle="--",
                     alpha=0.7, label="Event date (t=0)")
        ax2.set_xlabel("Trading Days Relative to Disclosure Date", fontsize=11)
        ax2.set_title("Panel B: Anchored at Disclosure Date",
                       fontsize=12, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Cumulative Abnormal Returns Around Congressional Trade Disclosures",
                      fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "figure_7_1_car_dual_anchor.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        log("Saved figure_7_1_car_dual_anchor.png")

        # Also save the plotted data
        disc_df = pd.DataFrame({"k": disc_k, "avg_daily_ar": disc_ar, "car": disc_car})
        disc_df["anchor"] = "disclosure"
        trade_df = pd.DataFrame({"k": trade_k, "avg_daily_ar": trade_ar, "car": trade_car})
        trade_df["anchor"] = "trade"
        car_plot_data = pd.concat([disc_df, trade_df])
        car_plot_data.to_csv(os.path.join(OUTPUT_DIR, "car_plot_data.csv"), index=False)
        log("Saved car_plot_data.csv (underlying data for Figure 7.1)")

    else:
        log("SKIPPED Figure 7.1 (matplotlib not available)")

    # ==================================================================
    # SUPPLEMENTARY FIGURE: AR BY HORIZON
    # ==================================================================
    if HAS_MPL:
        log("\n--- Supplementary: Signed AR by Horizon ---")

        fig, ax = plt.subplots(figsize=(10, 6))

        for model, color, marker in [
            ("market", "#4472C4", "o"),
            ("CAPM", "#ED7D31", "s"),
            ("FF3", "#A5A5A5", "^"),
            ("Carhart", "#FFC000", "D"),
            ("FF5", "#5B9BD5", "v"),
        ]:
            means = []
            for omega in HOLDING_PERIODS:
                col = f"AR_signed_{model}_{omega}d"
                if col in df.columns:
                    means.append(df[col].mean() * 100)
                else:
                    means.append(np.nan)
            ax.plot(HOLDING_PERIODS, means, marker=marker, label=model,
                    color=color, linewidth=1.5, markersize=6)

        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")
        ax.set_xlabel("Holding Period (trading days)", fontsize=11)
        ax.set_ylabel("Mean Signed Abnormal Return (%)", fontsize=11)
        ax.set_title("Mean Signed Abnormal Return by Model and Holding Period",
                      fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(HOLDING_PERIODS)

        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "figure_ar_by_horizon.png"),
                    dpi=200)
        plt.close(fig)
        log("Saved figure_ar_by_horizon.png")

    # ==================================================================
    # PRELIMINARY HYPOTHESES (Section 6.1.7)
    # ==================================================================
    log(f"\n{'=' * 70}")
    log("PRELIMINARY FINDINGS SUMMARY (Section 6.1.7)")
    log(f"{'=' * 70}")

    # H1: Do disclosures carry signal?
    ar_20d = df["AR_signed_market_20d"].dropna()
    try:
        from scipy import stats as scipy_stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        log("WARNING: scipy not installed. T-tests will be skipped.")
        log("Install with: pip install scipy")

    if HAS_SCIPY:
        t_stat, p_val = scipy_stats.ttest_1samp(ar_20d, 0)
        log(f"\nH1: Mean signed market-adjusted AR (20d) = {ar_20d.mean():.5f}")
        log(f"    t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
        log(f"    {'Significant' if p_val < 0.05 else 'Not significant'} at 5% level")

        # H2: Buys vs sells
        buys_ar = df[df["buy_indicator"] == 1]["AR_signed_market_20d"].dropna()
        sells_ar = df[df["buy_indicator"] == 0]["AR_signed_market_20d"].dropna()
        t_diff, p_diff = scipy_stats.ttest_ind(buys_ar, sells_ar)
        log(f"\nH2: Buy signed AR = {buys_ar.mean():.5f}, Sell signed AR = {sells_ar.mean():.5f}")
        log(f"    Difference t-stat = {t_diff:.3f}, p-value = {p_diff:.4f}")

        # H3: Senate vs House
        senate_ar = df[df["senate_indicator"] == 1]["AR_signed_market_20d"].dropna()
        house_ar = df[df["senate_indicator"] == 0]["AR_signed_market_20d"].dropna()
        t_ch, p_ch = scipy_stats.ttest_ind(senate_ar, house_ar)
        log(f"\nH3: Senate signed AR = {senate_ar.mean():.5f}, House signed AR = {house_ar.mean():.5f}")
        log(f"    Difference t-stat = {t_ch:.3f}, p-value = {p_ch:.4f}")

        # H4: Power committee
        pc_ar = df[df["power_committee"] == 1]["AR_signed_market_20d"].dropna()
        npc_ar = df[df["power_committee"] == 0]["AR_signed_market_20d"].dropna()
        t_pc, p_pc = scipy_stats.ttest_ind(pc_ar, npc_ar)
        log(f"\nH4: Power Comm AR = {pc_ar.mean():.5f}, Non-Power AR = {npc_ar.mean():.5f}")
        log(f"    Difference t-stat = {t_pc:.3f}, p-value = {p_pc:.4f}")
    else:
        log(f"\nH1: Mean signed market-adjusted AR (20d) = {ar_20d.mean():.5f}")
        log(f"    (t-test skipped — install scipy)")

    # H5: Dual-anchor comparison (no scipy needed)
    disc_mean = df["AR_market_20d"].mean()
    trade_mean = df["AR_market_trade_20d"].mean()
    log(f"\nH5: Disclosure-date AR (20d) = {disc_mean:.5f}")
    log(f"    Trade-date AR (20d) = {trade_mean:.5f}")
    log(f"    Signal erosion = {trade_mean - disc_mean:.5f}")
    if trade_mean > disc_mean:
        log(f"    Trade-date signal is STRONGER → disclosure lag erodes signal")
    else:
        log(f"    Disclosure-date signal is STRONGER → signal may post-date trade")

    # ==================================================================
    # DONE
    # ==================================================================
    elapsed = time.time() - start_time
    log(f"\n{'=' * 70}")
    log(f"STEPS 5-6 COMPLETE in {elapsed:.1f}s")
    log(f"{'=' * 70}")

    log(f"\nOutput files:")
    outputs = [
        "table_3_1_summary_stats.csv",
        "table_6_1_party_activity.csv",
        "table_chamber_activity.csv",
        "table_trade_clustering.csv",
        "table_6_2_correlation_matrix.csv",
        "table_6_3_univariate_returns.csv",
        "car_plot_data.csv",
    ]
    if HAS_MPL:
        outputs += [
            "figure_6_1_disclosure_lag_dist.png",
            "figure_7_1_car_dual_anchor.png",
            "figure_ar_by_horizon.png",
        ]
    for f in outputs:
        path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            log(f"  {f:<45} {size_kb:>8.1f} KB")

    with open(os.path.join(OUTPUT_DIR, "step5_6_diagnostics.txt"), "w") as f:
        f.write("\n".join(LOG))
    log(f"Diagnostics saved to: step5_6_diagnostics.txt")


if __name__ == "__main__":
    main()
