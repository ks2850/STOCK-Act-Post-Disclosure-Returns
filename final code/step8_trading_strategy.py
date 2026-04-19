"""
step8_trading_strategy.py
==========================
Backtests rules-based trading strategies derived from the empirical
findings in Chapters 6-7.

IMPORTANT: These are in-sample backtests. The strategies are designed
using patterns observed in the full sample and tested on the same data.
A walk-forward out-of-sample test is provided as robustness.

Strategies:
  1. All Disclosures: follow every disclosed trade
  2. Senate Buys: follow only Senate purchase disclosures
  3. Strategic Delay: follow disclosures with lag > 30 days
  4. Large Conviction: follow large trades (>$15K) by power committee members
  5. Combined Signal: Senate buys OR (lag > 30 AND size > $15K)

For each strategy, reports:
  - Mean return per trade
  - Cumulative return (compounded)
  - Annualized Sharpe ratio
  - Win rate (fraction of trades with positive AR)
  - Number of trades
  - Comparison to passive S&P 500

Also provides:
  - Walk-forward out-of-sample test (train on first half, test on second)
  - Transaction cost sensitivity analysis
  - Monthly strategy returns for time-series analysis

Input:
  - analysis_sample.csv (from step4)

Output:
  - strategy_performance.csv
  - strategy_monthly_returns.csv
  - strategy_walk_forward.csv
  - strategy_cost_sensitivity.csv
  - figure_8_1_cumulative_returns.png
  - figure_8_2_strategy_comparison.png
  - step8_diagnostics.txt

Usage:
    python step8_trading_strategy.py
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =====================================================================
# CONFIGURATION
# =====================================================================
INPUT_PATH = "analysis_sample.csv"
OUTPUT_DIR = "."

HOLDING_PERIODS = [5, 10, 20, 40, 60, 127]
PRIMARY_OMEGA = 20

# Transaction cost assumptions (one-way, in decimal)
COST_SCENARIOS = {
    "Zero": 0.0000,
    "Low (5 bps)": 0.0005,
    "Medium (10 bps)": 0.0010,
    "High (25 bps)": 0.0025,
}

LOG = []

def log(msg):
    print(msg)
    LOG.append(msg)


# =====================================================================
# STRATEGY DEFINITIONS
# =====================================================================

def strategy_all_disclosures(df):
    """Strategy 1: Follow every disclosed trade."""
    return df.copy()

def strategy_senate_buys(df):
    """Strategy 2: Buy stocks disclosed as Senate purchases."""
    return df[(df["senate_indicator"] == 1) & (df["buy_indicator"] == 1)].copy()

def strategy_strategic_delay(df):
    """Strategy 3: Follow disclosures with lag > 30 days."""
    return df[df["disclosure_lag"] > 30].copy()

def strategy_large_conviction(df):
    """Strategy 4: Large trades by power committee members."""
    return df[(df["size_midpoint"] > 15000) & (df["power_committee"] == 1)].copy()

def strategy_combined(df):
    """Strategy 5: Senate buys OR (lag > 30 AND size > $15K)."""
    senate_buys = (df["senate_indicator"] == 1) & (df["buy_indicator"] == 1)
    delayed_large = (df["disclosure_lag"] > 30) & (df["size_midpoint"] > 15000)
    return df[senate_buys | delayed_large].copy()


STRATEGIES = {
    "All Disclosures": strategy_all_disclosures,
    "Senate Buys": strategy_senate_buys,
    "Strategic Delay": strategy_strategic_delay,
    "Large Conviction": strategy_large_conviction,
    "Combined Signal": strategy_combined,
}


# =====================================================================
# PERFORMANCE METRICS
# =====================================================================

def compute_performance(trades, omega, cost_per_trade=0.0):
    """
    Compute strategy performance metrics.

    For buy trades: profit = AR_market_{omega}d (unsigned)
    For sell trades: profit = -AR_market_{omega}d (short the stock)
    Signed AR already encodes this: AR* = AR if buy, -AR if sell

    Transaction cost: applied twice (entry + exit) = 2 * cost_per_trade
    """
    ar_col = f"AR_signed_market_{omega}d"
    if ar_col not in trades.columns:
        return None

    returns = trades[ar_col].dropna().values
    n = len(returns)
    if n == 0:
        return None

    # Apply round-trip transaction cost
    net_returns = returns - 2 * cost_per_trade

    mean_ret = np.mean(net_returns)
    std_ret = np.std(net_returns, ddof=1)
    cumulative = np.prod(1 + net_returns) - 1
    win_rate = np.mean(net_returns > 0)

    # Annualized Sharpe: scale by sqrt(trades_per_year)
    # Approximate trades per year from the data
    if "t0_disc" in trades.columns:
        dates = pd.to_datetime(trades["t0_disc"])
        date_range_years = (dates.max() - dates.min()).days / 365.25
        if date_range_years > 0:
            trades_per_year = n / date_range_years
        else:
            trades_per_year = 252
    else:
        trades_per_year = 252

    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Max drawdown from cumulative returns
    cum_series = np.cumprod(1 + net_returns)
    running_max = np.maximum.accumulate(cum_series)
    drawdowns = (cum_series - running_max) / running_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

    return {
        "n_trades": n,
        "mean_return": mean_ret,
        "std_return": std_ret,
        "cumulative_return": cumulative,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "trades_per_year": trades_per_year,
    }


def compute_sp500_benchmark(df, omega):
    """Compute passive S&P 500 buy-and-hold over the sample period."""
    col = f"sp500_ret_{omega}d"
    if col not in df.columns:
        return None
    sp500_rets = df[col].dropna()
    # Average S&P 500 return over equivalent holding periods
    return {
        "mean_return": sp500_rets.mean(),
        "std_return": sp500_rets.std(),
    }


def compute_monthly_returns(trades, omega):
    """Compute monthly average strategy returns for time-series plot."""
    ar_col = f"AR_signed_market_{omega}d"
    if ar_col not in trades.columns or "t0_disc" not in trades.columns:
        return pd.DataFrame()

    trades = trades.copy()
    trades["t0_disc_dt"] = pd.to_datetime(trades["t0_disc"])
    trades["month"] = trades["t0_disc_dt"].dt.to_period("M")

    monthly = trades.groupby("month").agg(
        mean_ar=(ar_col, "mean"),
        n_trades=(ar_col, "count"),
        std_ar=(ar_col, "std"),
    ).reset_index()
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    return monthly


# =====================================================================
# WALK-FORWARD TEST
# =====================================================================

def walk_forward_test(df, strategy_func, omega):
    """
    Split sample at midpoint. Compute performance on second half only.
    This provides a crude out-of-sample estimate.
    """
    df = df.copy()
    df["t0_disc_dt"] = pd.to_datetime(df["t0_disc"])
    midpoint = df["t0_disc_dt"].quantile(0.5)

    train = df[df["t0_disc_dt"] <= midpoint]
    test = df[df["t0_disc_dt"] > midpoint]

    # Apply strategy filter to test set
    test_trades = strategy_func(test)
    train_trades = strategy_func(train)

    perf_train = compute_performance(train_trades, omega)
    perf_test = compute_performance(test_trades, omega)

    return {
        "train_period": f"{train['t0_disc_dt'].min().strftime('%Y-%m')} to {midpoint.strftime('%Y-%m')}",
        "test_period": f"{midpoint.strftime('%Y-%m')} to {test['t0_disc_dt'].max().strftime('%Y-%m')}",
        "train_n": perf_train["n_trades"] if perf_train else 0,
        "test_n": perf_test["n_trades"] if perf_test else 0,
        "train_mean": perf_train["mean_return"] if perf_train else np.nan,
        "test_mean": perf_test["mean_return"] if perf_test else np.nan,
        "train_sharpe": perf_train["sharpe_ratio"] if perf_train else np.nan,
        "test_sharpe": perf_test["sharpe_ratio"] if perf_test else np.nan,
        "train_win": perf_train["win_rate"] if perf_train else np.nan,
        "test_win": perf_test["win_rate"] if perf_test else np.nan,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    start_time = time.time()
    log("=" * 70)
    log("STEP 8: TRADING STRATEGY BACKTEST")
    log("=" * 70)

    # ==================================================================
    # LOAD
    # ==================================================================
    log("\n--- Loading analysis sample ---")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    log(f"Trades loaded: {len(df):,}")

    # Ensure numeric columns
    for col in ["buy_indicator", "senate_indicator", "republican_indicator",
                "power_committee", "disclosure_lag", "size_midpoint"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ==================================================================
    # TABLE 8.1: STRATEGY PERFORMANCE (PRIMARY — 20d)
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"TABLE 8.1: STRATEGY PERFORMANCE (20-day holding period)")
    log(f"{'='*70}")

    results = []
    for name, func in STRATEGIES.items():
        trades = func(df)
        perf = compute_performance(trades, PRIMARY_OMEGA)
        if perf:
            log(f"\n  {name}:")
            log(f"    N = {perf['n_trades']:,}")
            log(f"    Mean return = {perf['mean_return']*100:.3f}%")
            log(f"    Cumulative  = {perf['cumulative_return']*100:.1f}%")
            log(f"    Sharpe      = {perf['sharpe_ratio']:.3f}")
            log(f"    Win rate    = {perf['win_rate']*100:.1f}%")
            log(f"    Max DD      = {perf['max_drawdown']*100:.1f}%")
            results.append({"Strategy": name, **perf})

    perf_df = pd.DataFrame(results)
    perf_df.to_csv(os.path.join(OUTPUT_DIR, "strategy_performance.csv"), index=False)
    log(f"\nSaved: strategy_performance.csv")

    # ==================================================================
    # TABLE 8.2: PERFORMANCE ACROSS HOLDING PERIODS
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"TABLE 8.2: STRATEGY PERFORMANCE ACROSS HOLDING PERIODS")
    log(f"{'='*70}")

    horizon_results = []
    for omega in HOLDING_PERIODS:
        for name, func in STRATEGIES.items():
            trades = func(df)
            perf = compute_performance(trades, omega)
            if perf:
                horizon_results.append({
                    "Strategy": name,
                    "Horizon": f"{omega}d",
                    "N": perf["n_trades"],
                    "Mean Return": perf["mean_return"],
                    "Sharpe": perf["sharpe_ratio"],
                    "Win Rate": perf["win_rate"],
                })

    hz_df = pd.DataFrame(horizon_results)
    hz_df.to_csv(os.path.join(OUTPUT_DIR, "strategy_horizons.csv"), index=False)
    log(f"\nSaved: strategy_horizons.csv")

    # Print key horizon comparison
    for name in STRATEGIES.keys():
        subset = hz_df[hz_df["Strategy"] == name]
        if len(subset) > 0:
            log(f"\n  {name}:")
            for _, row in subset.iterrows():
                log(f"    {row['Horizon']:>5}: mean={row['Mean Return']*100:+.3f}%, "
                    f"Sharpe={row['Sharpe']:.3f}, Win={row['Win Rate']*100:.1f}%")

    # ==================================================================
    # TABLE 8.3: TRANSACTION COST SENSITIVITY
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"TABLE 8.3: TRANSACTION COST SENSITIVITY (20d)")
    log(f"{'='*70}")

    cost_results = []
    for name, func in STRATEGIES.items():
        trades = func(df)
        for cost_name, cost_val in COST_SCENARIOS.items():
            perf = compute_performance(trades, PRIMARY_OMEGA, cost_per_trade=cost_val)
            if perf:
                cost_results.append({
                    "Strategy": name,
                    "Cost Scenario": cost_name,
                    "Round-Trip Cost": 2 * cost_val,
                    "Mean Net Return": perf["mean_return"],
                    "Sharpe": perf["sharpe_ratio"],
                    "Cumulative": perf["cumulative_return"],
                })

    cost_df = pd.DataFrame(cost_results)
    cost_df.to_csv(os.path.join(OUTPUT_DIR, "strategy_cost_sensitivity.csv"), index=False)
    log(f"\nSaved: strategy_cost_sensitivity.csv")

    for name in STRATEGIES.keys():
        subset = cost_df[cost_df["Strategy"] == name]
        if len(subset) > 0:
            log(f"\n  {name}:")
            for _, row in subset.iterrows():
                log(f"    {row['Cost Scenario']:>15}: net mean={row['Mean Net Return']*100:+.3f}%, "
                    f"Sharpe={row['Sharpe']:.3f}")

    # ==================================================================
    # TABLE 8.4: WALK-FORWARD OUT-OF-SAMPLE
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"TABLE 8.4: WALK-FORWARD TEST (train first half, test second half)")
    log(f"{'='*70}")

    wf_results = []
    for name, func in STRATEGIES.items():
        wf = walk_forward_test(df, func, PRIMARY_OMEGA)
        wf["Strategy"] = name
        wf_results.append(wf)
        log(f"\n  {name}:")
        log(f"    Train: {wf['train_period']}, N={wf['train_n']}, "
            f"mean={wf['train_mean']*100:+.3f}%, Sharpe={wf['train_sharpe']:.3f}")
        log(f"    Test:  {wf['test_period']}, N={wf['test_n']}, "
            f"mean={wf['test_mean']*100:+.3f}%, Sharpe={wf['test_sharpe']:.3f}")

    wf_df = pd.DataFrame(wf_results)
    wf_df.to_csv(os.path.join(OUTPUT_DIR, "strategy_walk_forward.csv"), index=False)
    log(f"\nSaved: strategy_walk_forward.csv")

    # ==================================================================
    # FIGURE 8.1: MONTHLY RETURNS BY STRATEGY
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"FIGURE 8.1: MONTHLY STRATEGY RETURNS")
    log(f"{'='*70}")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    all_monthly = {}
    for idx, (name, func) in enumerate(STRATEGIES.items()):
        trades = func(df)
        monthly = compute_monthly_returns(trades, PRIMARY_OMEGA)
        all_monthly[name] = monthly

        if len(monthly) > 0 and idx < len(axes):
            ax = axes[idx]
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in monthly["mean_ar"]]
            ax.bar(monthly["month_dt"], monthly["mean_ar"] * 100, width=25,
                   color=colors, alpha=0.8, edgecolor="none")
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.set_ylabel("Mean AR (%)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
            ax.tick_params(axis="x", rotation=0, labelsize=8)

            # Add horizontal line for overall mean
            overall_mean = monthly["mean_ar"].mean() * 100
            ax.axhline(y=overall_mean, color="#3498db", linewidth=1.5,
                       linestyle="--", alpha=0.7)
            ax.annotate(f"Mean: {overall_mean:.2f}%",
                       xy=(0.98, 0.95), xycoords="axes fraction",
                       ha="right", va="top", fontsize=8,
                       color="#3498db", fontweight="bold")

    # Remove empty subplot if odd number of strategies
    if len(STRATEGIES) < len(axes):
        for idx in range(len(STRATEGIES), len(axes)):
            axes[idx].set_visible(False)

    plt.suptitle("Monthly Strategy Returns (20-day Signed AR)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_8_1_monthly_returns.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    log("Saved: figure_8_1_monthly_returns.png")

    # Save monthly data
    monthly_all = []
    for name, monthly in all_monthly.items():
        monthly = monthly.copy()
        monthly["Strategy"] = name
        monthly_all.append(monthly)
    if monthly_all:
        pd.concat(monthly_all).to_csv(
            os.path.join(OUTPUT_DIR, "strategy_monthly_returns.csv"), index=False)
        log("Saved: strategy_monthly_returns.csv")

    # ==================================================================
    # FIGURE 8.2: STRATEGY COMPARISON BAR CHART
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"FIGURE 8.2: STRATEGY COMPARISON")
    log(f"{'='*70}")

    if len(perf_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        strategies = perf_df["Strategy"].values
        x = np.arange(len(strategies))
        bar_colors = ["#95a5a6", "#3498db", "#e67e22", "#9b59b6", "#2ecc71"]

        # Panel A: Mean Return
        ax = axes[0]
        vals = perf_df["mean_return"].values * 100
        bars = ax.bar(x, vals, color=bar_colors[:len(strategies)], alpha=0.85)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Mean Return per Trade (%)")
        ax.set_title("Mean Return", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.3f}%", ha="center", va="bottom", fontsize=8)

        # Panel B: Sharpe Ratio
        ax = axes[1]
        vals = perf_df["sharpe_ratio"].values
        bars = ax.bar(x, vals, color=bar_colors[:len(strategies)], alpha=0.85)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Annualized Sharpe Ratio")
        ax.set_title("Sharpe Ratio", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        # Panel C: Win Rate
        ax = axes[2]
        vals = perf_df["win_rate"].values * 100
        bars = ax.bar(x, vals, color=bar_colors[:len(strategies)], alpha=0.85)
        ax.axhline(y=50, color="red", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Win Rate", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

        plt.suptitle("Trading Strategy Comparison (20-day Holding Period)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "figure_8_2_strategy_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        log("Saved: figure_8_2_strategy_comparison.png")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - start_time
    log(f"\n{'='*70}")
    log(f"STEP 8 COMPLETE in {elapsed:.1f}s")
    log(f"{'='*70}")

    log(f"\nOutput files:")
    for fname in ["strategy_performance.csv", "strategy_horizons.csv",
                   "strategy_cost_sensitivity.csv", "strategy_walk_forward.csv",
                   "strategy_monthly_returns.csv",
                   "figure_8_1_monthly_returns.png",
                   "figure_8_2_strategy_comparison.png"]:
        if os.path.exists(os.path.join(OUTPUT_DIR, fname)):
            log(f"  {fname}")

    with open(os.path.join(OUTPUT_DIR, "step8_diagnostics.txt"), "w") as f:
        f.write("\n".join(LOG))
    log("Diagnostics saved to: step8_diagnostics.txt")


if __name__ == "__main__":
    main()
