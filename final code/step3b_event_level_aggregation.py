"""
step3b_event_level_aggregation.py
==================================
Aggregates trade-level analysis_sample.csv into disclosure-event-level
observations, where each event = (legislator, disclosure_date).

Motivation:
  - Ro Khanna accounts for 42% of trade-level observations
  - Multiple trades filed on the same date share disclosure characteristics
    and have correlated returns
  - Event-level aggregation makes each observation economically independent
  - Reduces noise from small routine trades within larger filings

Aggregation:
  - ARs: Value-weighted average across trades within the event
    (weighted by trade size midpoint)
  - Covariates: Value-weighted averages for continuous firm variables
  - Indicators: Carried forward (same for all trades within a legislator)

New variables at event level:
  - n_trades:         Number of individual trades in the filing
  - n_unique_stocks:  Number of distinct PERMNOs
  - buy_fraction:     Share of trades that are buys (0=all sells, 1=all buys)
  - total_volume:     Sum of trade size midpoints ($)
  - ln_total_volume:  log(total_volume) — replaces ln_size

Input:
  - analysis_sample.csv (25,770 trade-level rows from Step 4)

Output:
  - analysis_sample_event_level.csv (disclosure-event-level dataset)
  - step3b_diagnostics.txt

Usage:
    python step3b_event_level_aggregation.py
"""

import pandas as pd
import numpy as np
import time

ANALYSIS_PATH = "analysis_sample.csv"
OUTPUT_PATH = "analysis_sample_event_level.csv"
OUTPUT_DIAGNOSTICS = "step3b_diagnostics.txt"

HOLDING_PERIODS = [5, 10, 20, 40, 60, 127]
MODELS = ["market", "CAPM", "FF3", "Carhart", "FF5"]

WINSORIZE_LO = 0.01
WINSORIZE_HI = 0.99

LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


def weighted_mean(values, weights):
    """Compute weighted mean, handling NaN in values or weights."""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def main():
    start_time = time.time()
    log("=" * 70)
    log("STEP 3B: DISCLOSURE-EVENT-LEVEL AGGREGATION")
    log("=" * 70)

    # ==================================================================
    # LOAD TRADE-LEVEL DATA
    # ==================================================================
    log("\n--- Loading trade-level data ---")
    df = pd.read_csv(ANALYSIS_PATH)
    df["t0_disc"] = pd.to_datetime(df["t0_disc"])
    df["t0_trade"] = pd.to_datetime(df["t0_trade"])
    log(f"Trade-level rows: {len(df):,}")
    log(f"Unique legislators: {df['politician_name'].nunique()}")
    log(f"Unique disclosure dates: {df['t0_disc'].nunique()}")

    # Ensure numeric
    for col in ["buy_indicator", "senate_indicator", "republican_indicator",
                "power_committee", "disclosure_lag", "disclosure_lag_sq",
                "ln_size", "ln_mktcap", "volatility", "size_midpoint"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ==================================================================
    # WINSORIZE TRADE-LEVEL ARs BEFORE AGGREGATING
    # ==================================================================
    log("\n--- Winsorizing trade-level ARs at 1%/99% ---")
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

    for col in ar_cols:
        vals = df[col].dropna()
        if len(vals) > 100:
            lo = vals.quantile(WINSORIZE_LO)
            hi = vals.quantile(WINSORIZE_HI)
            df[col] = df[col].clip(lo, hi)
    log(f"Winsorized {len(ar_cols)} AR columns")

    # ==================================================================
    # DEFINE GROUPING KEY
    # ==================================================================
    # Each event = (politician_name, t0_disc)
    df["event_key"] = df["politician_name"] + "_" + df["t0_disc"].astype(str)
    n_events = df["event_key"].nunique()
    log(f"\nUnique disclosure events: {n_events:,}")
    log(f"Average trades per event: {len(df) / n_events:.1f}")

    # ==================================================================
    # AGGREGATE TO EVENT LEVEL
    # ==================================================================
    log("\n--- Aggregating to event level ---")

    events = []
    for event_key, group in df.groupby("event_key"):
        g = group.copy()
        n = len(g)
        weights = g["size_midpoint"].values
        weights_valid = np.where(np.isfinite(weights) & (weights > 0), weights, 0)
        total_weight = weights_valid.sum()

        if total_weight == 0:
            # Equal weight fallback
            weights_valid = np.ones(n)
            total_weight = n

        event = {}

        # --- Identifiers ---
        event["politician_name"] = g["politician_name"].iloc[0]
        event["party"] = g["party"].iloc[0]
        event["chamber"] = g["chamber"].iloc[0]
        event["t0_disc"] = g["t0_disc"].iloc[0]
        event["disc_month_year"] = g["disc_month_year"].iloc[0]

        # Use earliest trade date for trade-date anchor
        event["t0_trade"] = g["t0_trade"].min()
        event["trade_month_year"] = g["trade_month_year"].iloc[0]

        # --- Event-level characteristics (NEW) ---
        event["n_trades"] = n
        event["n_unique_stocks"] = g["PERMNO"].nunique()
        # Value-weighted buy fraction (consistent with value-weighted DV)
        event["buy_fraction"] = np.average(g["buy_indicator"].values, weights=weights_valid)
        event["total_volume"] = g["size_midpoint"].sum()
        event["ln_total_volume"] = np.log(event["total_volume"]) if event["total_volume"] > 0 else np.nan

        # --- Indicators (same for all trades within a legislator) ---
        event["senate_indicator"] = int(g["senate_indicator"].iloc[0]) if pd.notna(g["senate_indicator"].iloc[0]) else 0
        event["republican_indicator"] = int(g["republican_indicator"].iloc[0]) if pd.notna(g["republican_indicator"].iloc[0]) else 0
        event["power_committee"] = int(g["power_committee"].iloc[0]) if pd.notna(g["power_committee"].iloc[0]) else 0

        # Buy indicator: 1 if majority buys, 0 if majority sells
        event["buy_indicator"] = 1 if event["buy_fraction"] > 0.5 else 0

        # --- Disclosure lag (should be similar across trades in same filing) ---
        event["disclosure_lag"] = g["disclosure_lag"].median()
        event["disclosure_lag_sq"] = event["disclosure_lag"] ** 2 if pd.notna(event["disclosure_lag"]) else np.nan

        # --- Value-weighted firm-level controls ---
        mktcap_vals = g["ln_mktcap"].values
        event["ln_mktcap"] = weighted_mean(mktcap_vals, weights_valid)

        vol_vals = g["volatility"].values
        event["volatility"] = weighted_mean(vol_vals, weights_valid)

        # --- Sector (mode across trades) ---
        if "siccd" in g.columns:
            event["siccd"] = g["siccd"].mode().iloc[0] if len(g["siccd"].mode()) > 0 else ""
        else:
            event["siccd"] = ""

        # --- Value-weighted ARs ---
        for omega in HOLDING_PERIODS:
            # Signed ARs for each model (primary regression DV)
            for model in MODELS:
                col_signed = f"AR_signed_{model}_{omega}d"
                col_unsigned = f"AR_{model}_{omega}d"

                if col_signed in g.columns:
                    event[col_signed] = weighted_mean(g[col_signed].values, weights_valid)
                if col_unsigned in g.columns:
                    event[col_unsigned] = weighted_mean(g[col_unsigned].values, weights_valid)

            # Raw and benchmark returns
            for prefix in [f"raw_ret_{omega}d", f"sp500_ret_{omega}d", f"AR_market_trade_{omega}d"]:
                if prefix in g.columns:
                    event[prefix] = weighted_mean(g[prefix].values, weights_valid)

        events.append(event)

    df_events = pd.DataFrame(events)
    log(f"Event-level dataset: {len(df_events):,} rows")

    # ==================================================================
    # COMPUTE EVENT-LEVEL RESIDUALIZED SIZE
    # ==================================================================
    log("\n--- Computing event-level residualized size ---")

    # ln_total_volume replaces ln_size at event level
    # Residualize against ln_mktcap (same logic as Eq 4.31)
    mask = df_events["ln_total_volume"].notna() & df_events["ln_mktcap"].notna()
    if mask.sum() > 100:
        coeffs = np.polyfit(
            df_events.loc[mask, "ln_mktcap"].values,
            df_events.loc[mask, "ln_total_volume"].values,
            1
        )
        df_events["ln_volume_resid"] = np.nan
        df_events.loc[mask, "ln_volume_resid"] = (
            df_events.loc[mask, "ln_total_volume"]
            - (coeffs[0] * df_events.loc[mask, "ln_mktcap"] + coeffs[1])
        )
        corr = df_events.loc[mask, "ln_volume_resid"].corr(df_events.loc[mask, "ln_mktcap"])
        log(f"  ln(Volume) = {coeffs[1]:.4f} + {coeffs[0]:.4f} * ln(MktCap)")
        log(f"  Correlation with ln_mktcap: {corr:.6f} (should be ~0)")
    else:
        df_events["ln_volume_resid"] = np.nan

    # ==================================================================
    # INTERACTION TERMS
    # ==================================================================
    df_events["buy_x_senate"] = df_events["buy_indicator"] * df_events["senate_indicator"]
    df_events["buy_x_republican"] = df_events["buy_indicator"] * df_events["republican_indicator"]

    # Cluster variables
    df_events["cluster_legislator"] = df_events["politician_name"].astype(str)
    df_events["cluster_month"] = df_events["disc_month_year"].astype(str)

    # ==================================================================
    # SAVE
    # ==================================================================
    df_events.to_csv(OUTPUT_PATH, index=False)
    log(f"\nSaved: {OUTPUT_PATH} ({len(df_events):,} rows, {len(df_events.columns)} columns)")

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================
    log(f"\n{'=' * 70}")
    log("EVENT-LEVEL AGGREGATION SUMMARY")
    log(f"{'=' * 70}")

    log(f"\n  Trade-level observations:   {len(df):>8,}")
    log(f"  Event-level observations:   {len(df_events):>8,}")
    log(f"  Compression ratio:          {len(df) / len(df_events):>8.1f}x")

    log(f"\n  Unique legislators:         {df_events['politician_name'].nunique():>8}")
    log(f"  Unique disclosure dates:    {df_events['t0_disc'].nunique():>8}")

    log(f"\n  Trades per event distribution:")
    tpe = df_events["n_trades"]
    log(f"    Mean:   {tpe.mean():>8.1f}")
    log(f"    Median: {tpe.median():>8.0f}")
    log(f"    P25:    {tpe.quantile(0.25):>8.0f}")
    log(f"    P75:    {tpe.quantile(0.75):>8.0f}")
    log(f"    Max:    {tpe.max():>8.0f}")

    log(f"\n  Events with 1 trade:        {(tpe == 1).sum():>8,} ({(tpe==1).mean()*100:.1f}%)")
    log(f"  Events with 2-5 trades:     {((tpe >= 2) & (tpe <= 5)).sum():>8,}")
    log(f"  Events with 6-20 trades:    {((tpe >= 6) & (tpe <= 20)).sum():>8,}")
    log(f"  Events with 20+ trades:     {(tpe > 20).sum():>8,}")

    log(f"\n  Buy fraction distribution:")
    bf = df_events["buy_fraction"]
    log(f"    Pure buy events (=1.0):    {(bf == 1.0).sum():>8,}")
    log(f"    Pure sell events (=0.0):   {(bf == 0.0).sum():>8,}")
    log(f"    Mixed events:             {((bf > 0) & (bf < 1)).sum():>8,}")

    log(f"\n  Ro Khanna at event level:")
    rk = df_events[df_events["politician_name"] == "Ro Khanna"]
    log(f"    Events: {len(rk):>6} ({len(rk)/len(df_events)*100:.1f}% of total)")
    log(f"    vs trade-level: 10,860 (42.1% of total)")
    log(f"    Compression: {10860/max(len(rk),1):.0f} trades → 1 event (avg)")

    log(f"\n  Mean signed market-adjusted AR (20d):")
    col20 = "AR_signed_market_20d"
    if col20 in df_events.columns:
        log(f"    Event-level mean: {df_events[col20].mean():.5f}")
        log(f"    Trade-level mean: {df[col20].mean():.5f}")

    # Top events by trade count
    log(f"\n  Top 10 events by trade count:")
    top_events = df_events.nlargest(10, "n_trades")[
        ["politician_name", "t0_disc", "n_trades", "n_unique_stocks",
         "buy_fraction", "total_volume", col20]
    ]
    log(top_events.to_string(index=False))

    elapsed = time.time() - start_time
    log(f"\nCompleted in {elapsed:.1f}s")

    with open(OUTPUT_DIAGNOSTICS, "w") as f:
        f.write("\n".join(LOG))
    log(f"Diagnostics saved to: {OUTPUT_DIAGNOSTICS}")


if __name__ == "__main__":
    main()
