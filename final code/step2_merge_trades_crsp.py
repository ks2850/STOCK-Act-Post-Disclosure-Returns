"""
step2_merge_trades_crsp.py
==========================
Step 2 of the JIW pipeline: Clean Capitol Trades data and merge with
CRSP to assign PERMNOs to each trade.

Input files:
  - capitol_trades_augmented.csv   (from power committee augmentation step)
  - crsp_trimmed.csv               (from Step 1 CRSP trim)

Output files:
  - trades_matched.csv             (trades successfully matched to PERMNOs)
  - trades_unmatched.csv           (trades that failed to match — for Appendix A)
  - sample_attrition.csv           (attrition table — Section 3.1.2 / Appendix A)
  - merge_diagnostics.txt          (detailed diagnostics log)

Implements:
  - Section 3.1.2: Sample Construction (inclusion criteria, ticker-PERMNO mapping)
  - Section 3.1.3: Data Quality (trade size brackets, disclosure lag)
  - Appendix A: Sample Attrition Table

Usage:
    python step2_merge_trades_crsp.py
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# =====================================================================
# CONFIGURATION
# =====================================================================
CAPITOL_TRADES_PATH = "capitol_trades_augmented.csv"
CRSP_PATH = "crsp_trimmed.csv"

OUTPUT_MATCHED = "trades_matched.csv"
OUTPUT_UNMATCHED = "trades_unmatched.csv"
OUTPUT_ATTRITION = "sample_attrition.csv"
OUTPUT_DIAGNOSTICS = "merge_diagnostics.txt"

# Trade size bracket midpoints (geometric mean — Section 4.1.4)
SIZE_MIDPOINTS = {
    "< 1K":       500,            # assume $1–$1,000, midpoint ~500
    "1K–15K":     np.sqrt(1_001 * 15_000),
    "15K–50K":    np.sqrt(15_001 * 50_000),
    "50K–100K":   np.sqrt(50_001 * 100_000),
    "100K–250K":  np.sqrt(100_001 * 250_000),
    "250K–500K":  np.sqrt(250_001 * 500_000),
    "500K–1M":    np.sqrt(500_001 * 1_000_000),
    "1M–5M":      np.sqrt(1_000_001 * 5_000_000),
    "5M–25M":     np.sqrt(5_000_001 * 25_000_000),
    "25M–50M":    np.sqrt(25_000_001 * 50_000_000),
}

LOG = []  # collect diagnostics


def log(msg):
    print(msg)
    LOG.append(msg)


def parse_traded_date(date_str):
    """Parse Capitol Trades traded_date field (e.g. '25 Dec 2025')."""
    if pd.isna(date_str) or str(date_str).strip() == "":
        return pd.NaT
    date_str = str(date_str).strip().replace("Sept", "Sep")
    for fmt in ("%d %b %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except:
        return pd.NaT


def parse_published_date(date_str):
    """Parse Capitol Trades published_date field.
    Most are like '25 Dec 2025'. Some are '09:01 Yesterday'."""
    if pd.isna(date_str) or str(date_str).strip() == "":
        return pd.NaT
    date_str = str(date_str).strip()
    # Skip relative dates like "09:01 Yesterday" — these can't be resolved
    if "Yesterday" in date_str or "Today" in date_str:
        return pd.NaT
    date_str = date_str.replace("Sept", "Sep")
    for fmt in ("%d %b %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except:
        return pd.NaT


def clean_ticker(ticker_str):
    """Strip exchange suffix (e.g., ':US') and clean ticker."""
    if pd.isna(ticker_str) or str(ticker_str).strip() == "":
        return None
    t = str(ticker_str).strip()
    # Remove :US suffix
    if t.endswith(":US"):
        t = t[:-3]
    # Skip non-US tickers (e.g., :LN, :UD)
    if ":" in t:
        return None
    # Skip tickers that don't look like valid US equity tickers
    if not t.replace(".", "").replace("-", "").isalnum():
        return None
    return t.upper()


def main():
    start_time = time.time()
    log("=" * 70)
    log("STEP 2: CLEAN CAPITOL TRADES & MERGE WITH CRSP PERMNOs")
    log("=" * 70)

    # ==================================================================
    # 2A: LOAD AND CLEAN CAPITOL TRADES
    # ==================================================================
    log("\n--- 2A: Loading Capitol Trades ---")
    ct = pd.read_csv(CAPITOL_TRADES_PATH, dtype=str)
    n_raw = len(ct)
    log(f"Raw Capitol Trades rows: {n_raw:,}")

    attrition = [{"Step": "Raw trades from Capitol Trades", "N": n_raw, "Dropped": 0}]

    # Parse dates
    log("Parsing dates...")
    ct["traded_date_parsed"] = ct["traded_date"].apply(parse_traded_date)
    ct["published_date_parsed"] = ct["published_date"].apply(parse_published_date)

    # For rows where published_date is relative ("Yesterday"), try to
    # impute from traded_date + filed_after_days
    mask_pub_missing = ct["published_date_parsed"].isna() & ct["traded_date_parsed"].notna()
    if mask_pub_missing.sum() > 0:
        ct.loc[mask_pub_missing, "filed_after_days_num"] = pd.to_numeric(
            ct.loc[mask_pub_missing, "filed_after_days"], errors="coerce"
        )
        ct.loc[mask_pub_missing, "published_date_parsed"] = (
            ct.loc[mask_pub_missing, "traded_date_parsed"]
            + pd.to_timedelta(ct.loc[mask_pub_missing, "filed_after_days_num"], unit="D")
        )
        n_imputed = ct.loc[mask_pub_missing, "published_date_parsed"].notna().sum()
        log(f"  Imputed {n_imputed} published dates from traded_date + filed_after_days")

    # Drop rows with unparseable traded_date
    mask_valid_dates = ct["traded_date_parsed"].notna()
    n_bad_dates = (~mask_valid_dates).sum()
    ct = ct[mask_valid_dates].copy()
    log(f"Dropped {n_bad_dates:,} rows with unparseable traded_date")
    attrition.append({"Step": "Drop unparseable traded_date", "N": len(ct), "Dropped": n_bad_dates})

    # Clean tickers
    log("Cleaning tickers...")
    ct["ticker_clean"] = ct["ticker"].apply(clean_ticker)

    # Drop rows with missing/non-US tickers
    mask_valid_ticker = ct["ticker_clean"].notna()
    n_bad_ticker = (~mask_valid_ticker).sum()
    ct = ct[mask_valid_ticker].copy()
    log(f"Dropped {n_bad_ticker:,} rows with missing or non-US tickers")
    attrition.append({"Step": "Drop missing/non-US tickers", "N": len(ct), "Dropped": n_bad_ticker})

    # Filter to BUY and SELL only (Section 3.1.2: common stock transactions)
    mask_valid_txtype = ct["tx_type"].isin(["BUY", "SELL"])
    n_bad_txtype = (~mask_valid_txtype).sum()
    ct = ct[mask_valid_txtype].copy()
    log(f"Dropped {n_bad_txtype:,} rows with non-BUY/SELL tx_type (EXCHANGE, RECEIVE, etc.)")
    attrition.append({"Step": "Drop non-BUY/SELL transactions", "N": len(ct), "Dropped": n_bad_txtype})

    # Filter out ETFs, mutual funds, bonds (Section 3.1.2 inclusion criterion 1)
    etf_keywords = [
        "ETF", "Fund", "Trust", "iShares", "Vanguard", "SPDR", "ProShares",
        "Invesco", "Schwab", "WisdomTree", "VanEck", " LP", " LLC",
        "BOND", "Treasury", "Municipal", "Fixed Income",
        "Futures", "Option", "Warrant",
    ]
    etf_pattern = "|".join(etf_keywords)
    mask_is_etf = ct["issuer_name"].str.contains(etf_pattern, case=False, na=False)

    # Also exclude tickers commonly known to be ETFs
    common_etf_tickers = {
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "EFA",
        "AGG", "BND", "LQD", "TLT", "HYG", "GLD", "SLV", "USO", "XLF",
        "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE",
        "IVV", "IJH", "IJR", "VB", "VO", "VTV", "VUG", "ARKK", "ARKG",
        "SOXL", "TQQQ", "SQQQ", "SPXL", "TNA",
    }
    mask_is_etf_ticker = ct["ticker_clean"].isin(common_etf_tickers)

    mask_exclude = mask_is_etf | mask_is_etf_ticker
    n_etf = mask_exclude.sum()
    ct = ct[~mask_exclude].copy()
    log(f"Dropped {n_etf:,} rows identified as ETFs/funds/bonds/options")
    attrition.append({"Step": "Drop ETFs, funds, bonds, options", "N": len(ct), "Dropped": n_etf})

    # Parse disclosure lag and filter (Section 3.1.2 criterion 4)
    ct["filed_after_days_num"] = pd.to_numeric(ct["filed_after_days"], errors="coerce")
    mask_valid_lag = (ct["filed_after_days_num"] >= 0) & (ct["filed_after_days_num"] <= 365)
    n_bad_lag = (~mask_valid_lag).sum()
    ct = ct[mask_valid_lag].copy()
    log(f"Dropped {n_bad_lag:,} rows with negative or >365 day disclosure lag")
    attrition.append({"Step": "Drop invalid disclosure lags", "N": len(ct), "Dropped": n_bad_lag})

    # Compute trade size midpoint (Section 4.1.4)
    ct["size_midpoint"] = ct["size"].map(SIZE_MIDPOINTS)
    ct["ln_size"] = np.log(ct["size_midpoint"])
    n_missing_size = ct["size_midpoint"].isna().sum()
    log(f"Rows with unmapped size bracket: {n_missing_size:,} (kept, but ln_size will be NaN)")

    # Create Buy indicator (Eq 4.7)
    ct["buy_indicator"] = (ct["tx_type"] == "BUY").astype(int)

    # Create Senate indicator
    ct["senate_indicator"] = (ct["chamber"] == "Senate").astype(int)

    # Create Republican indicator
    ct["republican_indicator"] = (ct["party"] == "Republican").astype(int)

    log(f"\nCapitol Trades after cleaning: {len(ct):,} trades")
    log(f"  Buys:  {ct['buy_indicator'].sum():,}")
    log(f"  Sells: {(ct['buy_indicator'] == 0).sum():,}")
    log(f"  Senate: {ct['senate_indicator'].sum():,}")
    log(f"  House:  {(ct['senate_indicator'] == 0).sum():,}")

    # ==================================================================
    # 2B: BUILD CRSP TICKER-PERMNO LOOKUP
    # ==================================================================
    log("\n--- 2B: Building CRSP ticker-PERMNO lookup ---")
    log("Loading CRSP trimmed data (this may take a minute)...")

    crsp = pd.read_csv(
        CRSP_PATH,
        usecols=["PERMNO", "DlyCalDt", "Ticker", "TradingSymbol"],
        dtype={"PERMNO": "int64", "Ticker": "str", "TradingSymbol": "str", "DlyCalDt": "str"},
        low_memory=False,
    )
    crsp["DlyCalDt"] = pd.to_datetime(crsp["DlyCalDt"])
    log(f"CRSP rows loaded: {len(crsp):,}")

    # Build a lookup: for each (Ticker, date), what is the PERMNO?
    # We need this to handle ticker changes over time.
    # Strategy: for each ticker, find all PERMNOs and their active date ranges.
    log("Building ticker-to-PERMNO date ranges...")

    ticker_permno = (
        crsp.groupby(["Ticker", "PERMNO"])["DlyCalDt"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "first_date", "max": "last_date"})
    )
    # Also build from TradingSymbol as backup
    tsym_permno = (
        crsp.groupby(["TradingSymbol", "PERMNO"])["DlyCalDt"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"TradingSymbol": "Ticker", "min": "first_date", "max": "last_date"})
    )
    # Combine
    lookup = pd.concat([ticker_permno, tsym_permno]).drop_duplicates(
        subset=["Ticker", "PERMNO", "first_date", "last_date"]
    )
    lookup["Ticker"] = lookup["Ticker"].str.upper().str.strip()
    log(f"Ticker-PERMNO lookup entries: {len(lookup):,}")
    log(f"Unique tickers in lookup: {lookup['Ticker'].nunique():,}")

    # Free CRSP from memory — we'll reload selectively later
    del crsp

    # ==================================================================
    # 2C: MERGE TRADES TO PERMNOs
    # ==================================================================
    log("\n--- 2C: Matching trades to CRSP PERMNOs ---")

    def find_permno(ticker, trade_date):
        """Find the PERMNO for a given ticker on a given date."""
        matches = lookup[
            (lookup["Ticker"] == ticker)
            & (lookup["first_date"] <= trade_date)
            & (lookup["last_date"] >= trade_date)
        ]
        if len(matches) == 1:
            return int(matches.iloc[0]["PERMNO"])
        elif len(matches) > 1:
            # Multiple PERMNOs — take the one with the longest date range
            matches = matches.copy()
            matches["range"] = (matches["last_date"] - matches["first_date"]).dt.days
            return int(matches.sort_values("range", ascending=False).iloc[0]["PERMNO"])
        return None

    log("Matching trades (this may take a few minutes)...")
    match_start = time.time()

    # Vectorized approach: merge on ticker first, then filter by date
    ct["ticker_upper"] = ct["ticker_clean"].str.upper().str.strip()
    ct["traded_date_dt"] = ct["traded_date_parsed"]

    # Merge on ticker
    merged = ct.merge(
        lookup,
        left_on="ticker_upper",
        right_on="Ticker",
        how="left",
        suffixes=("", "_crsp"),
    )

    # Filter to rows where trade date falls within the PERMNO's active range
    mask_date_valid = (
        (merged["traded_date_dt"] >= merged["first_date"])
        & (merged["traded_date_dt"] <= merged["last_date"])
    )
    merged_valid = merged[mask_date_valid].copy()

    # For trades with multiple PERMNO matches, keep the longest-range one
    merged_valid["date_range"] = (merged_valid["last_date"] - merged_valid["first_date"]).dt.days
    merged_valid = merged_valid.sort_values("date_range", ascending=False)
    merged_valid = merged_valid.drop_duplicates(
        subset=[col for col in ct.columns if col in merged_valid.columns],
        keep="first",
    )

    # Identify matched vs unmatched
    # Use original index to track
    ct_idx = ct.reset_index()
    matched_indices = set()

    # Rebuild a clean matched set by going back to original ct indices
    ct = ct.reset_index(drop=True)
    ct["_idx"] = ct.index

    merged2 = ct.merge(
        lookup,
        left_on="ticker_upper",
        right_on="Ticker",
        how="left",
    )
    mask_date = (
        (merged2["traded_date_dt"] >= merged2["first_date"])
        & (merged2["traded_date_dt"] <= merged2["last_date"])
    )
    merged2 = merged2[mask_date].copy()
    merged2["date_range"] = (merged2["last_date"] - merged2["first_date"]).dt.days
    merged2 = merged2.sort_values("date_range", ascending=False).drop_duplicates(subset=["_idx"], keep="first")

    matched_idx = set(merged2["_idx"].values)
    unmatched_idx = set(ct["_idx"].values) - matched_idx

    trades_matched = merged2.copy()
    trades_unmatched = ct[ct["_idx"].isin(unmatched_idx)].copy()

    n_matched = len(trades_matched)
    n_unmatched = len(trades_unmatched)

    match_elapsed = time.time() - match_start
    log(f"Matching completed in {match_elapsed:.1f}s")
    log(f"  Matched:   {n_matched:,}")
    log(f"  Unmatched: {n_unmatched:,}")
    log(f"  Match rate: {n_matched / (n_matched + n_unmatched) * 100:.1f}%")

    attrition.append({
        "Step": "Drop trades unmatched to CRSP PERMNO",
        "N": n_matched,
        "Dropped": n_unmatched,
    })

    # ==================================================================
    # 2D: CLEAN UP AND SAVE
    # ==================================================================
    log("\n--- 2D: Saving outputs ---")

    # Select final columns for matched trades
    output_cols = [
        # Trade identifiers
        "PERMNO", "politician_name", "party", "chamber",
        # Trade details
        "issuer_name", "ticker_clean", "published_date_parsed", "traded_date_parsed",
        "filed_after_days_num", "owner", "tx_type", "size", "price",
        "size_midpoint", "ln_size",
        # Indicators
        "buy_indicator", "senate_indicator", "republican_indicator",
        "power_committee", "power_committee_names",
        # CRSP identifiers
        "HdrCUSIP",
    ]
    # Keep only columns that exist
    output_cols = [c for c in output_cols if c in trades_matched.columns]

    trades_out = trades_matched[output_cols].copy()
    trades_out = trades_out.rename(columns={
        "ticker_clean": "ticker",
        "published_date_parsed": "published_date",
        "traded_date_parsed": "traded_date",
        "filed_after_days_num": "disclosure_lag",
    })

    trades_out.to_csv(OUTPUT_MATCHED, index=False)
    log(f"Saved matched trades: {OUTPUT_MATCHED} ({len(trades_out):,} rows)")

    # Save unmatched
    unmatched_out_cols = [
        "politician_name", "issuer_name", "ticker", "ticker_clean",
        "traded_date_parsed", "published_date_parsed", "tx_type", "size",
    ]
    unmatched_out_cols = [c for c in unmatched_out_cols if c in trades_unmatched.columns]
    trades_unmatched[unmatched_out_cols].to_csv(OUTPUT_UNMATCHED, index=False)
    log(f"Saved unmatched trades: {OUTPUT_UNMATCHED} ({len(trades_unmatched):,} rows)")

    # Save attrition table
    attrition_df = pd.DataFrame(attrition)
    attrition_df.to_csv(OUTPUT_ATTRITION, index=False)
    log(f"Saved attrition table: {OUTPUT_ATTRITION}")

    # Print attrition table
    log(f"\n{'=' * 70}")
    log("SAMPLE ATTRITION TABLE (Appendix A)")
    log(f"{'=' * 70}")
    for _, row in attrition_df.iterrows():
        log(f"  {row['Step']:<50} N={row['N']:>8,}  Dropped={row['Dropped']:>6,}")

    # Summary stats on final matched sample
    log(f"\n{'=' * 70}")
    log("MATCHED SAMPLE SUMMARY")
    log(f"{'=' * 70}")
    log(f"Total matched trades: {len(trades_out):,}")
    log(f"Unique PERMNOs:       {trades_out['PERMNO'].nunique():,}")
    log(f"Unique legislators:   {trades_out['politician_name'].nunique():,}")
    log(f"Date range:           {trades_out['traded_date'].min()} to {trades_out['traded_date'].max()}")
    log(f"Buys:                 {trades_out['buy_indicator'].sum():,}")
    log(f"Sells:                {(trades_out['buy_indicator'] == 0).sum():,}")
    log(f"Mean disclosure lag:  {trades_out['disclosure_lag'].mean():.1f} days")
    log(f"Median disclosure lag: {trades_out['disclosure_lag'].median():.0f} days")

    # Save diagnostics log
    with open(OUTPUT_DIAGNOSTICS, "w") as f:
        f.write("\n".join(LOG))
    log(f"\nDiagnostics saved to: {OUTPUT_DIAGNOSTICS}")

    elapsed = time.time() - start_time
    log(f"\nTotal elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
