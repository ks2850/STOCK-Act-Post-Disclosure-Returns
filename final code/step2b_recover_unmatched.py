"""
step2b_recover_unmatched.py
===========================
Recovers ~300-400 trades from trades_unmatched.csv that failed the
initial CRSP merge due to:
  1) Ticker changes (Capitol Trades ticker ≠ CRSP ticker)
  2) US stocks filtered out by ShareType (REITs, BDCs, etc.)

Input files:
  - trades_unmatched.csv             (from Step 2)
  - crsp_trimmed.csv                 (from Step 1 — for ticker change matches)
  - CSRP2022-2026.csv               (full CRSP — for ShareType-filtered stocks)
  - capitol_trades_augmented.csv     (for re-reading full trade details)

Output files:
  - trades_recovered.csv             (recovered trades with PERMNOs)
  - trades_final_unmatched.csv       (truly unrecoverable trades)
  - recovery_diagnostics.txt         (log)

Usage:
    python step2b_recover_unmatched.py

After running, concatenate with your main matched file:
    import pandas as pd
    matched = pd.read_csv("trades_matched.csv")
    recovered = pd.read_csv("trades_recovered.csv")
    combined = pd.concat([matched, recovered], ignore_index=True)
    combined.to_csv("trades_matched_final.csv", index=False)
"""

import pandas as pd
import numpy as np
import time

# =====================================================================
# CONFIGURATION
# =====================================================================
UNMATCHED_PATH = "trades_unmatched.csv"
CRSP_TRIMMED_PATH = "crsp_trimmed.csv"
CRSP_FULL_PATH = "CSRP2022-2026.csv"           # full 5GB file
CAPITOL_TRADES_PATH = "capitol_trades_augmented.csv"

OUTPUT_RECOVERED = "trades_recovered.csv"
OUTPUT_FINAL_UNMATCHED = "trades_final_unmatched.csv"
OUTPUT_DIAGNOSTICS = "recovery_diagnostics.txt"

# =====================================================================
# TICKER CHANGE MAPPING
# Capitol Trades ticker → CRSP ticker(s) to try
# Some companies changed tickers mid-sample, so we try both old and new
# =====================================================================
TICKER_MAP = {
    # Capitol Trades ticker : [list of CRSP tickers to try]
    "NCR":    ["NCR", "NCRV", "VYX"],      # NCR split Oct 2023
    "MRSH":   ["MMC"],                       # Marsh McLennan always MMC in CRSP
    "PSKY":   ["PARA", "PARAA"],             # Paramount Global
    "B":      ["BRK.B", "BRK-B", "BRK B"],  # Berkshire Hathaway Class B
    "CDAY":   ["DAY", "CDAY"],               # Ceridian → Dayforce 2024
    "SQ":     ["SQ", "XYZ"],                 # Block Inc, SQ→XYZ 2024
    "FLT":    ["FLT", "CPAY"],               # FleetCor → Corpay 2023
    "PEAK":   ["PEAK", "DOC"],               # Healthpeak → DOC 2023
    "RE":     ["RE", "EG"],                  # Everest Re → EG 2023
    "COR":    ["COR", "ABC"],                # Cencora, was AmerisourceBergen
    "LTHM":   ["LTHM", "ALTM"],             # Livent → Arcadium Lithium 2024
    "PRMW":   ["PRMW", "QDRN"],             # Primo Water → Primo Brands
    "SGI":    ["TPX"],                        # Tempur Sealy is TPX in CRSP
    "WWE":    ["WWE", "TKO"],                # WWE → TKO Group 2023
    "BWIN":   ["BRP", "BWIN"],               # BRP Group
    "JBGS":   ["JBGS"],                       # JBG Smith — try exact
    "DWAC":   ["DWAC", "DJT"],               # Digital World → Trump Media
    "GPS":    ["GPS", "GAP"],                 # Gap Inc → GAP 2024
    "SIX":    ["SIX", "FUN"],                 # Six Flags → merged with Cedar Fair
    "JBT":    ["JBT", "JBTX"],               # John Bean Technologies
    "CCL":    ["CCL", "CUK"],                 # Carnival — dual listed
    "SMAR":   ["SMAR"],                       # Smartsheet (acquired late 2024)
    "VMW":    ["VMW"],                        # VMware (acquired Nov 2023)
    "AMED":   ["AMED"],                       # Amedisys (acquired 2024)
    "SWTX":   ["SWTX"],                       # SpringWorks (acquired 2024)
    "BECN":   ["BECN"],                       # Beacon Roofing
    "RVTY":   ["RVTY"],                       # Revvity
    "SEAS":   ["SEAS"],                       # SeaWorld
    "ITCI":   ["ITCI"],                       # Intra-Cellular
    "X":      ["X"],                          # US Steel
}

# US stocks that may have been filtered by ShareType in Step 1
# These need to be looked up in the FULL CRSP file
US_STOCKS_RECHECK = {
    "EQR", "HTGC", "ARCC", "AMH", "CUBE", "NSA", "CBOE", "CG",
    "CSWI", "VAPO", "ADMR", "EPR", "RPT", "EQC", "FMCC", "FNMA",
    "SKX", "PNM", "ATH", "WTT",
}

SIZE_MIDPOINTS = {
    "< 1K":       500,
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

LOG = []

def log(msg):
    print(msg)
    LOG.append(msg)


def main():
    start_time = time.time()
    log("=" * 70)
    log("STEP 2B: RECOVER UNMATCHED TRADES")
    log("=" * 70)

    # ==================================================================
    # Load the full Capitol Trades file to get all columns for unmatched
    # ==================================================================
    log("\n--- Loading Capitol Trades for full trade details ---")
    ct_full = pd.read_csv(CAPITOL_TRADES_PATH, dtype=str)

    # Parse dates in the full file (same logic as Step 2)
    def parse_date(d):
        if pd.isna(d) or str(d).strip() == "":
            return pd.NaT
        d = str(d).strip().replace("Sept", "Sep")
        if "Yesterday" in d or "Today" in d:
            return pd.NaT
        for fmt in ("%d %b %Y", "%Y-%m-%d"):
            try:
                return pd.to_datetime(d, format=fmt)
            except:
                continue
        try:
            return pd.to_datetime(d, dayfirst=True)
        except:
            return pd.NaT

    ct_full["traded_date_parsed"] = ct_full["traded_date"].apply(parse_date)
    ct_full["published_date_parsed"] = ct_full["published_date"].apply(parse_date)
    ct_full["ticker_clean"] = ct_full["ticker"].str.replace(":US", "", regex=False).str.strip().str.upper()

    # ==================================================================
    # PART 1: RECOVER TICKER CHANGES FROM TRIMMED CRSP
    # ==================================================================
    log("\n--- Part 1: Recovering ticker changes from trimmed CRSP ---")

    # Build lookup from trimmed CRSP
    log("Loading CRSP trimmed for ticker lookup...")
    crsp_trim = pd.read_csv(
        CRSP_TRIMMED_PATH,
        usecols=["PERMNO", "DlyCalDt", "Ticker", "TradingSymbol"],
        dtype={"PERMNO": "int64", "Ticker": "str", "TradingSymbol": "str"},
        low_memory=False,
    )
    crsp_trim["DlyCalDt"] = pd.to_datetime(crsp_trim["DlyCalDt"])

    # Build lookup: Ticker → (PERMNO, first_date, last_date)
    lookup_ticker = (
        crsp_trim.groupby(["Ticker", "PERMNO"])["DlyCalDt"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "first_date", "max": "last_date"})
    )
    lookup_ticker["Ticker"] = lookup_ticker["Ticker"].str.upper().str.strip()

    # Also from TradingSymbol
    lookup_tsym = (
        crsp_trim.groupby(["TradingSymbol", "PERMNO"])["DlyCalDt"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"TradingSymbol": "Ticker", "min": "first_date", "max": "last_date"})
    )
    lookup_tsym["Ticker"] = lookup_tsym["Ticker"].str.upper().str.strip()

    lookup = pd.concat([lookup_ticker, lookup_tsym]).drop_duplicates()
    del crsp_trim

    def try_match(ticker_list, trade_date):
        """Try matching a list of possible tickers on a given date."""
        for t in ticker_list:
            t = t.upper().strip()
            matches = lookup[
                (lookup["Ticker"] == t)
                & (lookup["first_date"] <= trade_date)
                & (lookup["last_date"] >= trade_date)
            ]
            if len(matches) >= 1:
                best = matches.sort_values(
                    by=[(matches["last_date"] - matches["first_date"]).name
                        if (matches["last_date"] - matches["first_date"]).name
                        else "first_date"],
                    ascending=False
                )
                # Just pick first match
                return int(matches.iloc[0]["PERMNO"]), t
        return None, None

    # Get unmatched trades that have ticker change mappings
    ticker_change_tickers = set(TICKER_MAP.keys())

    # Find these in the full Capitol Trades file
    mask_ticker_change = (
        ct_full["ticker_clean"].isin(ticker_change_tickers)
        & ct_full["traded_date_parsed"].notna()
        & ct_full["tx_type"].isin(["BUY", "SELL"])
    )
    ticker_change_trades = ct_full[mask_ticker_change].copy()
    log(f"Trades with known ticker changes: {len(ticker_change_trades)}")

    recovered_tc = []
    for idx, row in ticker_change_trades.iterrows():
        orig_ticker = row["ticker_clean"]
        trade_date = row["traded_date_parsed"]
        crsp_tickers_to_try = TICKER_MAP.get(orig_ticker, [orig_ticker])

        permno, matched_ticker = try_match(crsp_tickers_to_try, trade_date)
        if permno is not None:
            recovered_tc.append({
                "PERMNO": permno,
                "politician_name": row["politician_name"],
                "party": row.get("party", ""),
                "chamber": row.get("chamber", ""),
                "issuer_name": row.get("issuer_name", ""),
                "ticker": orig_ticker,
                "published_date": row["published_date_parsed"],
                "traded_date": row["traded_date_parsed"],
                "disclosure_lag": pd.to_numeric(row.get("filed_after_days", np.nan), errors="coerce"),
                "owner": row.get("owner", ""),
                "tx_type": row["tx_type"],
                "size": row.get("size", ""),
                "price": row.get("price", ""),
                "size_midpoint": SIZE_MIDPOINTS.get(row.get("size", ""), np.nan),
                "buy_indicator": 1 if row["tx_type"] == "BUY" else 0,
                "senate_indicator": 1 if row.get("chamber", "") == "Senate" else 0,
                "republican_indicator": 1 if row.get("party", "") == "Republican" else 0,
                "power_committee": row.get("power_committee", 0),
                "power_committee_names": row.get("power_committee_names", ""),
                "matched_via": f"ticker_map:{orig_ticker}->{matched_ticker}",
            })

    df_recovered_tc = pd.DataFrame(recovered_tc)
    df_recovered_tc["ln_size"] = np.log(df_recovered_tc["size_midpoint"])
    log(f"Recovered via ticker mapping: {len(df_recovered_tc)}")

    # ==================================================================
    # PART 2: RECOVER US STOCKS FROM FULL CRSP (relaxed filters)
    # ==================================================================
    log("\n--- Part 2: Recovering US stocks from full CRSP (relaxed filters) ---")
    log("Reading full CRSP for specific tickers (this may take a minute)...")

    # Read full CRSP in chunks, only keeping rows for our target tickers
    target_tickers = US_STOCKS_RECHECK
    full_lookup_rows = []

    for chunk in pd.read_csv(
        CRSP_FULL_PATH,
        usecols=["PERMNO", "DlyCalDt", "Ticker", "TradingSymbol", "SecurityType",
                 "ShareType", "PrimaryExch"],
        dtype=str,
        chunksize=500_000,
        low_memory=False,
    ):
        chunk["Ticker_upper"] = chunk["Ticker"].str.upper().str.strip()
        mask = chunk["Ticker_upper"].isin(target_tickers)
        if mask.sum() > 0:
            full_lookup_rows.append(chunk[mask])

    if full_lookup_rows:
        full_lookup = pd.concat(full_lookup_rows, ignore_index=True)
        full_lookup["DlyCalDt"] = pd.to_datetime(full_lookup["DlyCalDt"], errors="coerce")
        full_lookup["PERMNO"] = pd.to_numeric(full_lookup["PERMNO"], errors="coerce")

        log(f"Found {len(full_lookup)} rows for target tickers in full CRSP")
        log(f"SecurityType distribution:")
        log(str(full_lookup["SecurityType"].value_counts().to_dict()))
        log(f"ShareType distribution:")
        log(str(full_lookup["ShareType"].value_counts().to_dict()))

        # Build date-range lookup from full CRSP (no ShareType filter)
        full_ticker_permno = (
            full_lookup.groupby(["Ticker_upper", "PERMNO"])["DlyCalDt"]
            .agg(["min", "max"])
            .reset_index()
            .rename(columns={"Ticker_upper": "Ticker", "min": "first_date", "max": "last_date"})
        )
    else:
        full_ticker_permno = pd.DataFrame(columns=["Ticker", "PERMNO", "first_date", "last_date"])
        log("No rows found for target tickers in full CRSP")

    # Find trades with these tickers in full Capitol Trades
    mask_us_recheck = (
        ct_full["ticker_clean"].isin(US_STOCKS_RECHECK)
        & ct_full["traded_date_parsed"].notna()
        & ct_full["tx_type"].isin(["BUY", "SELL"])
    )
    us_recheck_trades = ct_full[mask_us_recheck].copy()
    log(f"Trades to recheck against full CRSP: {len(us_recheck_trades)}")

    recovered_us = []
    for idx, row in us_recheck_trades.iterrows():
        ticker = row["ticker_clean"]
        trade_date = row["traded_date_parsed"]

        matches = full_ticker_permno[
            (full_ticker_permno["Ticker"] == ticker)
            & (full_ticker_permno["first_date"] <= trade_date)
            & (full_ticker_permno["last_date"] >= trade_date)
        ]
        if len(matches) >= 1:
            permno = int(matches.iloc[0]["PERMNO"])
            recovered_us.append({
                "PERMNO": permno,
                "politician_name": row["politician_name"],
                "party": row.get("party", ""),
                "chamber": row.get("chamber", ""),
                "issuer_name": row.get("issuer_name", ""),
                "ticker": ticker,
                "published_date": row["published_date_parsed"],
                "traded_date": row["traded_date_parsed"],
                "disclosure_lag": pd.to_numeric(row.get("filed_after_days", np.nan), errors="coerce"),
                "owner": row.get("owner", ""),
                "tx_type": row["tx_type"],
                "size": row.get("size", ""),
                "price": row.get("price", ""),
                "size_midpoint": SIZE_MIDPOINTS.get(row.get("size", ""), np.nan),
                "buy_indicator": 1 if row["tx_type"] == "BUY" else 0,
                "senate_indicator": 1 if row.get("chamber", "") == "Senate" else 0,
                "republican_indicator": 1 if row.get("party", "") == "Republican" else 0,
                "power_committee": row.get("power_committee", 0),
                "power_committee_names": row.get("power_committee_names", ""),
                "matched_via": f"full_crsp_recheck:{ticker}",
            })

    df_recovered_us = pd.DataFrame(recovered_us)
    if len(df_recovered_us) > 0:
        df_recovered_us["ln_size"] = np.log(df_recovered_us["size_midpoint"])
    log(f"Recovered via full CRSP recheck: {len(df_recovered_us)}")

    # ==================================================================
    # COMBINE AND SAVE
    # ==================================================================
    log("\n--- Combining and saving ---")

    all_recovered = pd.concat([df_recovered_tc, df_recovered_us], ignore_index=True)

    # Drop duplicates (a trade might appear in both recovery paths)
    before_dedup = len(all_recovered)
    all_recovered = all_recovered.drop_duplicates(
        subset=["politician_name", "ticker", "traded_date", "tx_type", "size"],
        keep="first",
    )
    log(f"Deduplication: {before_dedup} → {len(all_recovered)}")

    # Filter valid disclosure lags (same criteria as Step 2)
    if len(all_recovered) > 0:
        mask_lag = (all_recovered["disclosure_lag"] >= 0) & (all_recovered["disclosure_lag"] <= 365)
        n_bad_lag = (~mask_lag).sum()
        all_recovered = all_recovered[mask_lag].copy()
        log(f"Dropped {n_bad_lag} recovered trades with invalid disclosure lag")

    all_recovered.to_csv(OUTPUT_RECOVERED, index=False)
    log(f"\nSaved recovered trades: {OUTPUT_RECOVERED} ({len(all_recovered)} rows)")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - start_time
    log(f"\n{'=' * 70}")
    log(f"RECOVERY SUMMARY")
    log(f"{'=' * 70}")
    log(f"  Recovered via ticker mapping:  {len(df_recovered_tc):>5}")
    log(f"  Recovered via full CRSP:       {len(df_recovered_us):>5}")
    log(f"  Total recovered (deduped):     {len(all_recovered):>5}")
    log(f"  Elapsed time:                  {elapsed:.1f}s")

    if len(all_recovered) > 0:
        log(f"\n  Recovery breakdown by ticker:")
        for ticker, count in all_recovered["ticker"].value_counts().head(20).items():
            via = all_recovered[all_recovered["ticker"] == ticker]["matched_via"].iloc[0]
            log(f"    {ticker:<10} {count:>3} trades  ({via})")

    log(f"\n  NEXT STEP: Combine with trades_matched.csv:")
    log(f"    import pandas as pd")
    log(f"    matched = pd.read_csv('trades_matched.csv')")
    log(f"    recovered = pd.read_csv('trades_recovered.csv')")
    log(f"    combined = pd.concat([matched, recovered], ignore_index=True)")
    log(f"    combined.to_csv('trades_matched_final.csv', index=False)")
    log(f"    print(f'Final sample: {{len(combined)}} trades')")

    with open(OUTPUT_DIAGNOSTICS, "w") as f:
        f.write("\n".join(LOG))
    log(f"\nDiagnostics saved to: {OUTPUT_DIAGNOSTICS}")


if __name__ == "__main__":
    main()
