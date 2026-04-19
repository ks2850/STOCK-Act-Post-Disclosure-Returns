"""
trim_crsp.py
============
Trims the full 94-column CRSP daily stock file down to the 14 columns
required by the JIW empirical methodology, and applies basic filters
to remove non-common-stocks and non-US-exchange securities.

Input:  CSRP2022-2026.csv  (~5 GB, 94 columns, ~9.1M rows)
Output: crsp_trimmed.csv   (~500 MB–1 GB, 14 columns, fewer rows)

Usage:
    python trim_crsp.py

    Adjust INPUT_PATH and OUTPUT_PATH below if your file is in a
    different location.

Methodology mapping (each column → equation in JIW paper):
    PERMNO        — Primary security ID; merge key for all windows
    DlyCalDt      — Trading date; defines all event windows
    Ticker        — Merge with Capitol Trades ticker field
    TradingSymbol — Backup ticker merge (Section 3.1.2)
    HdrCUSIP      — Backup ID for PERMNO mapping (Section 3.1.2)
    PERMCO        — Firm-level linking
    SICCD         — Sector fixed effects, Specifications 4-5 (Eq 4.32)
    PrimaryExch   — Filter to NYSE (N), AMEX (A), NASDAQ (Q)
    SecurityType  — Filter to common equity (EQTY)
    ShareType     — Filter to common shares (NS), exclude ADRs (AD)
    DlyCap        — ln(MktCap_i), Specifications 3-5 (Eq 4.28)
    DlyRet        — Returns: Eqs 4.4, 4.8-4.22, 4.29
    vwretd        — r_m,t in factor models (Eqs 4.8-4.17)
    sprtrn        — Market-adjusted AR (Eqs 4.5-4.6)
"""

import pandas as pd
import os
import time

# =====================================================================
# CONFIGURATION — adjust these paths to match your local setup
# =====================================================================
INPUT_PATH = "CSRP2022-2026.csv"
OUTPUT_PATH = "crsp_trimmed.csv"

# =====================================================================
# COLUMNS TO KEEP (14 columns mapped to JIW equations)
# =====================================================================
KEEP_COLS = [
    "PERMNO",           # Primary ID
    "DlyCalDt",         # Date
    "Ticker",           # Merge with Capitol Trades
    "TradingSymbol",    # Backup ticker
    "HdrCUSIP",         # Backup ID
    "PERMCO",           # Firm-level link
    "SICCD",            # Sector FE
    "PrimaryExch",      # Exchange filter
    "SecurityType",     # Security type filter
    "ShareType",        # Share type filter (NS = common, AD = ADR)
    "DlyCap",           # Daily capitalization (PRC x SHROUT)
    "DlyRet",           # Daily total return
    "vwretd",           # CRSP value-weighted market return
    "sprtrn",           # S&P 500 return
]

# =====================================================================
# FILTERS (Section 3.1.1 and 3.1.2 of JIW paper)
# =====================================================================
# NYSE = N, AMEX = A, NASDAQ = Q
VALID_EXCHANGES = {"N", "A", "Q"}

# Common equity only (excludes ADRs, preferred, warrants, etc.)
VALID_SECURITY_TYPE = {"EQTY"}

# NS = Normal Shares (common stock); exclude AD (ADR), PF (preferred), etc.
VALID_SHARE_TYPE = {"NS"}


def main():
    print("=" * 60)
    print("CRSP DAILY STOCK FILE — TRIM & FILTER")
    print("=" * 60)

    # Check input file exists
    if not os.path.exists(INPUT_PATH):
        print(f"\nERROR: Input file not found: {INPUT_PATH}")
        print("Update INPUT_PATH at the top of this script.")
        return

    file_size_gb = os.path.getsize(INPUT_PATH) / (1024 ** 3)
    print(f"\nInput file: {INPUT_PATH} ({file_size_gb:.2f} GB)")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"\nKeeping {len(KEEP_COLS)} of 94 columns:")
    for col in KEEP_COLS:
        print(f"  - {col}")

    # ------------------------------------------------------------------
    # Read in chunks to handle the 5 GB file without blowing up RAM
    # ------------------------------------------------------------------
    CHUNK_SIZE = 500_000  # rows per chunk
    print(f"\nReading in chunks of {CHUNK_SIZE:,} rows...")

    start_time = time.time()
    total_rows_read = 0
    total_rows_kept = 0
    first_chunk = True

    reader = pd.read_csv(
        INPUT_PATH,
        usecols=KEEP_COLS,
        dtype={
            "PERMNO": "int64",
            "Ticker": "str",
            "TradingSymbol": "str",
            "HdrCUSIP": "str",
            "PERMCO": "int64",
            "SICCD": "str",         # SIC can have leading zeros
            "PrimaryExch": "str",
            "SecurityType": "str",
            "ShareType": "str",
            "DlyCalDt": "str",
            "DlyCap": "float64",
            "DlyRet": "float64",
            "vwretd": "float64",
            "sprtrn": "float64",
        },
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_num, chunk in enumerate(reader):
        total_rows_read += len(chunk)

        # Apply filters
        mask = (
            chunk["PrimaryExch"].isin(VALID_EXCHANGES)
            & chunk["SecurityType"].isin(VALID_SECURITY_TYPE)
            & chunk["ShareType"].isin(VALID_SHARE_TYPE)
        )
        filtered = chunk[mask]
        total_rows_kept += len(filtered)

        # Write to output
        filtered.to_csv(
            OUTPUT_PATH,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False

        # Progress update
        elapsed = time.time() - start_time
        print(
            f"  Chunk {chunk_num + 1}: "
            f"read {total_rows_read:>10,} | "
            f"kept {total_rows_kept:>10,} | "
            f"filtered out {total_rows_read - total_rows_kept:>10,} | "
            f"elapsed {elapsed:.0f}s"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    output_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)

    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.1f} seconds")
    print(f"{'=' * 60}")
    print(f"  Rows read:        {total_rows_read:>12,}")
    print(f"  Rows kept:        {total_rows_kept:>12,}")
    print(f"  Rows filtered:    {total_rows_read - total_rows_kept:>12,}")
    print(f"  Keep rate:        {total_rows_kept / total_rows_read * 100:>11.1f}%")
    print(f"  Output file size: {output_size_mb:>11.1f} MB")
    print(f"  Columns:          {len(KEEP_COLS):>12}")
    print(f"\nFilters applied:")
    print(f"  PrimaryExch in {VALID_EXCHANGES}")
    print(f"  SecurityType in {VALID_SECURITY_TYPE}")
    print(f"  ShareType in {VALID_SHARE_TYPE}")
    print(f"\nOutput: {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # Quick sanity check on output
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SANITY CHECK (first 5 rows of output)")
    print(f"{'=' * 60}")
    df_check = pd.read_csv(OUTPUT_PATH, nrows=5)
    print(f"\nColumns: {list(df_check.columns)}")
    print(f"\n{df_check.to_string(index=False)}")

    # Count unique tickers and PERMNOs
    print(f"\n{'=' * 60}")
    print("COUNTING UNIQUE IDENTIFIERS (sampling first 1M rows)...")
    print(f"{'=' * 60}")
    df_sample = pd.read_csv(OUTPUT_PATH, nrows=1_000_000, usecols=["PERMNO", "Ticker"])
    print(f"  Unique PERMNOs:  {df_sample['PERMNO'].nunique():,}")
    print(f"  Unique Tickers:  {df_sample['Ticker'].nunique():,}")


if __name__ == "__main__":
    main()
