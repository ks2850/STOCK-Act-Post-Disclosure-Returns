"""
step4_event_windows.py (FINAL — all fixes applied)
=====================================================
Step 4 of the JIW pipeline: For each matched trade, define event dates,
extract return windows, run factor regressions, and compute all variables
needed for the cross-sectional regressions.

FIXES APPLIED:
  1. Alpha (betas[0]) NOT included in predicted returns (Eqs 4.9/4.12/4.15/4.18)
  2. Estimation window [t0-130, t0-11] inclusive = 120 days (was 119)
  3. Raw and predicted returns computed over SAME clean days
  4. Duplicate CRSP dates per PERMNO deduplicated (distribution events)
  5. buy_indicator NaN-safe conversion
  6. n_est_days reuses existing computation (no wasted regression)
  7. Event study rows flushed periodically to prevent OOM
  8. disc_month_year, trade_month_year, disclosure_lag_sq added
  9. raw_ret and sp500_ret as separate columns
 10. Event study daily CAR output for Figure 7.1

Input files:
  - trades_matched_final.csv    (25,918 trades with PERMNOs)
  - crsp_trimmed.csv            (4.85M rows, daily stock data)
  - factors_daily.csv           (920 rows, FF5 + MOM)

Output files:
  - analysis_sample.csv         (regression-ready dataset, one row per trade)
  - event_study_daily.csv       (daily CARs for event study plots)
  - step4_diagnostics.txt       (log)

Usage:
    python step4_event_windows.py
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================
TRADES_PATH = "trades_matched_final.csv"
CRSP_PATH = "crsp_trimmed.csv"
FACTORS_PATH = "factors_daily.csv"

OUTPUT_PATH = "analysis_sample.csv"
OUTPUT_EVENT_STUDY = "event_study_daily.csv"
OUTPUT_DIAGNOSTICS = "step4_diagnostics.txt"

HOLDING_PERIODS = [5, 10, 20, 40, 60, 127]

EST_WINDOW_START = 130
EST_WINDOW_END = 11
MIN_EST_DAYS = 60

VOL_WINDOW = 60

CAR_PRE = 10
CAR_POST = 60

ES_FLUSH_THRESHOLD = 500_000  # flush event study rows to disk at this count

LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


def main():
    start_time = time.time()
    log("=" * 70)
    log("STEP 4: EVENT WINDOWS & REGRESSION VARIABLES (FINAL)")
    log("=" * 70)

    # ==================================================================
    # 4A: LOAD ALL DATA
    # ==================================================================
    log("\n--- 4A: Loading data ---")

    trades = pd.read_csv(TRADES_PATH)
    # FIX #5: NaN-safe PERMNO conversion
    trades = trades[trades["PERMNO"].notna()].copy()
    trades["PERMNO"] = trades["PERMNO"].astype(int)
    trades["published_date"] = pd.to_datetime(trades["published_date"])
    trades["traded_date"] = pd.to_datetime(trades["traded_date"])
    log(f"Trades loaded: {len(trades):,}")

    log("Loading CRSP (this may take a minute)...")
    crsp = pd.read_csv(
        CRSP_PATH,
        dtype={
            "PERMNO": "int64",
            "DlyCalDt": "str",
            "DlyRet": "float64",
            "DlyCap": "float64",
            "vwretd": "float64",
            "sprtrn": "float64",
            "SICCD": "str",
        },
        usecols=["PERMNO", "DlyCalDt", "DlyRet", "DlyCap", "vwretd", "sprtrn", "SICCD"],
        low_memory=False,
    )
    crsp["DlyCalDt"] = pd.to_datetime(crsp["DlyCalDt"])
    crsp = crsp.sort_values(["PERMNO", "DlyCalDt"]).reset_index(drop=True)
    crsp = crsp[crsp["DlyRet"].notna()].copy()
    log(f"CRSP loaded: {len(crsp):,} rows (after dropping missing returns)")

    factors = pd.read_csv(FACTORS_PATH)
    factors["date"] = pd.to_datetime(factors["date"])
    log(f"Factors loaded: {len(factors):,} rows")

    log("Merging factors onto CRSP...")
    crsp = crsp.merge(factors, left_on="DlyCalDt", right_on="date", how="left")
    crsp.drop(columns=["date"], inplace=True)
    n_missing_factors = crsp["MktRF"].isna().sum()
    log(f"CRSP rows missing factor data: {n_missing_factors:,}")

    trading_dates = sorted(crsp["DlyCalDt"].unique())
    trading_dates_arr = np.array(trading_dates)
    log(f"Trading calendar: {len(trading_dates):,} dates "
        f"from {trading_dates[0]} to {trading_dates[-1]}")

    # ==================================================================
    # 4B: BUILD PERMNO-INDEXED LOOKUP
    # ==================================================================
    log("\n--- 4B: Building PERMNO-indexed lookup ---")

    permno_data = {}
    relevant_permnos = set(trades["PERMNO"].unique())
    log(f"Relevant PERMNOs: {len(relevant_permnos):,}")

    n_dupes_total = 0
    for permno, group in crsp[crsp["PERMNO"].isin(relevant_permnos)].groupby("PERMNO"):
        # FIX #4: Drop duplicate dates from distribution events
        n_before = len(group)
        g = group.drop_duplicates(subset=["DlyCalDt"], keep="first")
        n_dupes_total += n_before - len(g)
        g = g.set_index("DlyCalDt").sort_index()
        permno_data[permno] = g

    log(f"PERMNO lookup built for {len(permno_data):,} securities")
    log(f"Duplicate date rows removed: {n_dupes_total:,}")
    del crsp

    # ==================================================================
    # 4C: HELPER FUNCTIONS
    # ==================================================================

    def find_event_date(target_date):
        """First trading day on or after target_date (Eq 4.2/4.3)."""
        if pd.isna(target_date):
            return pd.NaT
        idx = np.searchsorted(trading_dates_arr, target_date, side="left")
        if idx < len(trading_dates_arr):
            return pd.Timestamp(trading_dates_arr[idx])
        return pd.NaT

    def get_bh_return(ret_array):
        """Compound daily returns (Eq 4.4). Input: numpy array."""
        return np.prod(1.0 + ret_array) - 1.0

    def compute_market_ar(stock_data, t0, omega):
        """
        Market-adjusted AR (Eqs 4.4-4.6). Uses sprtrn (S&P 500).
        Returns start at k=1 (day AFTER event).
        Returns: (raw_ret, sp500_ret, ar, n_days)
        """
        all_dates = stock_data.index
        dates_after = all_dates[all_dates > t0]

        if len(dates_after) < omega:
            return np.nan, np.nan, np.nan, 0

        event_dates = dates_after[:omega]
        event_data = stock_data.loc[event_dates]

        # Drop any rows where either return is NaN
        clean = event_data[["DlyRet", "sprtrn"]].dropna()
        if len(clean) < omega * 0.8:
            return np.nan, np.nan, np.nan, 0

        raw_ret = get_bh_return(clean["DlyRet"].values)
        sp500_ret = get_bh_return(clean["sprtrn"].values)
        ar = raw_ret - sp500_ret

        return raw_ret, sp500_ret, ar, len(clean)

    def compute_factor_ar(stock_data, t0, omega, model):
        """
        Factor-model AR (Eqs 4.8-4.19).
        FIX #1: No alpha in predicted returns.
        FIX #2: Estimation window [t0-130, t0-11] inclusive = 120 days.
        FIX #3: Raw and predicted over same clean days.
        Returns: (ar, n_est_days)
        """
        all_dates = stock_data.index

        # Estimation window: [t0-130, t0-11] inclusive
        t0_pos = np.searchsorted(all_dates, t0, side="left")
        est_start_pos = max(0, t0_pos - EST_WINDOW_START)
        est_end_pos = max(0, t0_pos - EST_WINDOW_END + 1)  # +1 for inclusive

        if est_end_pos <= est_start_pos:
            return np.nan, 0

        est_dates = all_dates[est_start_pos:est_end_pos]
        est_data = stock_data.loc[est_dates].dropna(
            subset=["DlyRet", "MktRF", "RF"]
        )

        if len(est_data) < MIN_EST_DAYS:
            return np.nan, len(est_data)

        n_est = len(est_data)

        # LHS: excess stock return
        y = est_data["DlyRet"].values - est_data["RF"].values

        # RHS: intercept + factors
        if model == "CAPM":
            X = np.column_stack([np.ones(n_est), est_data["MktRF"].values])
        elif model == "FF3":
            X = np.column_stack([
                np.ones(n_est), est_data["MktRF"].values,
                est_data["SMB"].values, est_data["HML"].values,
            ])
        elif model == "Carhart":
            X = np.column_stack([
                np.ones(n_est), est_data["MktRF"].values,
                est_data["SMB"].values, est_data["HML"].values,
                est_data["MOM"].values,
            ])
        elif model == "FF5":
            X = np.column_stack([
                np.ones(n_est), est_data["MktRF"].values,
                est_data["SMB"].values, est_data["HML"].values,
                est_data["RMW"].values, est_data["CMA"].values,
            ])
        else:
            return np.nan, 0

        try:
            betas = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.nan, n_est

        # Event window: k=1 to k=omega
        dates_after = all_dates[all_dates > t0]
        if len(dates_after) < omega:
            return np.nan, n_est

        event_dates = dates_after[:omega]
        event_data = stock_data.loc[event_dates]

        # Required factor columns per model
        required = ["DlyRet", "MktRF", "RF"]
        if model in ("FF3", "Carhart", "FF5"):
            required += ["SMB", "HML"]
        if model == "Carhart":
            required += ["MOM"]
        if model == "FF5":
            required += ["RMW", "CMA"]

        # FIX #3: Both raw and predicted over same clean days
        event_clean = event_data.dropna(subset=required)
        if len(event_clean) < omega * 0.8:
            return np.nan, n_est

        raw_return = get_bh_return(event_clean["DlyRet"].values)

        # FIX #1: predicted = RF + beta*factors, NO alpha (betas[0] skipped)
        rf = event_clean["RF"].values
        mktrf = event_clean["MktRF"].values

        if model == "CAPM":
            predicted_daily = rf + betas[1] * mktrf
        elif model == "FF3":
            predicted_daily = (rf + betas[1] * mktrf
                               + betas[2] * event_clean["SMB"].values
                               + betas[3] * event_clean["HML"].values)
        elif model == "Carhart":
            predicted_daily = (rf + betas[1] * mktrf
                               + betas[2] * event_clean["SMB"].values
                               + betas[3] * event_clean["HML"].values
                               + betas[4] * event_clean["MOM"].values)
        elif model == "FF5":
            predicted_daily = (rf + betas[1] * mktrf
                               + betas[2] * event_clean["SMB"].values
                               + betas[3] * event_clean["HML"].values
                               + betas[4] * event_clean["RMW"].values
                               + betas[5] * event_clean["CMA"].values)

        predicted_bh = get_bh_return(predicted_daily)
        ar = raw_return - predicted_bh
        return ar, n_est

    def compute_car_daily(stock_data, t0, pre=CAR_PRE, post=CAR_POST):
        """
        Daily ARs and CARs for event study (Eqs 4.21-4.22).
        CARs start at k=0 (includes event day).
        Index is guaranteed unique (FIX #4), so .loc returns scalars.
        """
        all_dates = stock_data.index
        t0_pos = np.searchsorted(all_dates, t0, side="left")

        if t0_pos >= len(all_dates):
            return []

        start_pos = max(0, t0_pos - pre)
        end_pos = min(len(all_dates), t0_pos + post + 1)

        rows = []
        cumulative_ar = 0.0

        for pos in range(start_pos, end_pos):
            k = pos - t0_pos
            dt = all_dates[pos]
            ret_val = stock_data.at[dt, "DlyRet"]
            sp_val = stock_data.at[dt, "sprtrn"]

            # .at on a unique index always returns scalar, but guard anyway
            if isinstance(ret_val, pd.Series):
                ret_val = ret_val.iloc[0]
            if isinstance(sp_val, pd.Series):
                sp_val = sp_val.iloc[0]

            if pd.isna(ret_val) or pd.isna(sp_val):
                continue

            daily_ar = float(ret_val) - float(sp_val)

            if k >= 0:
                cumulative_ar += daily_ar

            rows.append({
                "k": k,
                "daily_ret": float(ret_val),
                "daily_sp500": float(sp_val),
                "daily_ar": daily_ar,
                "car": cumulative_ar if k >= 0 else np.nan,
            })

        return rows

    def find_month_end_mktcap(stock_data, t0):
        """
        ln(MktCap) at most recent month-end BEFORE t0 (Spec 3-5).
        If t0 = March 15 → last trading day of February.
        If t0 = March 1  → last trading day of February.
        """
        pre_dates = stock_data.index[stock_data.index < t0]
        if len(pre_dates) == 0:
            return np.nan

        pre_data = stock_data.loc[pre_dates]
        monthly_last = pre_data.resample("ME").last()

        first_of_month = t0.replace(day=1)
        valid = monthly_last[monthly_last.index < first_of_month]

        if len(valid) > 0:
            cap = valid["DlyCap"].iloc[-1]
            if pd.notna(cap) and cap > 0:
                return np.log(cap)

        if len(monthly_last) > 0:
            cap = monthly_last["DlyCap"].iloc[-1]
            if pd.notna(cap) and cap > 0:
                return np.log(cap)

        last_cap = pre_data["DlyCap"].dropna()
        if len(last_cap) > 0 and last_cap.iloc[-1] > 0:
            return np.log(last_cap.iloc[-1])

        return np.nan

    def compute_volatility(stock_data, t0):
        """
        Annualized volatility (Eq 4.29).
        60 trading days before t0, NOT including t0.
        ddof=1 → denominator = 59.
        """
        pre_dates = stock_data.index[stock_data.index < t0]
        if len(pre_dates) < VOL_WINDOW:
            return np.nan

        vol_dates = pre_dates[-VOL_WINDOW:]
        vol_returns = stock_data.loc[vol_dates, "DlyRet"].dropna().values

        if len(vol_returns) < 30:
            return np.nan

        return float(np.std(vol_returns, ddof=1) * np.sqrt(252))

    # ==================================================================
    # 4D: PROCESS EACH TRADE
    # ==================================================================
    log("\n--- 4D: Processing trades ---")

    # FIX #7: Clear event study file from any prior run
    if os.path.exists(OUTPUT_EVENT_STUDY):
        os.remove(OUTPUT_EVENT_STUDY)
    es_header_written = False

    results = []
    event_study_rows = []
    n_trades = len(trades)
    n_success = 0
    n_no_permno_data = 0
    n_no_event_date = 0
    n_insufficient_data = 0

    report_every = 2500

    for i, row in trades.iterrows():
        if (i + 1) % report_every == 0 or i == 0:
            elapsed = time.time() - start_time
            log(f"  Processing trade {i+1:>6,}/{n_trades:,} "
                f"(ok={n_success:,}, no_data={n_no_permno_data:,}, "
                f"no_event={n_no_event_date:,}, insuff={n_insufficient_data:,}) "
                f"[{elapsed:.0f}s]")

        permno = row["PERMNO"]
        pub_date = row["published_date"]
        trade_date = row["traded_date"]

        if permno not in permno_data:
            n_no_permno_data += 1
            continue
        stock = permno_data[permno]

        t0_disc = find_event_date(pub_date)
        t0_trade = find_event_date(trade_date)

        if pd.isna(t0_disc):
            n_no_event_date += 1
            continue

        dates_after_t0 = stock.index[stock.index > t0_disc]
        if len(dates_after_t0) < HOLDING_PERIODS[0]:
            n_insufficient_data += 1
            continue

        # --- Covariates ---
        ln_mktcap = find_month_end_mktcap(stock, t0_disc)
        volatility = compute_volatility(stock, t0_disc)

        siccd = ""
        if "SICCD" in stock.columns and stock["SICCD"].notna().any():
            siccd = str(stock["SICCD"].dropna().iloc[-1])

        # --- Build result row ---
        # FIX #5: NaN-safe indicator conversions
        buy_raw = row.get("buy_indicator", 0)
        is_buy = int(buy_raw) if pd.notna(buy_raw) else 0

        senate_raw = row.get("senate_indicator", 0)
        is_senate = int(senate_raw) if pd.notna(senate_raw) else 0

        repub_raw = row.get("republican_indicator", 0)
        is_repub = int(repub_raw) if pd.notna(repub_raw) else 0

        pc_raw = row.get("power_committee", 0)
        is_pc = int(pc_raw) if pd.notna(pc_raw) else 0

        lag_val = row.get("disclosure_lag", np.nan)
        lag_val = float(lag_val) if pd.notna(lag_val) else np.nan

        result = {
            "PERMNO": permno,
            "politician_name": row.get("politician_name", ""),
            "party": row.get("party", ""),
            "chamber": row.get("chamber", ""),
            "issuer_name": row.get("issuer_name", ""),
            "ticker": row.get("ticker", ""),

            "t0_disc": t0_disc,
            "t0_trade": t0_trade,
            "published_date": pub_date,
            "traded_date": trade_date,

            "disclosure_lag": lag_val,
            "disclosure_lag_sq": lag_val ** 2 if not np.isnan(lag_val) else np.nan,
            "tx_type": row.get("tx_type", ""),
            "size": row.get("size", ""),
            "size_midpoint": row.get("size_midpoint", np.nan),
            "ln_size": row.get("ln_size", np.nan),
            "owner": row.get("owner", ""),

            "buy_indicator": is_buy,
            "senate_indicator": is_senate,
            "republican_indicator": is_repub,
            "power_committee": is_pc,

            "ln_mktcap": ln_mktcap,
            "volatility": volatility,
            "siccd": siccd,

            "disc_month_year": t0_disc.strftime("%Y-%m"),
            "trade_month_year": t0_trade.strftime("%Y-%m")
                if pd.notna(t0_trade) else "",
        }

        # --- ARs for each holding period ---
        # FIX #6: Save n_est from first FF5 call at omega=20
        n_est_saved = 0

        for omega in HOLDING_PERIODS:
            # Market-adjusted (primary)
            raw_ret, sp500_ret, ar_mkt, _ = compute_market_ar(
                stock, t0_disc, omega)
            result[f"raw_ret_{omega}d"] = raw_ret
            result[f"sp500_ret_{omega}d"] = sp500_ret
            result[f"AR_market_{omega}d"] = ar_mkt

            # Factor models
            ar_capm, _ = compute_factor_ar(stock, t0_disc, omega, "CAPM")
            result[f"AR_CAPM_{omega}d"] = ar_capm

            ar_ff3, _ = compute_factor_ar(stock, t0_disc, omega, "FF3")
            result[f"AR_FF3_{omega}d"] = ar_ff3

            ar_c4, _ = compute_factor_ar(stock, t0_disc, omega, "Carhart")
            result[f"AR_Carhart_{omega}d"] = ar_c4

            ar_ff5, n_est_tmp = compute_factor_ar(stock, t0_disc, omega, "FF5")
            result[f"AR_FF5_{omega}d"] = ar_ff5

            # FIX #6: Capture n_est from omega=20 FF5
            if omega == 20:
                n_est_saved = n_est_tmp

            # Trade-date anchor
            if pd.notna(t0_trade):
                _, _, ar_trade, _ = compute_market_ar(stock, t0_trade, omega)
                result[f"AR_market_trade_{omega}d"] = ar_trade
            else:
                result[f"AR_market_trade_{omega}d"] = np.nan

        # FIX #6: Use saved value instead of recomputing
        result["n_est_days"] = n_est_saved

        # --- Signed AR (Eq 4.7) ---
        for omega in HOLDING_PERIODS:
            for model in ["market", "CAPM", "FF3", "Carhart", "FF5"]:
                col = f"AR_{model}_{omega}d"
                val = result.get(col, np.nan)
                if is_buy == 0 and pd.notna(val):
                    result[f"AR_signed_{model}_{omega}d"] = -val
                else:
                    result[f"AR_signed_{model}_{omega}d"] = val

        results.append(result)
        n_success += 1

        # --- Event study daily CARs ---
        car_disc = compute_car_daily(stock, t0_disc)
        for cr in car_disc:
            event_study_rows.append({
                "trade_idx": n_success - 1,
                "PERMNO": permno,
                "anchor": "disclosure",
                **cr,
            })

        if pd.notna(t0_trade):
            car_trade = compute_car_daily(stock, t0_trade)
            for cr in car_trade:
                event_study_rows.append({
                    "trade_idx": n_success - 1,
                    "PERMNO": permno,
                    "anchor": "trade",
                    **cr,
                })

        # FIX #7: Flush event study rows periodically to prevent OOM
        if len(event_study_rows) >= ES_FLUSH_THRESHOLD:
            es_chunk = pd.DataFrame(event_study_rows)
            es_chunk.to_csv(
                OUTPUT_EVENT_STUDY,
                mode="a",
                header=not es_header_written,
                index=False,
            )
            es_header_written = True
            event_study_rows = []

    # ==================================================================
    # 4E: ASSEMBLE AND SAVE
    # ==================================================================
    log(f"\n--- 4E: Assembling results ---")

    df = pd.DataFrame(results)
    log(f"Trades processed successfully: {n_success:,}")
    log(f"Trades failed (no PERMNO data): {n_no_permno_data:,}")
    log(f"Trades failed (no event date): {n_no_event_date:,}")
    log(f"Trades failed (insufficient data): {n_insufficient_data:,}")

    # --- Residualized trade size (Eq 4.31) ---
    log("\nComputing residualized trade size (Eq 4.31)...")
    mask_both = df["ln_size"].notna() & df["ln_mktcap"].notna()
    if mask_both.sum() > 100:
        X_resid = df.loc[mask_both, "ln_mktcap"].values
        y_resid = df.loc[mask_both, "ln_size"].values
        coeffs = np.polyfit(X_resid, y_resid, 1)
        df["ln_size_resid"] = np.nan
        df.loc[mask_both, "ln_size_resid"] = (
            df.loc[mask_both, "ln_size"] - (coeffs[0] * df.loc[mask_both, "ln_mktcap"] + coeffs[1])
        )
        corr = df.loc[mask_both, "ln_size_resid"].corr(df.loc[mask_both, "ln_mktcap"])
        log(f"  ln(Size) = {coeffs[1]:.4f} + {coeffs[0]:.4f} * ln(MktCap)")
        log(f"  Residualized on {mask_both.sum():,} obs")
        log(f"  Correlation with ln_mktcap: {corr:.6f} (should be ~0)")
    else:
        df["ln_size_resid"] = np.nan
        log("  WARNING: Insufficient observations for residualization")

    # Save analysis sample
    df.to_csv(OUTPUT_PATH, index=False)
    log(f"\nSaved: {OUTPUT_PATH} ({len(df):,} rows, {len(df.columns)} columns)")

    # Flush remaining event study rows
    if event_study_rows:
        es_chunk = pd.DataFrame(event_study_rows)
        es_chunk.to_csv(
            OUTPUT_EVENT_STUDY,
            mode="a",
            header=not es_header_written,
            index=False,
        )
    # Count total ES rows
    if os.path.exists(OUTPUT_EVENT_STUDY):
        es_line_count = sum(1 for _ in open(OUTPUT_EVENT_STUDY)) - 1
        log(f"Saved: {OUTPUT_EVENT_STUDY} ({es_line_count:,} rows)")
    else:
        log(f"WARNING: {OUTPUT_EVENT_STUDY} was not created")

    # ==================================================================
    # 4F: SUMMARY DIAGNOSTICS
    # ==================================================================
    log(f"\n{'=' * 70}")
    log("STEP 4 SUMMARY")
    log(f"{'=' * 70}")

    log(f"\nFinal sample: {len(df):,} trades")

    log(f"\nMean Market-Adjusted AR by holding period:")
    for omega in HOLDING_PERIODS:
        col = f"AR_market_{omega}d"
        scol = f"AR_signed_market_{omega}d"
        n_v = df[col].notna().sum()
        mean_u = df[col].mean() if n_v > 0 else np.nan
        mean_s = df[scol].mean() if scol in df.columns and df[scol].notna().sum() > 0 else np.nan
        log(f"  {omega:>3}d: unsigned={mean_u:>9.5f}  signed={mean_s:>9.5f}  N={n_v:,}")

    log(f"\nMean Signed AR by model (20-day):")
    for model in ["market", "CAPM", "FF3", "Carhart", "FF5"]:
        col = f"AR_signed_{model}_20d"
        if col in df.columns:
            n_v = df[col].notna().sum()
            mean_v = df[col].mean() if n_v > 0 else np.nan
            log(f"  {model:<10}: {mean_v:>9.5f}  N={n_v:,}")

    log(f"\nCovariate coverage:")
    for var in ["ln_mktcap", "volatility", "ln_size", "ln_size_resid"]:
        n_v = df[var].notna().sum() if var in df.columns else 0
        log(f"  {var:<16}: {n_v:>6,} / {len(df):,}")
    if "n_est_days" in df.columns and df["n_est_days"].notna().sum() > 0:
        log(f"  n_est_days     : mean={df['n_est_days'].mean():.1f}, "
            f"min={df['n_est_days'].min()}, median={df['n_est_days'].median():.0f}")

    log(f"\nDual-anchor comparison (20d market-adjusted):")
    d_col = "AR_market_20d"
    t_col = "AR_market_trade_20d"
    if d_col in df.columns and t_col in df.columns:
        bv = df[d_col].notna() & df[t_col].notna()
        if bv.sum() > 0:
            log(f"  Disclosure-date mean: {df.loc[bv, d_col].mean():.5f}")
            log(f"  Trade-date mean:      {df.loc[bv, t_col].mean():.5f}")
            log(f"  N (both available):   {bv.sum():,}")

    log(f"\nAR availability by holding period:")
    for omega in HOLDING_PERIODS:
        n_mkt = df[f"AR_market_{omega}d"].notna().sum() if f"AR_market_{omega}d" in df.columns else 0
        n_ff5 = df[f"AR_FF5_{omega}d"].notna().sum() if f"AR_FF5_{omega}d" in df.columns else 0
        log(f"  {omega:>3}d: market={n_mkt:,}  FF5={n_ff5:,}")

    elapsed = time.time() - start_time
    log(f"\nTotal elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    with open(OUTPUT_DIAGNOSTICS, "w") as f:
        f.write("\n".join(LOG))
    log(f"Diagnostics saved to: {OUTPUT_DIAGNOSTICS}")


if __name__ == "__main__":
    main()
