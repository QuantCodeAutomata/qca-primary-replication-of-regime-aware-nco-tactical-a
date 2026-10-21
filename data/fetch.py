"""data/fetch.py — Data acquisition and feature engineering.

Downloads daily prices for the five Indonesian multi-asset universe via
Yahoo Finance, computes monthly log returns, and builds the four-dimensional
HMM feature matrix used for regime detection.

Asset universe
--------------
LQ45     : Indonesian equity index (^JKLQ45)
BTC_IDR  : Bitcoin / Indonesian Rupiah (BTC-IDR)
CNY_IDR  : Chinese Yuan / Indonesian Rupiah FX (CNYIDR=X)
Gold_IDR : Gold in IDR  =  XAU/USD (GC=F)  ×  USD/IDR (USDIDR=X)
SBN      : Indonesian government bond proxy (EIDO)
"""

import os
import logging

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Module-level logger — callers configure handlers; we attach none here.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker map: short name  →  Yahoo Finance symbol
# ---------------------------------------------------------------------------
TICKERS: dict[str, str] = {
    'LQ45':      '^JKLQ45',
    'BTCIDR':    'BTC-IDR',    # direct; may be unavailable → fallback below
    'BTCUSD':    'BTC-USD',    # fallback source for BTC in IDR
    'CNYIDR':    'CNYIDR=X',
    'XAUUSD':    'GC=F',
    'USDIDR':    'USDIDR=X',
    'SBN_PROXY': 'EIDO',
}

# Canonical output column names (order matters for downstream code)
_OUTPUT_COLS = ['LQ45', 'BTC_IDR', 'CNY_IDR', 'Gold_IDR', 'SBN']


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_daily_prices(start: str = '2019-01-02', end: str = '2025-08-31') -> pd.DataFrame:
    """Download and clean daily closing prices for the five-asset universe.

    Steps
    -----
    1. Download all tickers from Yahoo Finance with ``yf.download()``.
    2. Extract ``Close`` prices (MultiIndex column level).
    3. Compute ``Gold_IDR = XAUUSD × USDIDR``.
    4. Retain the five final asset columns and rename them.
    5. Forward-fill missing values up to 3 consecutive days to bridge
       differing market holidays across exchanges.
    6. Drop rows where **every** column is NaN.

    Parameters
    ----------
    start:
        Inclusive start date string, e.g. ``'2019-01-02'``.
    end:
        Inclusive end date string, e.g. ``'2025-08-31'``.

    Returns
    -------
    pd.DataFrame
        Daily closing prices with columns
        ``['LQ45', 'BTC_IDR', 'CNY_IDR', 'Gold_IDR', 'SBN']``
        and a ``DatetimeIndex``.

    Raises
    ------
    ValueError
        If yfinance returns an empty result or required columns are absent.
    """
    ticker_symbols = list(TICKERS.values())
    logger.info("Downloading %d tickers from Yahoo Finance: %s", len(ticker_symbols), ticker_symbols)

    try:
        raw = yf.download(
            ticker_symbols,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("yfinance download raised an exception: %s", exc)
        raise

    if raw.empty:
        raise ValueError(
            "yfinance returned an empty DataFrame. "
            "Verify ticker symbols and that the date range contains trading days."
        )

    # ---- Extract Close prices -------------------------------------------
    # yf.download with multiple tickers returns MultiIndex columns:
    # level-0 = price field ('Close', 'Open', …), level-1 = ticker symbol.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close'].copy()
    else:
        # Defensive fallback — should not occur with six tickers
        close = raw[['Close']].copy()

    # ---- Rename ticker symbols → short names ----------------------------
    ticker_to_name = {symbol: name for name, symbol in TICKERS.items()}
    close = close.rename(columns=ticker_to_name)

    # Guard: warn if any expected raw column is missing
    missing_raw = set(TICKERS.keys()) - set(close.columns)
    if missing_raw:
        logger.warning(
            "The following columns are missing after download and will be NaN: %s", missing_raw
        )
        for col in missing_raw:
            close[col] = np.nan

    # ---- BTC in IDR: prefer direct BTC-IDR; fall back to BTC-USD × USD/IDR --
    # 'BTC-IDR' is sporadically unavailable on Yahoo Finance (HTTP 404).
    btcidr_missing = close.get('BTCIDR', pd.Series(dtype=float)).isna().all()
    if btcidr_missing:
        btcusd = close.get('BTCUSD', pd.Series(dtype=float))
        usdidr = close.get('USDIDR', pd.Series(dtype=float))
        if not btcusd.isna().all() and not usdidr.isna().all():
            close['BTCIDR'] = btcusd * usdidr
            logger.info("BTC_IDR: 'BTC-IDR' unavailable; using BTC-USD × USD/IDR fallback")
        else:
            logger.warning("BTC_IDR: both direct and fallback sources unavailable; column will be NaN")
            close['BTCIDR'] = np.nan

    # ---- Compute Gold in IDR: XAU/USD × USD/IDR -------------------------
    close['Gold_IDR'] = close['XAUUSD'] * close['USDIDR']

    # ---- Select and rename to canonical output columns ------------------
    close = close[['LQ45', 'BTCIDR', 'CNYIDR', 'Gold_IDR', 'SBN_PROXY']].copy()
    close.columns = _OUTPUT_COLS

    # ---- Forward-fill gaps of up to 3 consecutive trading days ----------
    # Handles cases where one exchange is closed (e.g. Indonesian public
    # holidays) but others trade normally.
    close = close.ffill(limit=3)

    # ---- Drop rows where every asset price is NaN -----------------------
    close = close.dropna(how='all')

    logger.info(
        "Daily prices ready — shape: %s | range: %s → %s",
        close.shape,
        close.index[0].date(),
        close.index[-1].date(),
    )
    return close


def compute_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end and compute log returns.

    Parameters
    ----------
    daily_prices:
        Daily closing prices with a ``DatetimeIndex`` (output of
        :func:`fetch_daily_prices`).

    Returns
    -------
    pd.DataFrame
        Monthly log returns with the same column names as ``daily_prices``.
        The first row (which would be NaN after the log-return shift) is
        dropped, so the returned index starts one month after the first
        available price.
    """
    # Sample the last available closing price within each calendar month
    monthly_prices = daily_prices.resample('ME').last()

    # Log returns: ln(P_t / P_{t-1})
    log_returns = np.log(monthly_prices / monthly_prices.shift(1))

    # Drop the first row — it is NaN because there is no prior month price
    log_returns = log_returns.iloc[1:]

    # Fill any remaining NaN with 0: a missing monthly price is treated as
    # no observed change (conservative; avoids NaN propagation into HMM features)
    n_nan = log_returns.isna().sum().sum()
    if n_nan > 0:
        logger.warning(
            "compute_monthly_returns: filling %d residual NaN cells with 0.0 "
            "(missing price data treated as zero return).",
            n_nan,
        )
    return log_returns.fillna(0.0)


def build_hmm_features(monthly_returns: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Build the four-dimensional HMM feature matrix using a rolling window.

    For each month *t* (starting at index = ``window``, 0-based), the
    trailing window ``[t - window + 1, t]`` (inclusive, length = ``window``)
    is used to compute:

    =========  =================================================================
    Feature    Definition
    =========  =================================================================
    mean_eq    Mean LQ45 log return over the trailing ``window`` months
    vol_eq     Sample standard deviation of LQ45 log returns (``ddof=1``)
    mean_bond  Mean SBN log return over the trailing ``window`` months
    delta_fx   CNY_IDR log return at month *t* (point observation, not a mean)
    =========  =================================================================

    Parameters
    ----------
    monthly_returns:
        Monthly log returns with columns ``['LQ45', 'SBN', 'CNY_IDR', …]``
        (output of :func:`compute_monthly_returns`).
    window:
        Rolling window length in months (default 24).

    Returns
    -------
    pd.DataFrame
        Feature matrix with columns
        ``['mean_eq', 'vol_eq', 'mean_bond', 'delta_fx']``,
        indexed by the month-end dates of ``monthly_returns`` starting at
        position ``window`` (0-based index).
    """
    rows: list[list[float]] = []
    idx: list[pd.Timestamp] = []

    for t in range(window, len(monthly_returns)):
        # Trailing slice: months [t-window+1 … t] inclusive → length = window
        win = monthly_returns.iloc[t - window + 1: t + 1]

        mean_eq   = float(win['LQ45'].mean())
        vol_eq    = float(win['LQ45'].std(ddof=1))
        mean_bond = float(win['SBN'].mean())
        # delta_fx is the single CNY_IDR observation at month t,
        # NOT the rolling mean — it captures current FX momentum
        delta_fx  = float(monthly_returns['CNY_IDR'].iloc[t])

        rows.append([mean_eq, vol_eq, mean_bond, delta_fx])
        idx.append(monthly_returns.index[t])

    return pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(idx),
        columns=['mean_eq', 'vol_eq', 'mean_bond', 'delta_fx'],
    )


def load_data(
    start: str = '2018-12-01',
    end: str = '2025-08-31',
    window: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Orchestrate the full data pipeline for the Regime-Aware NCO system.

    Calls :func:`fetch_daily_prices`, :func:`compute_monthly_returns`, and
    :func:`build_hmm_features` in sequence and logs key diagnostics.

    Parameters
    ----------
    start:
        Start date for price download.
    end:
        End date for price download.
    window:
        Rolling window length in months for HMM feature construction.

    Returns
    -------
    tuple of (daily_prices, monthly_returns, hmm_features)
        All three are ``pd.DataFrame`` instances.

    Raises
    ------
    RuntimeError
        If data cannot be fetched or the resulting series are too short for
        the requested window length.
    """
    logger.info(
        "=== load_data | range: %s → %s | HMM window: %d months ===",
        start, end, window,
    )

    # ------------------------------------------------------------------
    # Step 1: daily prices
    # ------------------------------------------------------------------
    try:
        daily_prices = fetch_daily_prices(start=start, end=end)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch daily prices: {exc}") from exc

    logger.info(
        "Daily prices  — shape: %s | %s → %s",
        daily_prices.shape,
        daily_prices.index[0].date(),
        daily_prices.index[-1].date(),
    )

    # ------------------------------------------------------------------
    # Step 2: monthly log returns
    # ------------------------------------------------------------------
    monthly_returns = compute_monthly_returns(daily_prices)

    if len(monthly_returns) <= window:
        raise RuntimeError(
            f"Insufficient monthly data: {len(monthly_returns)} months available, "
            f"but window={window} requires at least {window + 1} months."
        )

    logger.info(
        "Monthly returns — shape: %s | %s → %s | cols: %s",
        monthly_returns.shape,
        monthly_returns.index[0].date(),
        monthly_returns.index[-1].date(),
        list(monthly_returns.columns),
    )

    # ------------------------------------------------------------------
    # Step 3: HMM features
    # ------------------------------------------------------------------
    hmm_features = build_hmm_features(monthly_returns, window=window)

    logger.info(
        "HMM features  — shape: %s | %s → %s | cols: %s",
        hmm_features.shape,
        hmm_features.index[0].date(),
        hmm_features.index[-1].date(),
        list(hmm_features.columns),
    )

    return daily_prices, monthly_returns, hmm_features


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    daily, monthly, features = load_data()

    sep = '-' * 60

    print(f"\n{sep}")
    print("DAILY PRICES")
    print(sep)
    print(f"  Shape : {daily.shape}")
    print(f"  Range : {daily.index[0].date()}  →  {daily.index[-1].date()}")
    print(f"  Cols  : {list(daily.columns)}")
    print(f"  NaNs  :\n{daily.isna().sum().to_string()}")
    print("\n  Last 5 rows:")
    print(daily.tail(5).to_string())

    print(f"\n{sep}")
    print("MONTHLY LOG RETURNS")
    print(sep)
    print(f"  Shape : {monthly.shape}")
    print(f"  Range : {monthly.index[0].date()}  →  {monthly.index[-1].date()}")
    print("\n  Descriptive statistics:")
    print(monthly.describe().to_string())
    print("\n  Last 5 rows:")
    print(monthly.tail(5).to_string())

    print(f"\n{sep}")
    print("HMM FEATURES")
    print(sep)
    print(f"  Shape : {features.shape}")
    print(f"  Range : {features.index[0].date()}  →  {features.index[-1].date()}")
    print("\n  Descriptive statistics:")
    print(features.describe().to_string())
    print("\n  Last 5 rows:")
    print(features.tail(5).to_string())

    print(f"\n{sep}")
    print("DATA PIPELINE COMPLETE")
    print(sep)
