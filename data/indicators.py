"""
SAGE – Solana Analysis & Guidance Engine
indicators.py
---------------------------------------
Vectorised implementations of core technical indicators used by SAGE.
No TA‑Lib dependency; pure NumPy / pandas for portability.

All functions are *side‑effect‑free*: they return new Series/DataFrames and
never mutate the input.  Missing values are forward‑filled where sensible so
models don’t choke on NaNs.

Author: Senior Software Engineer (PhD Quant) – June 24 2025
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger_bands",
    "atr",
    "vwap",
    "add_all_indicators",
]

logger = logging.getLogger("SAGE.Indicators")

# ---------------------------------------------------------------------------
# Simple / Exponential Moving Averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average (SMA)."""
    return series.rolling(window, min_periods=window).mean().rename(f"SMA_{window}")


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (EMA)."""
    return series.ewm(span=span, adjust=False).mean().rename(f"EMA_{span}")

# ---------------------------------------------------------------------------
# Relative Strength Index (RSI)
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=window, adjust=False).mean()
    roll_down = down.ewm(span=window, adjust=False).mean()
    rs = roll_up / roll_down
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    return rsi_series.rename(f"RSI_{window}")

# ---------------------------------------------------------------------------
# Moving Average Convergence Divergence (MACD)
# ---------------------------------------------------------------------------

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    df = pd.DataFrame({
        f"MACD_{fast}_{slow}": macd_line,
        f"MACD_signal_{signal}": signal_line,
        "MACD_hist": hist,
    })
    return df

# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Return upper, middle, and lower Bollinger Bands."""
    mid = sma(series, window)
    std = series.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return pd.DataFrame({
        f"BB_upper_{window}": upper,
        f"BB_mid_{window}": mid,
        f"BB_lower_{window}": lower,
    })

# ---------------------------------------------------------------------------
# Average True Range (ATR)
# ---------------------------------------------------------------------------

def atr(ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (ATR).  Expects columns: high, low, close."""
    high = ohlc["high"].astype(float)
    low = ohlc["low"].astype(float)
    close = ohlc["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.ewm(span=window, adjust=False).mean()
    return atr_series.rename(f"ATR_{window}")

# ---------------------------------------------------------------------------
# Volume‑Weighted Average Price (VWAP)
# ---------------------------------------------------------------------------

def vwap(ohlc: pd.DataFrame) -> pd.Series:
    """Cumulative intraday VWAP. Requires columns: high, low, close, volume."""
    price = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3.0
    vol = ohlc["volume"].astype(float)
    cum_vol_price = (price * vol).cumsum()
    cum_vol = vol.cumsum()
    return (cum_vol_price / cum_vol).rename("VWAP")

# ---------------------------------------------------------------------------
# Convenience – add all core indicators to a DataFrame
# ---------------------------------------------------------------------------

def add_all_indicators(df: pd.DataFrame, *, price_col: str = "close") -> pd.DataFrame:
    """Return *df* plus a standard set of indicators for *price_col*.

    The function does **not** mutate *df*; a new DataFrame is returned.
    Missing OHLC columns for ATR/VWAP are ignored gracefully.
    """
    price = df[price_col].astype(float)

    indicators: List[pd.DataFrame | pd.Series] = [
        sma(price, 20),
        ema(price, 50),
        ema(price, 200),
        rsi(price, 14),
        macd(price),
        bollinger_bands(price),
    ]

    if {"high", "low", "close"}.issubset(df.columns):
        indicators.append(atr(df))
    if {"high", "low", "close", "volume"}.issubset(df.columns):
        indicators.append(vwap(df))

    combined = pd.concat([df.copy()] + indicators, axis=1)
    combined.fillna(method="ffill", inplace=True)
    combined.fillna(method="bfill", inplace=True)
    return combined
