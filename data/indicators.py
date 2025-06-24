from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Final

import ccxt
import numpy as np
import pandas as pd

__all__ = [
    "fetch_ohlcv",
    "compute_indicators",
    "interpret_latest",
    "compute_score",          # NEW
    "fetch_and_interpret",
]

# ---------------------------------------------------------------------------
# Configuration & logger
# ---------------------------------------------------------------------------

_LOG_LEVEL: Final = "INFO"
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
)
logger = logging.getLogger("SAGE.Indicators")

EXCHANGE: Final = "binance"
PAIR: Final = "SOL/USDT"
TIMEFRAME: Final = "1m"
LIMIT: Final = 500

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0.0, delta, 0.0)
    dn = np.where(delta < 0.0, -delta, 0.0)

    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_dn = pd.Series(dn, index=series.index).rolling(period).mean()

    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# 1 · Fetch OHLCV
# ---------------------------------------------------------------------------


def fetch_ohlcv(
    exchange: str = EXCHANGE,
    pair: str = PAIR,
    timeframe: str = TIMEFRAME,
    limit: int = LIMIT,
) -> pd.DataFrame:
    client = getattr(ccxt, exchange)({"enableRateLimit": True})
    raw = client.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df.astype(float)


# ---------------------------------------------------------------------------
# 2 · Compute indicators
# ---------------------------------------------------------------------------


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame empty – cannot compute indicators")

    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()

    df["ema12"] = _ema(df["close"], 12)
    df["ema26"] = _ema(df["close"], 26)

    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["rsi14"] = _rsi(df["close"], 14)

    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_percent"] = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    )

    vol_mean = df["volume"].rolling(30).mean()
    vol_std = df["volume"].rolling(30).std()
    df["vol_z"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# 3 · Interpret latest candle → feature vector
# ---------------------------------------------------------------------------


def interpret_latest(df: pd.DataFrame) -> Dict[str, float]:
    row = df.iloc[-1]

    rsi_state = 1.0 if row["rsi14"] > 70 else (-1.0 if row["rsi14"] < 30 else 0.0)
    macd_state = 1.0 if row["macd"] > row["macd_signal"] else -1.0
    price_sma_state = 1.0 if row["close"] > row["sma20"] else -1.0

    return {
        "price": row["close"],
        "rsi14": row["rsi14"],
        "macd_hist": row["macd_hist"],
        "bb_percent": row["bb_percent"],
        "vol_z": row["vol_z"],
        "rsi_state": rsi_state,
        "macd_state": macd_state,
        "price_vs_sma20": price_sma_state,
        "timestamp": row.name.timestamp(),
    }


# 3.5 · Composite score  (NEW)
def compute_score(feats: Dict[str, float]) -> float:
    """
    Simple sentiment score in [-1, +1].

    Current definition = mean of the three discrete states.
    Adapt/weight as you refine the edge.
    """
    states = [
        feats.get("rsi_state", 0.0),
        feats.get("macd_state", 0.0),
        feats.get("price_vs_sma20", 0.0),
    ]
    return float(sum(states)) / len(states)


# Convenience wrapper
def fetch_and_interpret(
    timeframe: str = TIMEFRAME,
    limit: int = LIMIT,
) -> Dict[str, float]:
    df = fetch_ohlcv(timeframe=timeframe, limit=limit)
    df = compute_indicators(df)
    return interpret_latest(df)


# ---------------------------------------------------------------------------
# 4 · CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    cli = argparse.ArgumentParser(description="SAGE – Solana indicators")
    cli.add_argument("--timeframe", default=TIMEFRAME, help="ccxt timeframe (e.g. 1m)")
    cli.add_argument("--limit", type=int, default=LIMIT, help="candles to fetch")
    cli.add_argument(
        "--json-min",
        action="store_true",
        help="print minified JSON (good for piping)",
    )
    args = cli.parse_args()

    feats = fetch_and_interpret(timeframe=args.timeframe, limit=args.limit)
    score = compute_score(feats)               # NEW
    logger.info("Composite signal score = %.3f", score)   # NEW

    if args.json_min:
        print(json.dumps(feats, separators=(",", ":")))
    else:
        print(json.dumps(feats, indent=2, default=float))


if __name__ == "__main__":
    _cli()
