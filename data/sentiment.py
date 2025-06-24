from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import yfinance as yf
from transformers import TextClassificationPipeline, pipeline

__all__ = [
    "NewsItem",
    "SentimentAnalyzer",
    "fetch_news",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_LOG_LEVEL = os.getenv("SAGE_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=_LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s | %(message)s")
logger = logging.getLogger("SAGE.Sentiment")

_HF_MODEL_NAME = os.getenv("SAGE_FINBERT_MODEL", "ProsusAI/finbert")
_DEVICE = 0 if os.getenv("SAGE_USE_GPU", "0") == "1" else -1  # –1 → CPU

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NewsItem:
    """Represents one news headline pulled from *yfinance*."""

    title: str
    summary: str
    link: str
    provider: str
    published_at: datetime

    def to_dict(self) -> Dict[str, Union[str, int]]:
        data = asdict(self)
        data["published_at"] = int(self.published_at.timestamp())
        return data


# ---------------------------------------------------------------------------
# FinBERT wrapper
# ---------------------------------------------------------------------------
class _FinBERT:
    """Lazy‑loads and wraps the FinBERT sentiment model."""

    def __init__(self, model_name: str = _HF_MODEL_NAME, device: int = _DEVICE) -> None:
        logger.debug("Initializing FinBERT (model=%s, device=%s)", model_name, device)
        self._pipeline: TextClassificationPipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
            device=device,
        )

    def score(self, text: str) -> Dict[str, float]:
        """Return a dict with *positive*, *neutral*, *negative* probabilities."""
        outputs = self._pipeline(text, truncation=True)[0]
        probs = {d["label"].lower(): d["score"] for d in outputs}
        return {
            "positive": float(probs.get("positive", 0.0)),
            "neutral": float(probs.get("neutral", 0.0)),
            "negative": float(probs.get("negative", 0.0)),
        }


# Singleton instance
a_finbert: Optional[_FinBERT] = None

def _get_finbert() -> _FinBERT:
    global a_finbert
    if a_finbert is None:
        a_finbert = _FinBERT()
    return a_finbert

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(d: Dict, *keys, default="") -> str:
    cur: Union[Dict, str, None] = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if isinstance(cur, str) else default


def _parse_pubdate(value: Union[str, int, float]) -> Optional[datetime]:
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception as exc:
        logger.warning("Failed to parse pubDate %s: %s", value, exc)
    return None

# ---------------------------------------------------------------------------
# Public – fetch & analyse
# ---------------------------------------------------------------------------

def fetch_news(
    ticker: str = "SOL-USD",
    max_items: int = 30,
    lookback_hours: int = 24,
) -> List[NewsItem]:
    yf_ticker = yf.Ticker(ticker)
    raw_news: List[Dict[str, any]] = yf_ticker.news  # type: ignore[attr-defined]
    logger.debug("Fetched %d raw news articles from yfinance", len(raw_news))

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    items: List[NewsItem] = []

    for raw in raw_news[:max_items]:
        art = raw.get("content", raw)
        ts = _parse_pubdate(art.get("pubDate", raw.get("pubDate")))
        if ts is None or ts < cutoff:
            continue

        title = art.get("title", raw.get("title", ""))
        summary = art.get("summary", art.get("description", raw.get("summary", "")))
        link = _safe_get(art, "canonicalUrl", "url") or art.get("previewUrl", raw.get("link", ""))
        provider = _safe_get(art, "provider", "displayName") or raw.get("publisher", "")

        items.append(
            NewsItem(title=title, summary=summary, link=link, provider=provider, published_at=ts)
        )

    logger.info("Returning %d news articles", len(items))
    return items


class SentimentAnalyzer:
    def __init__(self) -> None:
        self._model = _get_finbert()

    @staticmethod
    def _txt(item: NewsItem) -> str:
        return f"{item.title}. {item.summary}" if item.summary else item.title

    def score_items(self, items: Iterable[NewsItem]) -> List[Dict[str, float]]:
        return [self._model.score(self._txt(i)) for i in items]

    def aggregate(self, items: List[NewsItem]) -> float:
        if not items:
            logger.warning("aggregate() empty – returning 0.0")
            return 0.0
        scores = self.score_items(items)
        pos = np.array([s["positive"] for s in scores])
        neg = np.array([s["negative"] for s in scores])
        return float(pos.mean() - neg.mean())

# ---------------------------------------------------------------------------
# CLI – outputs ONLY the net sentiment score
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="SAGE sentiment – print net score")
    cli.add_argument("--ticker", default="SOL-USD")
    cli.add_argument("--hours", type=int, default=24)
    cli.add_argument("--max", type=int, default=30)
    args = cli.parse_args()

    news_items = fetch_news(args.ticker, max_items=args.max, lookback_hours=args.hours)
    score = SentimentAnalyzer().aggregate(news_items)

    # Print just the scalar (rounded to 4 decimals) so scripts can pipe/parse it easily
    print(f"{score:+0.4f}")
