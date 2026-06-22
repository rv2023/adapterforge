"""RestAPIAdapter — pulls LIVE financial news from the Alpha Vantage API.

The third source, and the messy/real one. read() makes an HTTP request, gets
JSON back, and BUILDS our standard (text, label) table from it. The API key is
read from the environment (loaded from .env) — never hardcoded.
"""

import os

import pandas as pd
import requests
from dotenv import load_dotenv

from adapter_sdk.base import BaseAdapter
from adapter_sdk.schemas.v1 import schema_v1

# Load .env so os.getenv can see ALPHAVANTAGE_API_KEY.
load_dotenv()

API_URL = "https://www.alphavantage.co/query"

# Alpha Vantage's 5 sentiment words -> our 3-word vocabulary.
LABEL_MAP = {
    "Bullish": "bullish",
    "Somewhat-Bullish": "bullish",
    "Neutral": "neutral",
    "Somewhat-Bearish": "bearish",
    "Bearish": "bearish",
}


class RestAPIAdapter(BaseAdapter):
    """Adapter for live financial news (Alpha Vantage NEWS_SENTIMENT endpoint)."""

    schema = schema_v1
    name = "alphavantage_news"

    def __init__(
        self,
        topics: str | None = None,
        tickers: str | None = None,
        time_from: str | None = None,
        time_to: str | None = None,
        limit: int = 50,
    ) -> None:
        self.topics = topics
        self.tickers = tickers
        self.time_from = time_from
        self.time_to = time_to
        self.limit = limit

    def read(self) -> pd.DataFrame:
        """Fetch live financial news and normalize it to our (text, label) table."""
        if not self.topics and not self.tickers:
            raise ValueError("RestAPIAdapter requires at least one of topics or tickers")

        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        params = {
            "function": "NEWS_SENTIMENT",
            "limit": str(self.limit),
            "apikey": api_key,
        }
        if self.topics:
            params["topics"] = self.topics
        if self.tickers:
            params["tickers"] = self.tickers
        if self.time_from:
            params["time_from"] = self.time_from
        if self.time_to:
            params["time_to"] = self.time_to

        response = requests.get(
            API_URL,
            params=params,
            timeout=30,
        )
        data = response.json()
        df = pd.DataFrame(data["feed"])[["title", "overall_sentiment_label"]]
        df = df.rename(columns={"title": "text", "overall_sentiment_label": "label"})
        df["label"] = df["label"].map(LABEL_MAP)
        return df
