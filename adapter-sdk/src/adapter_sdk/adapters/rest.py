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

    def read(self) -> pd.DataFrame:
        """Fetch live financial news and normalize it to our (text, label) table."""
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        response = requests.get(
            API_URL,
            params={
                "function": "NEWS_SENTIMENT",
                "topics": "financial_markets",
                "limit": "50",
                "apikey": api_key,
            },
            timeout=30,
        )
        data = response.json()
        df = pd.DataFrame(data["feed"])[["title", "overall_sentiment_label"]]
        df = df.rename(columns={"title": "text", "overall_sentiment_label": "label"})
        df["label"] = df["label"].map(LABEL_MAP)
        return df
