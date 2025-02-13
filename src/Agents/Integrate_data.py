import logging

from src.Agents.fetch_data import (fetch_economic_indicators,
                                   fetch_market_sentiment)
from src.Agents.validate_map import validate_and_map_data


def integrate_data(ticker, indicator):
    """
    Integrates data by fetching and validating market sentiment and economic indicators.
    """
    market_sentiment = fetch_market_sentiment(ticker)
    economic_indicators = fetch_economic_indicators(indicator)

    if market_sentiment and economic_indicators:
        raw_data = {
            "market_sentiment": market_sentiment,
            "economic_indicators": economic_indicators
        }
        mapped_data = validate_and_map_data(raw_data)
        return mapped_data
    else:
        logging.error("Failed to fetch one or more data sources.")
        return None
