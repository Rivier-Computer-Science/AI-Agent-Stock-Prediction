import logging

import requests

# Replace with your actual API keys
STOCKGEIST_API_KEY = 'stockgeist_api_key'
ALPHA_VANTAGE_API_KEY = 'alpha_vantage_api_key'

def fetch_market_sentiment(ticker):
    """
    Fetches market sentiment data for a specific stock using the StockGeist API.
    """
    try:
        url = f"https://api.stockgeist.ai/v1/sentiment/{ticker}"
        headers = {'Authorization': f'Bearer {STOCKGEIST_API_KEY}'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        logging.error(f"Error fetching market sentiment data: {e}")
        return None

def fetch_economic_indicators(indicator):
    """
    Fetches economic indicators such as inflation rates using the Alpha Vantage API.
    """
    try:
        url = f"https://www.alphavantage.co/query?function={indicator}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        logging.error(f"Error fetching economic indicators: {e}")
        return None
