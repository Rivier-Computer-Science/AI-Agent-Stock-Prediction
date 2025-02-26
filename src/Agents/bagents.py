import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API Key
client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

def chatgpt_agent(query, system_prompt):
    """
    Sends a query to ChatGPT with a specific system instruction.
    
    Args:
        query (str): The user query.
        system_prompt (str): The system prompt to guide GPT's behavior.

    Returns:
        str: GPT's response.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error while interacting with ChatGPT: {e}")
        return "Error processing the query."

import logging

import requests

# Replace with your actual API keys
STOCKGEIST_API_KEY = 'MpKqXYZ4gKY43teezeTfhTdYRPeHLiZE'
ALPHA_VANTAGE_API_KEY = 'E93YOVT9315U16U3'

def fetch_market_sentiment(ticker, location="us", sources="twitter,reddit,stocktwits", metrics="pos_total_count,neu_total_count,neg_total_count", timeframe="5m"):
    """
    Fetches real-time sentiment metrics for a given stock ticker from StockGeist API.

    :param ticker: Stock symbol (e.g., "AAPL" for Apple).
    :param location: Market location ('us' for US stocks).
    :param sources: Comma-separated sources (twitter, reddit, stocktwits).
    :param metrics: Comma-separated metrics (e.g., pos_total_count, neg_total_count).
    :param timeframe: Timeframe for sentiment data ('1m', '5m', '1h', '1d').
    :return: JSON response with sentiment data or None if an error occurs.
    """
    url = f"https://api.stockgeist.ai/stock/{location}/stream/message-metrics"
    headers = {"Authorization": f"Bearer {STOCKGEIST_API_KEY}"}
    params = {
        "symbols": ticker,
        "sources": sources,
        "metrics": metrics,
        "timeframe": timeframe
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching sentiment data for {ticker}: {e}")
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

import logging


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

import logging

# System prompt to instruct GPT on how to extract financial indicators
PARSING_PROMPT = """
You are a financial AI specializing in data extraction.
Your task is to **ONLY identify and return** the financial indicators mentioned in the query. 

### **Instructions:**
- **DO NOT** generate or assume any values, numbers, companies, or trends.
- Only extract indicator **names**, nothing else.
- Return a JSON list with just the indicator names.

### **Expected JSON Format:**
{
  "indicators": ["inflation_rate", "search_trends", "corporate_announcements"],
  "explanation": "Indicators mentioned in the query."
}

### **Example 1:**
**User Query:** "Give me insights on Apple stock based on inflation and search trends."

**AI Response:**
{
  "indicators": ["inflation_rate", "search_trends"],
  "explanation": "The query mentions inflation and search trends."
}

### **Example 2:**
**User Query:** "What’s the effect of the Fear and Greed Index on the stock market?"

**AI Response:**
{
  "indicators": ["fear_and_greed_index"],
  "explanation": "The query explicitly refers to the Fear and Greed Index."
}

If **no indicators** are found, return:
{
  "indicators": [],
  "explanation": "No relevant indicators found in the query."
}
"""



def parse_query(query):
    """
    Uses GPT to parse financial indicators from a user query.

    Args:
        query (str): The user's financial query.

    Returns:
        dict: Parsed financial indicators and explanation.
    """
    try:
        response = chatgpt_agent(query, PARSING_PROMPT)
        
        import json
        parsed_data = json.loads(response)

        if not parsed_data.get("indicators"):
            logging.warning("No financial indicators identified in query.")
            return None

        logging.info(f"Parsed indicators: {parsed_data['indicators']}")
        return parsed_data
    except Exception as e:
        logging.error(f"Error parsing query: {e}")
        return None


import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API Key
client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

# System prompt for ChatGPT
MAPPING_PROMPT = """
You are an AI specializing in financial data validation.
Your task is to **ONLY map given data** to the correct financial indicators **WITHOUT** modifying or creating new values.

### **Instructions:**
- Validate the data structure.
- Ensure financial indicators match their expected format.
- **DO NOT** generate new numbers, trends, or insights.
- Return the structured data as provided, ensuring correct mapping.

### **Expected JSON Format:**
{
  "validated_data": {
    "inflation_rate": {...},  
    "search_trends": {...},  
    "corporate_announcements": {...}  
  }
}

If a key is missing or invalid, return:
{
  "error": "Missing or invalid data for some indicators."
}
"""


def chatgpt_agent(query, prompt):
    """
    Interacts with ChatGPT for data validation and mapping.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error while interacting with ChatGPT: {e}")
        return None

def validate_and_map_data(raw_data):
    """
    Validates and maps raw data using ChatGPT.
    Ensures that no new data is generated by AI.

    Args:
        raw_data (dict): The raw extracted financial data.

    Returns:
        dict: The validated and mapped financial data.
    """
    try:
        formatted_query = f"Validate and map the following data WITHOUT modifying: {raw_data}"
        response = chatgpt_agent(formatted_query, MAPPING_PROMPT)
        
        import json
        mapped_data = json.loads(response)
        
        if "error" in mapped_data:
            logging.error(f"Data validation error: {mapped_data['error']}")
            return None
        
        return mapped_data

    except Exception as e:
        logging.error(f"Error validating and mapping data: {e}")
        return None

