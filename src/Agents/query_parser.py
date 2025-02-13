import logging

from src.Agents.chatgpt_agent import chatgpt_agent

# System prompt to instruct GPT on how to extract financial indicators
PARSING_PROMPT = """
You are an AI that specializes in financial data extraction. 
Given a user query, extract and categorize financial indicators such as:
- Inflation rates
- Search engine trends
- Fear and Greed Index
- Corporate announcements

Return a structured JSON response in the following format:
{
  "indicators": ["inflation_rate", "search_trends", "fear_and_greed_index", "corporate_announcements"],
  "explanation": "Brief explanation of the identified indicators."
}
If no indicators are found, return an empty list in "indicators".
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
        
        # Convert GPT response into a dictionary
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
