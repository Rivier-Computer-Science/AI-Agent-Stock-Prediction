import logging

import openai

# System prompt for ChatGPT
MAPPING_PROMPT = """
You are an AI that specializes in financial data mapping and validation.
Given raw financial and economic data, validate the data, identify anomalies, 
and map it to a standardized model for predictive analysis. Return a structured JSON response.
"""

def chatgpt_agent(query, prompt):
    """
    Interacts with ChatGPT for data validation and mapping.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"Error while interacting with ChatGPT: {e}")
        return None

def validate_and_map_data(raw_data):
    """
    Validates and maps raw data using ChatGPT.
    """
    try:
        formatted_query = f"Validate and map the following data: {raw_data}"
        response = chatgpt_agent(formatted_query, MAPPING_PROMPT)
        import json
        mapped_data = json.loads(response)
        return mapped_data
    except Exception as e:
        logging.error(f"Error validating and mapping data: {e}")
        return None
