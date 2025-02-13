import logging

import openai
from src.config import OPENAI_API_KEY

# Initialize OpenAI API Key
openai.api_key = OPENAI_API_KEY

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
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"Error while interacting with ChatGPT: {e}")
        return "Error processing the query."
