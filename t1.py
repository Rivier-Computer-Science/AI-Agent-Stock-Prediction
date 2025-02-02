import logging
import openai
from openai import OpenAI

# Initialize logging
logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key="")

# OpenAI API Key
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

# Function to interact with ChatGPT for query enhancement
def chatgpt_agent(query):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI specializing in financial analysis."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message
    except Exception as e:
        logging.error(f"Error while interacting with ChatGPT: {e}")
        return "Error processing the query."

# Function to parse and categorize financial queries
def parse_query(query):
    try:
        query = query.lower()  # Normalize text
        indicators = {}

        # Identify key financial indicators
        if "inflation rate" in query:
            indicators["inflation_rate"] = True
        if "search trends" in query or "google trends" in query:
            indicators["search_trends"] = True
        if "fear and greed index" in query:
            indicators["fear_and_greed_index"] = True
        if "corporate announcements" in query or "company news" in query:
            indicators["corporate_announcements"] = True
        
        if not indicators:
            logging.error("No recognized indicators found in query.")
            return None
        
        logging.info(f"Indicators identified: {', '.join(indicators.keys())}")
        chatgpt_response = chatgpt_agent(query)  # Get insights from ChatGPT

        return {"indicators": indicators, "chatgpt_insights": chatgpt_response}
    except Exception as e:
        logging.error(f"Error while parsing query: {e}")
        return None

# Example usage
if __name__ == "__main__":
    queries = [
        "What’s the buying signal for Company X based on inflation rates and recent announcements?",
        "How are search trends affecting stock prices?",
        "What’s the current fear and greed index and its impact on the market?"
    ]
    
    for query in queries:
        parsed_data = parse_query(query)
        if parsed_data:
            print(f"Processed indicators: {parsed_data}")
