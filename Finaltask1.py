# Import necessary libraries
import re
import json
import openai  # Ensure OpenAI library is installed
import logging
from openai import OpenAI

# Initialize logging
logging.basicConfig(level=logging.INFO)

# OpenAI API Key
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

client = OpenAI(api_key="")

class ScenarioInputAgent:
    def __init__(self):
        self.tax_keywords = ["tax", "taxation", "liability", "optimization"]
        self.signal_keywords = ["buy", "sell", "hold", "trade", "signal"]
        self.testing_keywords = ["backtest", "forward test", "strategy validation"]

    def process_input(self, user_input):
        """
        Parse user input and route the request to the appropriate agent based on keywords.
        """
        cleaned_input = self.clean_input(user_input)
        
        # Identify the type of query based on keywords
        if self.contains_tax_keywords(cleaned_input):
            return self.route_to_tax_agent(cleaned_input)
        elif self.contains_signal_keywords(cleaned_input):
            return self.route_to_signal_agent(cleaned_input)
        elif self.contains_testing_keywords(cleaned_input):
            return self.route_to_testing_agent(cleaned_input)
        else:
            return self.handle_unknown_input()

    def chatgpt_agent(self, query, context):
        """
        Interact with ChatGPT to refine and enhance the query.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message
        except Exception as e:
            logging.error(f"Error while interacting with ChatGPT: {e}")
            return "Error processing the query."

    def clean_input(self, user_input):
        """
        Clean the user input by removing unnecessary spaces and converting to lowercase.
        """
        return user_input.strip().lower()

    def contains_tax_keywords(self, cleaned_input):
        return any(keyword in cleaned_input for keyword in self.tax_keywords)

    def contains_signal_keywords(self, cleaned_input):
        return any(keyword in cleaned_input for keyword in self.signal_keywords)

    def contains_testing_keywords(self, cleaned_input):
        return any(keyword in cleaned_input for keyword in self.testing_keywords)

    def route_to_tax_agent(self, cleaned_input):
        logging.info("Routing to Tax Rules Agent...")
        tax_data = self.extract_tax_data(cleaned_input)
        tax_data["chatgpt_insights"] = self.chatgpt_agent(cleaned_input, "Provide insights on tax implications for investments.")
        return tax_data

    def route_to_signal_agent(self, cleaned_input):
        logging.info("Routing to Signal Generation Agent...")
        signal_data = self.extract_signal_data(cleaned_input)
        signal_data["chatgpt_insights"] = self.chatgpt_agent(cleaned_input, "Generate AI-driven stock trading signals.")
        return signal_data

    def route_to_testing_agent(self, cleaned_input):
        logging.info("Routing to Testing Agent...")
        testing_data = self.extract_testing_data(cleaned_input)
        testing_data["chatgpt_insights"] = self.chatgpt_agent(cleaned_input, "Validate trading strategies through backtesting and forward testing.")
        return testing_data

    def handle_unknown_input(self):
        logging.warning("Unrecognized input. Please provide a valid query.")
        return {"error": "Unrecognized input"}

    def extract_tax_data(self, cleaned_input):
        jurisdiction = self.extract_jurisdiction(cleaned_input)
        return {"action": "tax_analysis", "jurisdiction": jurisdiction}

    def extract_signal_data(self, cleaned_input):
        action = self.extract_action(cleaned_input)
        return {"action": action}

    def extract_testing_data(self, cleaned_input):
        test_type = self.extract_test_type(cleaned_input)
        return {"test_type": test_type}

    def extract_jurisdiction(self, cleaned_input):
        if "us" in cleaned_input:
            return "US"
        elif "uk" in cleaned_input:
            return "UK"
        else:
            return "Global"

    def extract_action(self, cleaned_input):
        if "buy" in cleaned_input:
            return "buy"
        elif "sell" in cleaned_input:
            return "sell"
        elif "hold" in cleaned_input:
            return "hold"
        else:
            return "unknown"

    def extract_test_type(self, cleaned_input):
        if "backtest" in cleaned_input:
            return "backtest"
        elif "forward test" in cleaned_input:
            return "forward test"
        else:
            return "unknown"

# Sample usage
scenario_input_agent = ScenarioInputAgent()

user_query = "What are the tax implications for my investment in the US?"
result = scenario_input_agent.process_input(user_query)
print(result)

user_query2 = "Generate a buy signal for Apple based on RSI and MACD"
result2 = scenario_input_agent.process_input(user_query2)
print(result2)

user_query3 = "Backtest my strategy using historical data"
result3 = scenario_input_agent.process_input(user_query3)
print(result3)
