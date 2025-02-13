# src/Agents/PortfolioDataAgent.py

import openai

# OpenAI API Key (Ensure it's securely stored in environment variables)
OPENAI_API_KEY = "OPENAI_API_KEY"

def chatgpt_query(prompt: str) -> str:
    """Fetches a response from OpenAI's ChatGPT API."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        api_key=OPENAI_API_KEY
    )
    return response["choices"][0]["message"]["content"].strip()

class PortfolioDataAgent:
    def __init__(self):
        self.portfolio_data = {}

    def fetch_portfolio_data(self, user_input):
        """
        Fetch portfolio details securely from user input.
        """
        # Assign user input to portfolio data
        self.portfolio_data = {
            "user_id": user_input.get("user_id"),
            "holdings": user_input.get("holdings", [])
        }
        return self.portfolio_data

    def validate_portfolio_data(self):
        """
        Validate portfolio data for accuracy and completeness using GPT.
        """
        if not self.portfolio_data:
            return False, "No portfolio data found."
        
        # Prepare prompt for GPT validation
        prompt = f"""
        Validate the following portfolio data:
        {self.portfolio_data}

        Ensure each holding has:
        - A valid stock symbol (e.g., AAPL, GOOGL)
        - Quantity as a positive number
        - Purchase price as a positive number

        Respond with 'Valid' if everything is correct, otherwise list issues.
        """
        validation_result = chatgpt_query(prompt)

        if "Valid" in validation_result:
            return True, "Portfolio data is valid."
        else:
            return False, validation_result

    def normalize_portfolio_data(self):
        """
        Normalize portfolio data for compatibility with analysis workflows.
        """
        normalized_data = []
        for holding in self.portfolio_data.get("holdings", []):
            normalized_data.append({
                "symbol": holding["symbol"].upper(),
                "quantity": float(holding["quantity"]),
                "purchase_price": float(holding["purchase_price"])
            })
        
        self.portfolio_data["holdings"] = normalized_data
        return self.portfolio_data
