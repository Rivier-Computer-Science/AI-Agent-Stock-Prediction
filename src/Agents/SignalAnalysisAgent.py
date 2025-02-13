# src/Agents/SignalAnalysisAgent.py

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

class SignalAnalysisAgent:
    def __init__(self):
        self.signals = []

    def analyze_portfolio(self, portfolio_data):
        """
        Analyze portfolio data to generate trading signals.
        """
        if not portfolio_data or not portfolio_data.get("holdings"):
            return {"status": "error", "message": "No valid portfolio data provided."}
        
        # Prepare prompt for GPT-based signal analysis
        prompt = f"""
        Analyze the following portfolio data and provide trading signals:
        {portfolio_data}

        For each holding, recommend:
        - Buy, Sell, or Hold
        - Short reasoning behind each recommendation

        Respond in JSON format like this:
        [
            {{"symbol": "AAPL", "signal": "Buy", "reason": "Price below intrinsic value"}},
            {{"symbol": "GOOGL", "signal": "Hold", "reason": "Stable growth expected"}}
        ]
        """
        analysis_result = chatgpt_query(prompt)
        
        # Store and return the analysis result
        self.signals = analysis_result
        return {"status": "success", "signals": self.signals}
