from typing import Any, Dict

from crewai import Agent


class TaxRulesAgent(Agent):
    tax_data: Dict[str, Any] = {} 

    def __init__(self):
        super().__init__(
            name="Tax Rules Agent",
            role="Tax Compliance and Optimization Specialist",
            goal="Apply jurisdiction-specific tax rules and identify tax optimization strategies.",
            backstory=(
                "A financial AI expert in tax regulations, ensuring compliance while "
                "maximizing tax efficiency for portfolios."
            )
        )

    def apply_tax_rules(self, portfolio_data: Dict[str, Any], jurisdiction: str) -> Dict[str, Any]:
        """
        Apply jurisdiction-specific tax rules on portfolio data.
        """
        tax_rates = {
        "US": 0.37,  # Taxed as ordinary income, max rate 37%
        "UK": 0.20,  # General capital gains tax rate
        "EU": 0.26,  # Approximate highest rate across EU countries (e.g., Germany)
        "IN": 0.20   # For holdings up to 24 months
        }
        tax_rate = tax_rates.get(jurisdiction, 0.15)  # Default to US rate

        tax_liability = 0
        for holding in portfolio_data.get("holdings", []):
            capital_gain = (holding["quantity"] * 200) - (holding["quantity"] * holding["purchase_price"])  # Assume current price is $200
            tax = capital_gain * tax_rate
            tax_liability += tax
        
        self.tax_data = {
            "jurisdiction": jurisdiction,
            "tax_rate": tax_rate,
            "tax_liability": tax_liability
        }
        return self.tax_data

    def identify_tax_optimizations(self) -> Dict[str, Any]:
        """
        Identify opportunities for tax optimizations and offsets.
        """
        optimizations = []

        if self.tax_data.get("tax_liability", 0) > 1000:
            optimizations.append("Consider long-term holdings for reduced tax rates.")
        
        if self.tax_data.get("jurisdiction") == "US":
            optimizations.append("Utilize tax-loss harvesting to offset gains.")
        
        return {"optimizations": optimizations}

    def generate_tax_report(self) -> Dict[str, Any]:
        """
        Generate detailed tax liability reports for users.
        """
        report = {
            "Jurisdiction": self.tax_data.get("jurisdiction", "Unknown"),
            "Tax Rate": f"{self.tax_data.get('tax_rate', 0) * 100}%",
            "Total Tax Liability": self.tax_data.get("tax_liability", 0),
            "Optimizations": self.identify_tax_optimizations()["optimizations"]
        }
        return report

    def execute(self, portfolio_data: Dict[str, Any], jurisdiction: str) -> Dict[str, Any]:
        """
        Main method to execute tax rules, optimizations, and report generation.
        """
        self.apply_tax_rules(portfolio_data, jurisdiction)
        tax_report = self.generate_tax_report()
        return {
            "status": "success",
            "task": "tax_analysis",
            "tax_report": tax_report
        }
        
import os
from typing import Any, Dict

from crewai import Agent
from openai import OpenAI

# OpenAI API Key (Ensure it's securely stored in environment variables)
client = OpenAI(api_key="key")

def chatgpt_query(prompt: str) -> str:
    """Fetches a response from OpenAI's ChatGPT API."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]

    )
    return response.choices[0].message.content

class PortfolioDataAgent(Agent):
    portfolio_data: Dict[str, Any] = {}  # Explicitly define portfolio_data

    def __init__(self):
        super().__init__(
            name="Portfolio Data Agent",
            role="Portfolio Validator",
            goal="Ensure the accuracy and integrity of portfolio data for analysis.",
            backstory=(
                "An AI-driven financial assistant that validates and normalizes portfolio data, "
                "ensuring compatibility with advanced analytical tools for trading and investment insights."
            )
        )

    def fetch_portfolio_data(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch portfolio details securely from user input."""
        self.portfolio_data = {
            "user_id": user_input.get("user_id"),
            "holdings": user_input.get("holdings", [])
        }
        return self.portfolio_data

    def validate_portfolio_data(self) -> (bool, str):
        """Validate portfolio data for accuracy and completeness using GPT."""
        if not self.portfolio_data.get("holdings"):
            return False, "No valid portfolio data found."

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
        return False, validation_result

    def normalize_portfolio_data(self) -> Dict[str, Any]:
        """Normalize portfolio data for compatibility with analysis workflows."""
        normalized_data = []
        for holding in self.portfolio_data.get("holdings", []):
            try:
                normalized_data.append({
                    "symbol": holding["symbol"].upper(),
                    "quantity": float(holding["quantity"]),
                    "purchase_price": float(holding["purchase_price"])
                })
            except (ValueError, KeyError):
                return {"status": "error", "message": "Invalid portfolio data format."}

        self.portfolio_data["holdings"] = normalized_data
        return self.portfolio_data
    
from typing import Any, Dict, List

from crewai import Agent
from openai import OpenAI
from pydantic import PrivateAttr

# src/Agents/SignalAnalysisAgent.py



# OpenAI API Key (Ensure it's securely stored in environment variables)
client = OpenAI(api_key="key")
def chatgpt_query(prompt: str) -> str:
    """Fetches a response from OpenAI's ChatGPT API."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]

    )
    return response.choices[0].message.content

class SignalAnalysisAgent(Agent):
    signals: List[Dict[str, str]] = []

    def __init__(self):
        super().__init__(
            name="Signal Analysis Agent",
            role="Market Signal Analyzer",
            goal="Analyze financial portfolios and generate actionable trading signals.",
            backstory=(
                "A market-savvy AI specializing in stock analysis, trend detection, and "
                "providing actionable buy, sell, or hold recommendations based on market conditions."
            )
        )

    def analyze_portfolio(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
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

        # Parse the GPT-3 response into structured signals
        parsed_result = []
        try:
            parsed_result
        except Exception as e:
            return {"status": "error", "message": "Failed to parse GPT-3 response."}

        # Store and return the analysis result
        self.signals = parsed_result
        return {"status": "success", "signals": self.signals}

class ScenarioInputAgent(Agent):
    _portfolio_data_agent: PortfolioDataAgent = PrivateAttr()
    _signal_analysis_agent: SignalAnalysisAgent = PrivateAttr()

    def __init__(self, portfolio_data_agent: PortfolioDataAgent, signal_analysis_agent: SignalAnalysisAgent, **kwargs):
        super().__init__(
            name="Scenario Input Agent",
            role="Query Analyzer",
            goal="Analyze user queries and route them to appropriate agents for processing.",
            backstory=(
                "An AI-driven assistant specializing in understanding financial queries and "
                "directing them to the best-suited analytical tools for precise decision-making."
            ),
            **kwargs
        )
        self._portfolio_data_agent = portfolio_data_agent
        self._signal_analysis_agent = signal_analysis_agent

    def execute(self, query: str, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes user queries, validates input, and routes to appropriate agents.
        """
        task = "signal_generation"
        
        # Route to Signal Analysis Agent
        if task == "signal_generation":
            portfolio_data = self._portfolio_data_agent.fetch_portfolio_data(user_input)
            is_valid, message = self._portfolio_data_agent.validate_portfolio_data()
            
            if not is_valid:
                return {"status": "error", "message": message}
            
            normalized_data = self._portfolio_data_agent.normalize_portfolio_data()
            signals = self._signal_analysis_agent.analyze_portfolio(normalized_data)
            
            return {"status": "success", "task": task, "signals": signals}

# src/Agents/SignalAnalysisAgent.py

from typing import Any, Dict, List

from crewai import Agent
from openai import OpenAI

# OpenAI API Key (Ensure it's securely stored in environment variables)
client = OpenAI(api_key="key")
def chatgpt_query(prompt: str) -> str:
    """Fetches a response from OpenAI's ChatGPT API."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]

    )
    return response.choices[0].message.content

class SignalAnalysisAgent(Agent):
    signals: List[Dict[str, str]] = []

    def __init__(self):
        super().__init__(
            name="Signal Analysis Agent",
            role="Market Signal Analyzer",
            goal="Analyze financial portfolios and generate actionable trading signals.",
            backstory=(
                "A market-savvy AI specializing in stock analysis, trend detection, and "
                "providing actionable buy, sell, or hold recommendations based on market conditions."
            )
        )

    def analyze_portfolio(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
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

        # Parse the GPT-3 response into structured signals
        parsed_result = []
        try:
            parsed_result
        except Exception as e:
            return {"status": "error", "message": "Failed to parse GPT-3 response."}

        # Store and return the analysis result
        self.signals = parsed_result
        return {"status": "success", "signals": self.signals}