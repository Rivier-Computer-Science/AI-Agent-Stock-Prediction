# src/Agents/ScenarioInputAgent.py

from typing import Any, Dict

from crewai import Agent

from .PortfolioDataAgent import PortfolioDataAgent
from .SignalAnalysisAgent import SignalAnalysisAgent  # NEW


class ScenarioInputAgent(Agent):
    def __init__(self, portfolio_data_agent: PortfolioDataAgent, signal_analysis_agent: SignalAnalysisAgent):
        super().__init__(
            name="Scenario Input Agent",
            description="Analyzes user queries and routes them appropriately."
        )
        self.portfolio_data_agent = portfolio_data_agent
        self.signal_analysis_agent = signal_analysis_agent   # NEW
    
    def execute(self, query: str, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes user queries, validates input, and routes to appropriate agents.
        """
        task = "signal_generation"
        
        # Route to Signal Analysis Agent
        if task == "signal_generation":
            portfolio_data = self.portfolio_data_agent.fetch_portfolio_data(user_input)
            is_valid, message = self.portfolio_data_agent.validate_portfolio_data()
            
            if not is_valid:
                return {"status": "error", "message": message}
            
            normalized_data = self.portfolio_data_agent.normalize_portfolio_data()
            signals = self.signal_analysis_agent.analyze_portfolio(normalized_data)
            
            return {"status": "success", "task": task, "signals": signals}
