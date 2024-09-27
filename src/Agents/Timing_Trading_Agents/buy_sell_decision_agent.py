import crewai as crewai
from textwrap import dedent
from src.Agents.base_agent import BaseAgent
import logging

class BuySellDecisionAgent(BaseAgent):
    def __init__(self, stock, **kwargs):
        super().__init__(
            role='Investment Advisor',
            goal="""Make a buy/sell decision based on the sentiment analysis before earnings.""",
            backstory="""You are an experienced investment advisor known for making strategic decisions based on market sentiments and trends.""",
            verbose=True,
            tools=[],
            allow_delegation=False,
            **kwargs
        )
        self.stock = stock
        logging.info("BuySellDecisionAgent initialized")

    def make_decision(self):
        logging.info("Buy/Sell decision task being prepared")
        return crewai.Task(
            description=dedent(f"""
                Based on the sentiment analysis of {self.stock}, provide a buy/sell recommendation.
                Take into account the overall market conditions and historical performance around earnings periods.
            """),
            agent=self,
            expected_output=f"Investment decision report for {self.stock}, including buy/sell recommendation."
        )
