import crewai as crewai
from textwrap import dedent
from src.Agents.base_agent import BaseAgent
from src.Data_Retrieval.timing_trading_data_fetcher import DataFetcher
import logging

class SentimentAnalysisAgent(BaseAgent):
    def __init__(self, stock, earnings_dates, **kwargs):
        super().__init__(
            role='Sentiment Analyst',
            goal="""Analyze market sentiment for the stock just before the earnings release.""",
            backstory="""You specialize in analyzing sentiment in market news and reports, focusing on events just before earnings.""",
            verbose=True,
            tools=[],
            allow_delegation=False,
            **kwargs
        )
        self.stock = stock
        self.earnings_dates = earnings_dates
        logging.info("SentimentAnalysisAgent initialized")

    def analyze_sentiment(self):
        logging.info("Sentiment analysis task being prepared")
        return crewai.Task(
            description=dedent(f"""
                Perform a sentiment analysis for {self.stock} based on the latest news and market data before the earnings release.
                The upcoming earnings date is {self.earnings_dates}.
                Provide an overall sentiment score and a brief summary of key findings.
            """),
            agent=self,
            expected_output=f"Sentiment analysis report for {self.stock}, including sentiment score, summary, and upcoming earnings date: {self.earnings_dates}."
        )
