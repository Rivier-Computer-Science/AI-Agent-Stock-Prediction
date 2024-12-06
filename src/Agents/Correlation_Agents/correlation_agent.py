import json
import crewai as crewai
from textwrap import dedent
from src.Agents.base_agent import BaseAgent
from src.Data_Retrieval.data_fetcher import DataFetcher
import logging
import pandas as pd

# src/Agents/Correlation_Agents/correlation_agent.py

class CorrelationAgent(BaseAgent):
    def __init__(self, stock1: str = "AAPL", stock2: str = "MSFT", **kwargs):
        super().__init__(
            role='Stock Correlation Analyst',
            goal="""Analyze the correlation between two specified stocks using historical data.""",
            backstory="""You are a market analyst specialized in identifying highly correlated stocks for potential pairs trading opportunities.""",
            verbose=True,
            tools=[], 
            allow_delegation=False,
            **kwargs
        )
        self.stock1 = stock1
        self.stock2 = stock2
        logging.info("CorrelationAgent initialized") 

    def calculate_correlation(self) -> crewai.Task:        
        fetcher = DataFetcher()
        stock1_data = fetcher.get_stock_data(self.stock1)
        print(stock1_data)
        stock2_data = fetcher.get_stock_data(self.stock2)
        logging.info(f"Data retrieved for {self.stock1} and {self.stock2}")
        
        # Extract the 'Close' Series for each stock
        close_stock1 = stock1_data['Close'][self.stock1]
        close_stock2 = stock2_data['Close'][self.stock2]
        
        # Ensure both Series are aligned
        combined_data = pd.concat([close_stock1, close_stock2], axis=1).dropna()
        close_stock1_aligned = combined_data[self.stock1]
        close_stock2_aligned = combined_data[self.stock2]
        
        # Calculate the correlation coefficient
        correlation_coef = close_stock1_aligned.corr(close_stock2_aligned)
        logging.info(f"Correlation coefficient between {self.stock1} and {self.stock2}: {correlation_coef:.4f}")
        
        return crewai.Task(
            description=dedent(f"""
                Analyze the historical price data of {self.stock1} and {self.stock2} to calculate their correlation.
                Based on the correlation results, determine if the stocks are suitable for pairs trading.
            """),
            agent=self,
            expected_output=f"The correlation between {self.stock1} and {self.stock2} is: {correlation_coef:.4f}"
        )



