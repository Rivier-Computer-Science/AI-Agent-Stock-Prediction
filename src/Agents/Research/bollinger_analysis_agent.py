from crewai import Agent, Task
from textwrap import dedent
import pandas as pd

from src.Agents.base_agent import BaseAgent



class BollingerAnalysisAgent(BaseAgent):
   def __init__(self, **kwargs):
        super().__init__(
            role='Bollinger Bands Investment Advisor',
            goal="""Provide buy, sell, hold signals by analyzing Bollinger Bands data.""",
            backstory="""As a highly skilled investment advisor, you're specialized in analyzing Bollinger Bands to
                        provide buy, sell, hold trading signals for your clients.""",
            allow_delegation= False,
            verbose=True,
            **kwargs)
        

   def analyse_bollinger_data(self, bollinger_band_data):
        """
        Create a new task to analyze Bollinger Bands.

        Args:
            agent: The financial analyst agent responsible for analyzing the Bollinger Bands.
            bollinger_bands (dict): The calculated Bollinger Bands.

        Returns:
            Task: The task object for analyzing Bollinger Bands.
        """
        description = dedent(f"""
            Analyze the provided Bollinger Bands data, which includes the Price, Upper Band and Lower Band data.
                             
            Treat each day as independent data. Start with the earliest date and progress one day at a time.

            When price crosses below the upper band from above, you issue a SELL signal.
            When price crosses above the lower band from below, you issue a BUY signal.
            Otherwise, you issue a HOLD signal.             

            Bollinger Bands pandas dataframe data: {bollinger_band_data}
            Colunn names: Price is Close, Upper Band is Upper Band, Lower Band is Lower Band        

        """)
        # {bollinger_band_data['Upper Band'].iloc[-1]}
        # Creating and returning the Task object
        return Task(
            description=description,
            agent=self,
            expected_output="""A JSON dict with the date as the key and BUY, SELL, or HOLD as the value. " \
                               Include an entry for every date in pandas dataframe"""
        )
   


