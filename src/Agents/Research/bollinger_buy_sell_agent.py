import crewai as crewai
from textwrap import dedent
from src.Agents.base_agent import BaseAgent
from pydantic import BaseModel
from typing import Dict
import logging


class BollingerBuySellAgentOutput(BaseModel):
    output: Dict[str,str]


class BollingerBuySellAgent(BaseAgent):
    def __init__(self, ticker="AAPL", **kwargs):
        super().__init__(
            role=f'Make buy sell hold decisions for a ticker {ticker}',
            goal=f"Evaluate all the data provided and make a decision on whether to buy or sell the {ticker} ticker",
            backstory=f"You are a seasoned trader who knows when to buy, sell, or hold ticker {ticker}.",
            verbose=True,
            tools=[], 
            allow_delegation=False,
            **kwargs
        )
        self.ticker = ticker
        logging.info("BollingerBuySellAgent initialized")

    def buy_sell_decision(self):
        return crewai.Task(
            description=dedent(f"""
                Based on the Bollinger band data provided for {self.ticker},
                    make a daily decision for each day of data. 

                Similar to a backtest, consider each day independently as a new decision.
                    Start with the earliest date and advance one day to determine each signal.
                    
                Output a signal as a date and a single word: BUY, SELL, or HOLD.

                Highlight signals you changed from the standard bollinger signals.
                              
                If you predict correctly, you get to keep a 10% commission on profits.                                          
            """),
            agent=self,
            output_json=BollingerBuySellAgentOutput,
            expected_output=f"""A JSON dict with the date as the key and BUY, SELL, or HOLD as the value. 
                                Include an entry for every date in pandas dataframe for {self.ticker}"""
        )
    
    def revise_buy_sell_decision(self):
        return crewai.Task(
            description=dedent(f"""
                Based on the critique of the trading signals provied for {self.ticker},
                    improve the buy, sell, hold signals.

                Highlight signals you changed from the standard bollinger signals.
                              
                If you predict correctly, you get to keep a 10% commission on profits.                                          
            """),
            agent=self,
            output_json=BollingerBuySellAgentOutput,
            expected_output=f"""A JSON dict with the date as the key and BUY, SELL, or HOLD as the value. 
                                Include an entry for every date in pandas dataframe for {self.ticker}"""
        )