# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import pandas as pd
import datetime as dt

import crewai as crewai
#from crewai import Crew
from textwrap import dedent
import logging
import crewai as crewai
import langchain_openai as lang_oai
import crewai_tools as crewai_tools

from src.Helpers.pretty_print_crewai_output import display_crew_output

from src.Indicators.bollinger import BollingerBands  
from src.Data_Retrieval.data_fetcher import DataFetcher  
from src.Agents.Research.bollinger_analysis_agent import BollingerAnalysisAgent
from src.Agents.Research.bollinger_buy_sell_agent import BollingerBuySellAgent
from src.Agents.Research.bollinger_buy_sell_critic_agent import BollingerBuySellCriticAgent

# Initialize logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

gpt_4o_high_tokens = lang_oai.ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.0,
    max_tokens=1500
)


class FinancialCrew:
    def __init__(self, ticker):
        self.ticker = ticker
        today = dt.datetime.today()
        #start_date = dt.datetime(2014, 1, 1)
        start_date = today - dt.timedelta(days=90)  # make sure inclusive
        end_date = today        
        self.stock_data = DataFetcher().get_stock_data(symbol=ticker, start_date=start_date, end_date=end_date )

    def run(self):
        # Bollinger Bands Data Calculation
        bollinger_data = BollingerBands(self.stock_data)
        bollinger_bands_data = bollinger_data.calculate_bands()

        # Print signals manually
        bollinger_data.manually_compute_buy_sell_hold_signals()

        # Plot the data
        bollinger_data.plot_bollinger_band_data()

        # Initialize agents
        bollinger_investment_advisor_agent = BollingerAnalysisAgent(llm=gpt_4o_high_tokens)
        bollinger_buy_sell_agent = BollingerBuySellAgent(ticker=self.ticker, llm=gpt_4o_high_tokens)
        bollinger_critic_agent = BollingerBuySellCriticAgent(ticker=self.ticker, llm=gpt_4o_high_tokens)

        agents = [bollinger_investment_advisor_agent, bollinger_buy_sell_agent, bollinger_critic_agent]
              
 


        # Create tasks for Bollinger Bands analysis        
        analyze_bollinger_data = bollinger_investment_advisor_agent.analyse_bollinger_data(bollinger_bands_data)
        buy_sell_decision = bollinger_buy_sell_agent.buy_sell_decision()
        critique_agent_decisions = bollinger_critic_agent.critique_buy_sell_agent()
        revise_buy_sell_decisions = bollinger_buy_sell_agent.revise_buy_sell_decision()
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print('buy sell signals \n', bollinger_data.get_buy_sell_signals_drop_hold())

        tasks_baseline=[           
            analyze_bollinger_data,
            buy_sell_decision            
             ]


        tasks_1critique=[
            analyze_bollinger_data,
            buy_sell_decision,
            critique_agent_decisions,
            revise_buy_sell_decisions
            ]
        
        tasks_2critiques=[
            analyze_bollinger_data,
            buy_sell_decision,
            critique_agent_decisions,
            revise_buy_sell_decisions,
            critique_agent_decisions,
            revise_buy_sell_decisions
            ]
       
        tasks_3critiques=[
            analyze_bollinger_data,
            buy_sell_decision,
            critique_agent_decisions,
            revise_buy_sell_decisions,
            critique_agent_decisions,
            revise_buy_sell_decisions,
            critique_agent_decisions,
            revise_buy_sell_decisions
            ]

        # Kickoff CrewAI agents and tasks
        crew = crewai.Crew(
            agents=agents,
            tasks=tasks_baseline,
            verbose=True,
            process=crewai.Process.sequential
        )

        result = crew.kickoff()
        return result

if __name__ == "__main__":
    print("## Research Interation Analysis")
    print('-------------------------------')

    ticker='aapl'    
  
    financial_crew = FinancialCrew(ticker=ticker)
    logging.info("Financial crew initialized successfully")

    try:
        crew_output = financial_crew.run()
        logging.info("Financial crew execution run() successfully")
    except Exception as e:
        logging.error(f"Error during crew execution: {e}")
        sys.exit(1)
    
    # Accessing the crew output
    print("\n\n########################")
    print("## Here is the Report")
    print("########################\n")

    display_crew_output(crew_output)

    print("Collaboration complete")
    sys.exit(0)