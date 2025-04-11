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




if __name__ == "__main__":
    print("## Research Interation Analysis")
    print('-------------------------------')

    ticker='nvda'    
  
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