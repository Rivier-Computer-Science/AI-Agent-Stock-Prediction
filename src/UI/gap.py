# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import datetime as dt

from src.Data_Retrieval.data_fetcher import DataFetcher  
import crewai as crewai
from src.Agents.Research.bollinger_crew import BollingerCrew

from textwrap import dedent
import logging


from src.Helpers.pretty_print_crewai_output import display_crew_output



# Initialize logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)



if __name__ == "__main__":
    print("## Research Interation Analysis")
    print('-------------------------------')

    ticker='nvda'    

    today = dt.datetime.today()
    #start_date = dt.datetime(2014, 1, 1)
    start_date = today - dt.timedelta(days=90)  # make sure inclusive
    end_date = today        
    stock_data = DataFetcher().get_stock_data(symbol=ticker, start_date=start_date, end_date=end_date )
  
    financial_crew = BollingerCrew(ticker=ticker, stock_data=stock_data, length=20, std=2.0)
    logging.info("Financial crew initialized successfully")

    try:
        indicator, crew_output = financial_crew.run()
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