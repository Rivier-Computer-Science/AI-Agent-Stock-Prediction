######################################
# This code comes from the MACD Trading System project
# And is licensed under MIT
######################################
#
# The following keys must be defined in the environment shell
# OPENAI_API_KEY=sk-
# SEC_API_API_KEY=
# SERPER_API_KEY
#
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from crewai import Crew
from textwrap import dedent

from src.Agents.Analysis.stock_analysis_agents import StockAnalysisAgents
from src.Agents.Analysis.stock_analysis_tasks import StockAnalysisTasks
from src.Indicators.macd import MACDIndicator
from src.Data_Retrieval.data_fetcher import DataFetcher

from dotenv import load_dotenv
load_dotenv()

class FinancialCrew:
    def __init__(self, company):
        self.company = company

    def run(self):
        # Initialize agents and tasks
        agents = StockAnalysisAgents()
        tasks = StockAnalysisTasks()

        # Initialize the MACD trading advisor agent
        macd_agent = agents.macd_trading_advisor()

        # Fetch stock data using DataFetcher
        data_fetcher = DataFetcher()
        stock_data = data_fetcher.get_stock_data(self.company)

        # Calculate MACD using the MACDIndicator
        macd_indicator = MACDIndicator()
        macd_data = macd_indicator.calculate(stock_data)

        # Create a MACD analysis task
        macd_task = tasks.macd_analysis(macd_agent, macd_data)

        # Run the agents and tasks in the Crew
        crew = Crew(
            agents=[
                macd_agent
            ],
            tasks=[
                macd_task
            ],
            verbose=True
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    print("## Welcome to MACD Trading System")
    print('-------------------------------')

    # Prompt user for the company to analyze
    company = input(
        dedent("""
            What is the company you want to analyze?
        """))

    # Create FinancialCrew instance and run the analysis
    financial_crew = FinancialCrew(company)
    result = financial_crew.run()

    print("\n\n########################")
    print("## Here is the Report")
    print("########################\n")
    print(result)
