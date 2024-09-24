import os
import sys
from crewai import Crew
from textwrap import dedent
from dotenv import load_dotenv
from src.Agents.Analysis.stock_analysis_agents import StockAnalysisAgents
from src.Agents.Analysis.stock_analysis_tasks import StockAnalysisTasks
from src.Indicators.bollinger import BollingerBands  # Import BollingerBands class
from src.Data_Retrieval.data_fetcher import DataFetcher  # Import DataFetcher class

# Load environment variables
load_dotenv()

class FinancialCrew:
    def __init__(self, company, stock_data):
        self.company = company
        self.stock_data = stock_data

    def run(self):
        # Initialize agents and tasks
        agents = StockAnalysisAgents()
        tasks = StockAnalysisTasks()

        # Initialize agents
        research_analyst_agent = agents.research_analyst()
        financial_analyst_agent = agents.financial_analyst()
        investment_advisor_agent = agents.investment_advisor()
        bollinger_agent = agents.bollinger_bands_investment_advisor()

        """ # Create tasks
        research_task = tasks.research(research_analyst_agent, self.company)
        financial_task = tasks.financial_analysis(financial_analyst_agent)
        filings_task = tasks.filings_analysis(financial_analyst_agent)
        recommend_task = tasks.recommend(investment_advisor_agent)"""

        # Bollinger Bands Calculation
        bollinger = BollingerBands(self.stock_data)
        bollinger_bands = bollinger.calculate_bands()

        # Create a new task for Bollinger Bands analysis
        bollinger_task1 = tasks.bollinger_analysis(research_analyst_agent, bollinger_bands)
        bollinger_task2 = tasks.bollinger_analysis(financial_analyst_agent, bollinger_bands)
        bollinger_task3 = tasks.bollinger_analysis(investment_advisor_agent, bollinger_bands)
        bollinger_task4 = tasks.bollinger_analysis(bollinger_agent, bollinger_bands)
        # Kickoff CrewAI agents and tasks
        crew = Crew(
            agents=[
                research_analyst_agent,
                financial_analyst_agent,
                investment_advisor_agent
            ],
            tasks=[
                bollinger_task1,
                bollinger_task2,
                bollinger_task3,
                bollinger_task4  # Include the Bollinger Bands analysis task
            ],
            verbose=True
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    print("## Welcome to Financial Analysis Crew")
    print('-------------------------------')

    # Prompt user for company name
    company = input(dedent("""
        What is the company you want to analyze?
    """))

    # Fetch stock data using DataFetcher
    data_fetcher = DataFetcher()
    stock_data = data_fetcher.get_stock_data(company)
    #print(stock_data.head()) 
    if stock_data.empty:
        print(f"No data found for {company}. Please try again.")
    else:
        # Create FinancialCrew instance
        financial_crew = FinancialCrew(company, stock_data)

        # Run the analysis and display the result
        result = financial_crew.run()

        print("\n\n########################")
        print("## Here is the Report")
        print("########################\n")
        print(result)
