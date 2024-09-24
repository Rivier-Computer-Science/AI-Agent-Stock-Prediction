import os
from yahooquery import Ticker
import requests
from crewai import Crew
from textwrap import dedent
from dotenv import load_dotenv

# Load API keys
load_dotenv()

class FinancialCrew:
    def __init__(self, company, exchange):
        self.company = company
        self.exchange = exchange
        self.earnings_api_key = os.getenv("EARNINGSCAST_API_KEY", "demo")

    def fetch_sec_filings(self):
        # Using YahooQuery to fetch SEC filings
        ticker = Ticker(self.company)
        sec_filings = ticker.sec_filings
        if sec_filings.empty:
            print("No SEC filings found for the company.")
            return None
        return sec_filings.to_json()

    def fetch_earnings_calls(self):
        # Using EarningsCast API to fetch earnings events
        url = f'https://v2.api.earningscall.biz/events?apikey={self.earnings_api_key}&exchange={self.exchange}&symbol={self.company}'
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print("Error fetching earnings calls:", response.status_code)
            return None

    def fetch_earnings_transcript(self, year, quarter):
        # Fetch earnings call transcript from EarningsCast
        url = f'https://v2.api.earningscall.biz/transcript?apikey={self.earnings_api_key}&exchange={self.exchange}&symbol={self.company}&year={year}&quarter={quarter}'
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('text')
        else:
            print(f"Error fetching earnings transcript for {year} Q{quarter}: {response.status_code}")
            return None

    def run(self):
        agents = self.init_agents()
        tasks = self.init_tasks(agents)

        crew = Crew(
            agents=[agents['financial_analyst']],
            tasks=[tasks['sec_analysis'], tasks['earnings_analysis']],
            verbose=True
        )

        result = crew.kickoff()
        return result

    def init_agents(self):
        from src.Agents.Analysis.stock_analysis_agents import StockAnalysisAgents
        agents = StockAnalysisAgents()
        return {
            'financial_analyst': agents.financial_analyst()
        }

    def init_tasks(self, agents):
        from src.Agents.Analysis.stock_analysis_tasks import StockAnalysisTasks
        tasks = StockAnalysisTasks()

        # Fetch SEC filings and earnings call data
        sec_data = self.fetch_sec_filings()
        earnings_data = self.fetch_earnings_calls()

        return {
            'sec_analysis': tasks.analyze_sec_filings(agents['financial_analyst'], sec_data),
            'earnings_analysis': tasks.analyze_earnings_calls(agents['financial_analyst'], earnings_data)
        }

if __name__ == "__main__":
    print("## Welcome to Financial Analysis Crew")
    print('-------------------------------')
    company = input("What is the company you want to analyze? ")
    exchange = input("What is the exchange (e.g., NASDAQ, NYSE)? ")

    financial_crew = FinancialCrew(company, exchange)
    result = financial_crew.run()

    print("\n\n########################")
    print("## Analysis Report")
    print("########################\n")
    print(result)
