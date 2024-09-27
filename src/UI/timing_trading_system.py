import os
import sys
import crewai as crewai
from dotenv import load_dotenv
from src.Agents.Timing_Trading_Agents.sentiment_analysis_agent import SentimentAnalysisAgent
from src.Agents.Timing_Trading_Agents.buy_sell_decision_agent import BuySellDecisionAgent
from src.Data_Retrieval.timing_trading_data_fetcher import DataFetcher
from src.Helpers.pretty_print_crewai_output import display_crew_output

load_dotenv()

class TimingTradingSystem:
    def __init__(self, stock):
        self.stock = stock

    def run(self):
        # Fetch earnings dates
        fetcher = DataFetcher()
        earnings_dates = fetcher.get_earnings_date(self.stock)
        print(f"Earnings Dates for {self.stock}: {earnings_dates}")

        # Initialize agents
        sentiment_agent = SentimentAnalysisAgent(self.stock, earnings_dates)
        decision_agent = BuySellDecisionAgent(self.stock)

        # Use Crew to coordinate agents and tasks
        crew = crewai.Crew(
            agents=[sentiment_agent, decision_agent],
            tasks=[
                sentiment_agent.analyze_sentiment(),
                decision_agent.make_decision()
            ],
            process=crewai.Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        return result

if __name__ == "__main__":
    print("## Timing Trading System ##")
    stock = input("Enter the stock ticker: ")
    trading_system = TimingTradingSystem(stock)
    crew_output = trading_system.run()

    print("\n\n########################")
    print("## Trading Decision Report")
    print("########################\n")

    display_crew_output(crew_output)
    sys.exit(0)

