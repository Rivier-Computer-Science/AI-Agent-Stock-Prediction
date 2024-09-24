######################################
# This code comes from: https://github.com/crewAIInc/crewAI-examples/blob/main/stock_analysis/stock_analysis_agents.py
# And is licensed under MIT
######################################

from crewai import Agent

from src.Agents.Analysis.Tools.browser_tools import BrowserTools
from src.Agents.Analysis.Tools.calculator_tools import CalculatorTools
from src.Agents.Analysis.Tools.search_tools import SearchTools
from src.Agents.Analysis.Tools.sec_tools import SECTools

from langchain_community.tools import YahooFinanceNewsTool
from langchain_openai import ChatOpenAI

gpt_model = ChatOpenAI(
  temperature=0,
  #model_name="gpt-3.5-turbo",
  model_name="gpt-4o"
)

class StockAnalysisAgents():
  def financial_analyst(self):
    return Agent(
      llm=gpt_model,
      role='The Best Financial Analyst',
      goal="""Impress all customers with your financial data 
      and market trends analysis""",
      backstory="""The most seasoned financial analyst with 
      lots of expertise in stock market analysis and investment
      strategies that is working for a super important customer.""",
      verbose=True,
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        CalculatorTools.calculate,
        SECTools.search_10q,
        SECTools.search_10k
      ]
    )

  def research_analyst(self):
    return Agent(
      llm=gpt_model,
      role='Staff Research Analyst',
      goal="""Being the best at gather, interpret data and amaze
      your customer with it""",
      backstory="""Known as the BEST research analyst, you're
      skilled in sifting through news, company announcements, 
      and market sentiments. Now you're working on a super 
      important customer""",
      verbose=True,
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        SearchTools.search_news,
        YahooFinanceNewsTool(),
        SECTools.search_10q,
        SECTools.search_10k
      ]
  )

  def investment_advisor(self):
    return Agent(
      llm=gpt_model,
      role='Private Investment Advisor',
      goal="""Impress your customers with full analyses over stocks
      and completer investment recommendations""",
      backstory="""You're the most experienced investment advisor
      and you combine various analytical insights to formulate
      strategic investment advice. You are now working for
      a super important customer you need to impress.""",
      verbose=True,
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        SearchTools.search_news,
        CalculatorTools.calculate,
        YahooFinanceNewsTool()
      ]
    )
  
  def macd_trading_advisor(self):
        """
        Returns an agent that interprets MACD signals (bullish/bearish) and provides trading insights
        to help traders make better decisions and improve their profitability.
        """
        return Agent(
            llm=gpt_model,
            role='MACD Trading Advisor',
            goal="""Interpret MACD signals and provide actionable insights on market trends. 
            Help traders identify potential bullish or bearish movements and offer advice 
            on whether to buy, sell, or hold a stock.""",
            backstory="""As an expert in technical analysis, this agent specializes in MACD interpretation. 
            It uses its deep knowledge of stock market trends to assist traders in making informed decisions 
            based on the MACD signals.""",
            verbose=True,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet,
                CalculatorTools.calculate,
                SECTools.search_10q,
                SECTools.search_10k,
                YahooFinanceNewsTool()
            ]
        )