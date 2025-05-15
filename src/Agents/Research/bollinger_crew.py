import pandas as pd

import crewai as crewai
import langchain_openai as lang_oai
import crewai_tools as crewai_tools

from src.Indicators.bollinger import BollingerBands  
from src.Agents.Research.bollinger_analysis_agent import BollingerAnalysisAgent
from src.Agents.Research.bollinger_buy_sell_agent import BollingerBuySellAgent
from src.Agents.Research.bollinger_buy_sell_critic_agent import BollingerBuySellCriticAgent


gpt_4o_high_tokens = lang_oai.ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.0,
    max_tokens=1500
)

class BollingerCrew:
    def __init__(self, ticker, stock_data, length=20, std=2):
        self.ticker = ticker
        self.stock_data = stock_data
        self.length = length
        self.std = std        

    def run(self):
        # Bollinger Bands Data Calculation
        bollinger_data = BollingerBands(self.stock_data, length=self.length, std=self.std)
        bollinger_bands_data = bollinger_data.calculate_bands()

        # Print signals manually
        bollinger_signals = bollinger_data.manually_compute_buy_sell_hold_signals()
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print("bollinger manual signals:\n",bollinger_signals)
            print('only buy sell signals \n', bollinger_data.get_buy_sell_signals_drop_hold())

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


        tasks_baseline=[           
            analyze_bollinger_data,
            buy_sell_decision            
             ]


        # tasks_1critique=[
        #     analyze_bollinger_data,
        #     buy_sell_decision,
        #     critique_agent_decisions,
        #     revise_buy_sell_decisions
        #     ]
        
        # tasks_2critiques=[
        #     analyze_bollinger_data,
        #     buy_sell_decision,
        #     critique_agent_decisions,
        #     revise_buy_sell_decisions,
        #     critique_agent_decisions,
        #     revise_buy_sell_decisions
        #     ]
       
        # tasks_3critiques=[
        #     analyze_bollinger_data,
        #     buy_sell_decision,
        #     critique_agent_decisions,
        #     revise_buy_sell_decisions,
        #     critique_agent_decisions,
        #     revise_buy_sell_decisions,
        #     critique_agent_decisions,
        #     revise_buy_sell_decisions
        #     ]

        # Kickoff CrewAI agents and tasks
        crew = crewai.Crew(
            agents=agents,
            tasks=tasks_baseline,
            verbose=True,
            process=crewai.Process.sequential
        )

        result = crew.kickoff()

        task_output = buy_sell_decision.output  # Example of how to get task output
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print("all signals\n",task_output.json ) 

        return task_output.json, result