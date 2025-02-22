

import re
from pydantic import BaseModel, Field
from textwrap import dedent
import crewai as crewai
from datetime import datetime
import os
from src.Agents.base_agent import BaseAgent
from src.Agents.Analysis.Tools.search_tools import SearchTools


class ScenarioInputAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            role='Scenario Input Agent',
            goal="Provide comprehensive market scenarios affecting portfolio tickers",
            backstory='An expert in market analysis and reporting',
            tools=[SearchTools.search_news],
            **kwargs)
        
        self.previous_report = None

    def get_scenarios_from_news(self):
        if os.name == 'nt':  # For Windows
            formatted_date = datetime.now().strftime('%b %#d, %Y')  # Example: 'Jan 1, 2024'
        else:  # For Unix/Linux/Mac
            formatted_date = datetime.now().strftime('%b %-d, %Y')  

        return crewai.Task(
            description=dedent(f"""
                Collect and summarize recent news articles, press
                releases, and market analyses related to events that
                could affect tickers in the portfolio.
                               
                Pay special attention to any inflation, interest rate,
                and large movements in the stock market.
                                                         
               Make sure to use the most recent data as possible. Do not consider news older than 1 week from {formatted_date}.

               "If you do your BEST WORK, I'll give you a $10,000 commission!"
          
          
            """),
            agent=self,
            expected_output="A comprehensive report on recent market scenarios"
        )
    
    def revise_report(self):       
        # Define the revision task logic
        def task_logic():
            # Prepare the prompt for the agent
            prompt = dedent(f"""
                As the Scenario Input Agent, you are tasked with revising your previous market scenarios report based on the critique provided by the Scenario Input Critic Agent.
                
                **Instructions:**

                - Carefully read the critique and identify all the points that need to be addressed.
                - Revise your report to incorporate the feedback, ensuring that all suggestions are implemented.
                - Enhance the level of detail, analysis, and clarity in your report.
                - Focus on providing comprehensive market scenarios affecting SPY, DIA, and AGG.
                - Ensure that the report is well-structured, accurate, and insightful.

                Please provide the revised report below without including these instructions or any meta-commentary.
            """)
            # Return the prompt for the agent to process
            return prompt

        return crewai.Task(
            description=dedent("""
                Revise your previous market scenarios report based on the critique provided by the Scenario Input Critic Agent.
                Ensure that all feedback is addressed, and the report is enhanced accordingly.
            """),
            agent=self,
            expected_output="An improved market scenarios report",
            action=task_logic  
        )
