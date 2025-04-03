from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from textwrap import dedent
from src.Agents.base_agent import BaseAgent  # Ensure this base agent exists in your project

# Initialize the GPT model (adjust parameters as needed)
gpt_model = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o"
)

class ElderRayAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role="Elder-Ray Investment Advisor",
            goal="""Interpret Elder-Ray index values along with the current stock price 
                    and provide a clear investment recommendation (BUY, SELL, or HOLD) with supporting reasoning.""",
            backstory="""As an expert technical analyst specializing in the Elder-Ray indicator, 
                         I evaluate buying and selling pressures and integrate current price data 
                         to offer actionable investment decisions."""
        )
    
    def elder_ray_investment_advisor(self):
        return Agent(
            llm=gpt_model,
            role="Elder-Ray Investment Advisor",
            goal="""Interpret Elder-Ray index values along with the current stock price 
                    and provide a clear investment recommendation (BUY, SELL, or HOLD) with supporting reasoning.""",
            backstory="""I am a technical analysis expert focusing on the Elder-Ray indicator, 
                         utilizing data-driven insights to guide investment decisions.""",
            verbose=True,
            tools=[]
        )
    
    def elder_ray_analysis(self, agent, elder_ray_data, current_price):
        """
        Creates a task for analyzing the Elder-Ray data with the current stock price.
        """
        last_row = elder_ray_data.tail(1).to_string(index=False)
        description = dedent(f"""
            Analyze the following Elder-Ray index data:
            {last_row}
            
            Current Stock Price: {current_price}
            
            Based on the above data, please provide an investment recommendation: BUY, SELL, or HOLD.
            Include detailed supporting reasoning.
        """)
        return Task(
            description=description,
            agent=agent,
            expected_output="A comprehensive investment recommendation (BUY/SELL/HOLD) with detailed reasoning."
        )
