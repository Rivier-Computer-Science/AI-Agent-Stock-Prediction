from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from textwrap import dedent

# Initialize the GPT model
gpt_model = ChatOpenAI(temperature=0, model_name="gpt-4o")

class VPTAnalysisAgent:
    """
    VPTAnalysisAgent analyzes Volume Price Trend (VPT) data along with the current stock price
    and provides a clear investment decision: either "BUY" or "SELL".
    """
    def __init__(self):
        self.role = "VPT Trading Advisor"
        self.goal = (
            "Interpret Volume Price Trend (VPT) signals and provide a clear investment decision. "
            "Help traders decide whether to BUY or SELL based on the current VPT value and market conditions."
        )
        self.backstory = (
            "As a seasoned technical analyst specializing in the Volume Price Trend, "
            "this agent leverages advanced natural language processing to distill complex market data into a binary decision. "
            "Its focus is on delivering a concise recommendation: either BUY or SELL."
        )

    def vpt_trading_advisor(self):
        """
        Configures and returns a CrewAI agent for VPT analysis.
        """
        return Agent(
            llm=gpt_model,
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            tools=[]  # Add any necessary tools if needed
        )

    def vpt_analysis(self, agent, vpt_data, current_price):
        """
        Creates a task for the agent to analyze the latest VPT data along with the current stock price,
        and provide a clear investment decision.

        Args:
            agent (Agent): The CrewAI agent instance for VPT analysis.
            vpt_data (pd.DataFrame): The DataFrame containing calculated VPT values.
            current_price (float): The current stock price.

        Returns:
            Task: A CrewAI task instructing the agent to output either "BUY" or "SELL".
        """
        latest_vpt = vpt_data['VPT'].iloc[-1]
        report = dedent(f"""
            VPT Analysis Report:
            - Latest VPT Value: {latest_vpt}
            - Current Stock Price: {current_price}
            
            Based on the above data, please provide a single-word investment decision.
            Your final answer must be either "BUY" or "SELL" with no additional commentary.
        """)
        return Task(
            description=report,
            agent=agent,
            expected_output="A single word decision: either BUY or SELL."
        )
