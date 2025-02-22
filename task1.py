from crewai import Agent, Task, Crew
import re
from typing import Dict, Any
import openai

# OpenAI API Key (Ensure it's securely stored in environment variables)
OPENAI_API_KEY = " "

def chatgpt_query(prompt: str) -> str:
    """Fetches a response from OpenAI's ChatGPT API."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        api_key=OPENAI_API_KEY
    )
    return response["choices"][0]["message"]["content"].strip()

class PortfolioDataAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Portfolio Data Agent",
            description="Handles portfolio data retrieval and management."
        )

    def get_portfolio_data(self, query: str) -> Dict[str, Any]:
        """
        Retrieves relevant portfolio data based on the user query.
        This can be extended to fetch actual portfolio data from a database or API.
        """
        # Simulate portfolio data retrieval for tax-related queries
        if "capital gains" in query:
            return {"status": "success", "capital_gains": 5000}  # Example response
        return {"status": "error", "message": "No relevant portfolio data found."}

class ScenarioInputAgent(Agent):
    def __init__(self, portfolio_data_agent: PortfolioDataAgent):
        super().__init__(
            name="Scenario Input Agent",
            description="Analyzes user queries and routes them appropriately."
        )
        self.portfolio_data_agent = portfolio_data_agent
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Analyzes user queries, validates input, and routes to appropriate agents.
        If tax-related, it integrates portfolio data for analysis.
        """
        if not self.validate_input(query):
            return {"status": "error", "message": "Invalid input. Please provide a meaningful query."}
        
        task = self.analyze_query(query)
        if task == "unknown":
            return {"status": "error", "message": "Task not recognized. Please refine your query."}
        
        # If task is tax analysis, fetch portfolio data
        if task == "tax_analysis":
            portfolio_data = self.portfolio_data_agent.get_portfolio_data(query)
            if portfolio_data["status"] == "error":
                return portfolio_data  # Return the error message if data retrieval fails
            return {"status": "success", "task": task, "message": f"Tax analysis initiated. Portfolio Data: {portfolio_data}"}
        
        return {"status": "success", "task": task, "message": f"Request routed to {task} agent."}
    
    def validate_input(self, query: str) -> bool:
        """Validates user input to ensure accuracy and relevance."""
        return bool(query and isinstance(query, str) and len(query.strip()) > 3)
    
    def analyze_query(self, query: str) -> str:
        """Analyzes the query and classifies it into the appropriate task category."""
        query = query.lower()
        prompt = f"Classify the following query into one of these categories: tax_analysis, signal_generation, backtesting, forward_testing, visualization, security_compliance. Query: {query}"
        response = chatgpt_query(prompt)
        return response if response in ["tax_analysis", "signal_generation", "backtesting", "forward_testing", "visualization", "security_compliance"] else "unknown"

# Example Usage
if __name__ == "__main__":
    # Initialize the CrewAI system and agents
    crew = Crew()

    # Create instances of both agents
    portfolio_data_agent = PortfolioDataAgent()
    agent = ScenarioInputAgent(portfolio_data_agent)
    
    # Add agents to Crew
    crew.add_agent(portfolio_data_agent)
    crew.add_agent(agent)

    # Simulate a user query related to tax analysis
    user_query = "Can you analyze my capital gains tax?"
    response = agent.execute(user_query)
    print(response)

    # Optionally, you can check if the agents are registered in Crew
    print(f"Registered agents: {crew.get_agents()}")
