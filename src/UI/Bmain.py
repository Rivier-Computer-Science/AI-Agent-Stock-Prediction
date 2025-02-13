import logging

from crewai import Agent, Crew, Task

from Agents.Integrate_data import integrate_data
from Agents.query_parser import parse_query

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define Agents
query_parser_agent = Agent(
    role="Query Parsing Agent",
    goal="Extract relevant financial indicators from user queries.",
    backstory="An expert AI that understands financial terminology and maps queries to relevant data points.",
    verbose=True
)

data_integration_agent = Agent(
    role="Data Integration Agent",
    goal="Retrieve and validate real-time financial data.",
    backstory="A skilled AI specializing in fetching and structuring stock market data from multiple sources.",
    verbose=True
)

# Define Tasks
parse_task = Task(
    description="Extract indicators such as inflation rates, search trends, and corporate announcements from user queries.",
    agent=query_parser_agent,
    expected_output="A structured dictionary of parsed indicators.",
    function=parse_query
)

integrate_data_task = Task(
    description="Fetch and validate real-time stock market data based on the parsed indicators.",
    agent=data_integration_agent,
    expected_output="A JSON structure containing relevant financial data.",
    function=integrate_data
)

# Create Crew with Agents
finance_crew = Crew(
    agents=[query_parser_agent, data_integration_agent],
    tasks=[parse_task, integrate_data_task],
    verbose=True
)

if __name__ == "__main__":
    queries = [
        "What’s the buying signal for Company X based on inflation rates and recent announcements?",
        "How are search trends affecting stock prices?",
        "What’s the current fear and greed index and its impact on the market?",
        "Give me insights on Apple stock based on inflation and search trends."
    ]

    print("\nStarting Financial Analysis Crew...\n")

    for query in queries:
        print(f"\nProcessing Query: {query}")
        parsed_data = finance_crew.kickoff(inputs={"query": query})

        if parsed_data:
            print(f"Processed Indicators: {parsed_data['indicators']}")

            # Extract relevant indicators for integration
            indicators = parsed_data['indicators']
            ticker = "AAPL"  # Default ticker, can be dynamic based on query

            # Fetching and Integrating Data
            if "inflation_rate" in indicators:
                integrated_data = integrate_data(ticker, "INFLATION")
                print(f"Integrated Data (Inflation): {integrated_data}")

            if "search_trends" in indicators:
                integrated_data = integrate_data(ticker, "TRENDING")
                print(f"Integrated Data (Search Trends): {integrated_data}")

            if "fear_and_greed_index" in indicators:
                integrated_data = integrate_data(ticker, "FEAR_GREED")
                print(f"Integrated Data (Fear and Greed Index): {integrated_data}")

            if "corporate_announcements" in indicators:
                integrated_data = integrate_data(ticker, "NEWS")
                print(f"Integrated Data (Corporate Announcements): {integrated_data}")

            print("\n" + "-" * 50)
        else:
            logging.error("Failed to parse query.")