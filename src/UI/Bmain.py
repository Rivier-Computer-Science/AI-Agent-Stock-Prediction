import logging

from crewai import Agent, Crew, Task

from Agents.bagents import integrate_data, parse_query

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
        "What’s the buying signal for Apple based on inflation rates and recent announcements?"
    ]

    print("\nStarting Financial Analysis Crew...\n")

    for query in queries:
        print(f"\nProcessing Query: {query}")
        
        # Step 1: Get Parsed Indicators from AI
        parsed_data = finance_crew.kickoff(inputs={"query": query})

        if parsed_data:
            parsed_dict = parsed_data.dict()  # Convert CrewOutput to dictionary
            
            # Extract available indicators dynamically
            indicators = {}
            
            if "InflationRates" in parsed_dict:
                indicators["inflation_rate"] = parsed_dict["InflationRates"]

            if "SearchTrends" in parsed_dict:
                indicators["search_trends"] = parsed_dict["SearchTrends"]

            if "CorporateAnnouncements" in parsed_dict:
                indicators["corporate_announcements"] = parsed_dict["CorporateAnnouncements"]

            if not indicators:
                logging.error("No relevant indicators found in parsed data.")
                continue

            print(f"Identified Indicators: {list(indicators.keys())}")

            # Step 2: Fetch Real Data from APIs
            ticker = "AAPL"  # Can be dynamic based on query
            integrated_data = {}

            if "inflation_rate" in indicators:
                integrated_data["inflation"] = integrate_data(ticker, "INFLATION")
                print(f"Fetched Inflation Data: {integrated_data['inflation']}")

            if "search_trends" in indicators:
                integrated_data["trends"] = integrate_data(ticker, "TRENDING")
                print(f"Fetched Search Trends: {integrated_data['trends']}")

            if "corporate_announcements" in indicators:
                integrated_data["news"] = integrate_data(ticker, "NEWS")
                print(f"Fetched Corporate Announcements: {integrated_data['news']}")

            if not integrated_data:
                print("No real-time data fetched.")
                continue

            # Step 3: AI Recommendation
            recommendation = finance_crew.recommend_action(inputs={"query": query, "data": integrated_data})
            print(f"\nAI Recommendation: {recommendation}")

            print("\n" + "-" * 50)

        else:
            logging.error("Failed to parse query.")
