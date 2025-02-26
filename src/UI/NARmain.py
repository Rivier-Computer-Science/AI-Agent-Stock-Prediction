# src/UI/main.py

from crewai import Crew, Task

from Agents.NARTaxRulesAgent import (PortfolioDataAgent, ScenarioInputAgent,
                                     SignalAnalysisAgent, TaxRulesAgent)


def main():
    # Create instances of agents
    portfolio_data_agent = PortfolioDataAgent()
    signal_analysis_agent = SignalAnalysisAgent()
    tax_rules_agent = TaxRulesAgent()
    scenario_input_agent = ScenarioInputAgent(portfolio_data_agent, signal_analysis_agent)

    # Define sample tasks for CrewAI
    task1 = Task(
        description="Perform tax analysis",
        agent=tax_rules_agent,
        expected_output="A report detailing tax efficiency for portfolios."
    )

    task2 = Task(
        description="Generate trading signals",
        agent=signal_analysis_agent,
        expected_output="A JSON list of trading signals with buy, sell, or hold recommendations and reasoning."
    )

    task3 = Task(
        description="Analyze scenario impact",
        agent=scenario_input_agent,
        expected_output="A detailed impact analysis of the given financial scenario on portfolio performance."
    )

    # Initialize Crew with agents and tasks
    crew = Crew(agents=[portfolio_data_agent, signal_analysis_agent, tax_rules_agent, scenario_input_agent],
                tasks=[task1, task2, task3])

    # Collect user input for portfolio data
    user_input = {
        "user_id": input("Enter your User ID: "),
        "holdings": []
    }

    # Loop to collect holdings data
    while True:
        symbol = input("Enter stock symbol (or 'done' to finish): ").upper()
        if symbol == 'DONE':
            break
        
        # Input validation for quantity and purchase price
        while True:
            quantity = input(f"Enter quantity for {symbol}: ")
            if quantity.isdigit() and int(quantity) > 0:
                quantity = float(quantity)
                break
            print("Invalid quantity. Please enter a positive number.")
        
        while True:
            purchase_price = input(f"Enter purchase price for {symbol}: ")
            try:
                purchase_price = float(purchase_price)
                if purchase_price > 0:
                    break
            except ValueError:
                pass
            print("Invalid purchase price. Please enter a positive number.")
        
        # Append holding to the list
        user_input["holdings"].append({
            "symbol": symbol,
            "quantity": quantity,
            "purchase_price": purchase_price
        })

    # Select Task: Tax Analysis, Signal Generation, or Scenario Analysis
    print("\nSelect Task:")
    print("1. Tax Analysis")
    print("2. Signal Generation")
    print("3. Scenario Analysis")
    task_choice = input("Enter choice (1-3): ")

    if task_choice == "1":
        print("\nSelect Jurisdiction for Tax Calculation:")
        print("1. US")
        print("2. UK")
        print("3. EU")
        print("4. IN")
        jurisdiction_map = {"1": "US", "2": "UK", "3": "EU", "4": "IN"}
        jurisdiction_choice = input("Enter choice (1-4): ")
        jurisdiction = jurisdiction_map.get(jurisdiction_choice, "US")
        response = tax_rules_agent.execute(user_input, jurisdiction)
    elif task_choice == "2":
        user_query = "Generate trading signals"
        response = scenario_input_agent.execute(user_query, user_input)
    elif task_choice == "3":
        user_query = "Analyze scenario impact"
        response = scenario_input_agent.execute(user_query, user_input)
    else:
        print("Invalid choice. Exiting.")
        return

    # Display Response
    print("\nResponse:")
    print(response)

    # Check registered agents in Crew
    print("\nRegistered agents in Crew:")
    print(f"{crew.agents}")

if __name__ == "__main__":
    main()
    
# All the things are workingfine.