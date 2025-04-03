# Portfolio Data Agent
# Role: Retrieves, normalizes, and prepares portfolio data.

class PortfolioDataAgent:
    def __init__(self):
        self.portfolio_data = {}
    
    def fetch_portfolio_data(self, user_id):
        """
        Fetch portfolio details securely from user accounts.
        Placeholder function to simulate data retrieval.
        """
        # Simulate fetching data from a secure database or API
        self.portfolio_data = {
            "user_id": user_id,
            "holdings": [
                {"symbol": "AAPL", "quantity": 10, "purchase_price": 150},
                {"symbol": "GOOGL", "quantity": 5, "purchase_price": 2800},
                {"symbol": "TSLA", "quantity": 8, "purchase_price": 700}
            ]
        }
        return self.portfolio_data
    
    def validate_portfolio_data(self):
        """
        Validate portfolio data for accuracy and completeness.
        """
        if not self.portfolio_data:
            return False, "No portfolio data found."
        
        required_keys = {"symbol", "quantity", "purchase_price"}
        for holding in self.portfolio_data.get("holdings", []):
            if not required_keys.issubset(holding.keys()):
                return False, "Incomplete portfolio data."
        
        return True, "Portfolio data is valid."
    
    def normalize_portfolio_data(self):
        """
        Normalize portfolio data for compatibility with analysis workflows.
        """
        normalized_data = []
        for holding in self.portfolio_data.get("holdings", []):
            normalized_data.append({
                "symbol": holding["symbol"].upper(),
                "quantity": float(holding["quantity"]),
                "purchase_price": float(holding["purchase_price"])
            })
        
        self.portfolio_data["holdings"] = normalized_data
        return self.portfolio_data
    
    def map_data_to_agents(self):
        """
        Map data to tax and signal analysis agents.
        """
        mapped_data = {
            "tax_analysis": self.portfolio_data,
            "signal_analysis": self.portfolio_data
        }
        return mapped_data

# Example usage
if __name__ == "__main__":
    agent = PortfolioDataAgent()
    portfolio = agent.fetch_portfolio_data(user_id=12345)
    print("Fetched Portfolio Data:", portfolio)
    
    is_valid, message = agent.validate_portfolio_data()
    print("Validation Result:", message)
    
    if is_valid:
        normalized_data = agent.normalize_portfolio_data()
        print("Normalized Portfolio Data:", normalized_data)
        
        mapped_data = agent.map_data_to_agents()
        print("Mapped Data:", mapped_data)
