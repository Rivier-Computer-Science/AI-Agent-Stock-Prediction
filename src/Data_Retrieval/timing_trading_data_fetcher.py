from yahooquery import Ticker

class DataFetcher:
    def get_stock_news(self, stock):
        ticker = Ticker(stock)
        news = ticker.news()
        return news

    def get_earnings_date(self, stock):
        # Get earnings date using yahooquery
        ticker = Ticker(stock)
        earnings = ticker.earnings
        try:
            next_earnings_date = earnings['quarterly']['earningsDate'].iloc[0]
            return next_earnings_date
        except (KeyError, IndexError):
            return "Earnings date not found"
