import yfinance as yf

ticker = "AAPL"  # Try another ticker to check if SPY is the issue
df = yf.download(ticker, start="2020-01-01")
print(df.head())



#Run in terminal to check if Yahoo Finance is reachable
"""
curl -I https://query1.finance.yahoo.com/v7/finance/download/AAPL
"""
