import numpy as np
import pandas as pd
import backtrader as bt
import logging
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import os
# Import USI calculation functions from usi_calculation.py
from src.Indicators.usi import calculate_usi

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#################################
# USI DEFAULTS (global)
#################################
USI_DEFAULTS = {
    'period': 28,
    'smoothing_period': 4,
    'allocation': 1.0
}

def dict_to_params(d: dict) -> tuple:
    return tuple((k, v) for k, v in d.items())

def load_stock_data_from_excel(file_path, sheet_name='StockData'):
    """
    Load stock data from an Excel file
    
    Args:
        file_path (str): Path to the Excel file
        sheet_name (str, optional): Name of the sheet containing stock data. Defaults to 'StockData'.
    
    Returns:
        pandas.DataFrame: DataFrame containing stock data
    """
    try:
        # Read Excel file into DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Check if required columns exist
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"Missing required columns in Excel file: {missing_columns}")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Set Date as index and ensure it's datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Ensure all price and volume data are numeric
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        logging.info(f"Loaded {len(df)} data points from {file_path}")
        return df
        
    except Exception as e:
        logging.error(f"Error loading stock data from Excel: {e}")
        return pd.DataFrame()  # Return empty DataFrame

#####################################
# USI Indicator wrapped for BT using imported functions
#####################################
class USIIndicatorBT(bt.Indicator):
    lines = ('usi_signal',)
    params = dict_to_params(USI_DEFAULTS)

    def __init__(self):
        self.addminperiod(self.p.period + 2*self.p.smoothing_period)
        
        # Get data for USI calculation
        size = self.data.buflen()
        
        # Calculate USI
        self.usi_df = calculate_usi(
            df=data,  # This is a global variable set in __main__
            length=self.p.period,
            window=self.p.smoothing_period
        )

    # Assign indicator values to backtrader
    def once(self, start, end):
        for i in range(self.data.buflen()):
            self.lines.usi_signal[i] = self.usi_df[i] 


#######################################
# Strategy with One Buy Signal at a Time
#######################################
class USICrossStrategy(bt.Strategy):
    params = dict_to_params(USI_DEFAULTS)

    def __init__(self):
        self.usi_ind = USIIndicatorBT(self.data, 
                                      period=self.p.period,
                                      smoothing_period=self.p.smoothing_period)
        self.usi_signal = self.usi_ind.usi_signal
        self.order = None
        self.pending_entry = None

    def bullish_cross(self, prev_bar, current_bar):
        return prev_bar < 0 and current_bar >= 0

    def bearish_cross(self, prev_bar, current_bar):
        return prev_bar > 0 and current_bar <= 0

    def log_position(self):
        pos_size = self.position.size if self.position else 0
        pos_type = 'NONE'
        if pos_size > 0:
            pos_type = 'LONG'
        elif pos_size < 0:
            pos_type = 'SHORT'
        logging.info(f"{self.data.datetime.date(0)}: POSITION UPDATE: {pos_type} {pos_size} shares")

    def notify_order(self, order):
        date = self.data.datetime.date(0)
        if order.status in [order.Completed]:
            if order.isbuy():
                logging.info(f"{date}: BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}")
            elif order.issell():
                logging.info(f"{date}: SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}")

            self.log_position()

            # Enter pending position after close executes
            if self.pending_entry:
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash / price) * 0.95)
                if size > 0:
                    if self.pending_entry == 'LONG':
                        self.order = self.buy(size=size)
                        logging.info(f"{date}: BUY {size} shares at {price:.2f}")
                    elif self.pending_entry == 'SHORT':
                        self.order = self.sell(size=size)
                        logging.info(f"{date}: SELL {size} shares at {price:.2f}")
                self.pending_entry = None

            self.log_position()

        elif order.status in [order.Margin, order.Rejected]:
            logging.warning(f"{self.data.datetime.date(0)}: Order Failed - Margin/Rejected")
            self.order = None
            self.pending_entry = None

        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        date = self.data.datetime.date(0)
        if self.order:
            return  # Wait for pending order to complete

        usi_val = self.usi_signal[0]
        usi_prev = self.usi_signal[-1] if len(self.usi_signal) > 1 else 0

        if self.bullish_cross(usi_prev, usi_val):
            if self.position:
                if self.position.size < 0:  # Short position active
                    logging.info(f"{date}: CLOSING SHORT POSITION BEFORE GOING LONG")
                    self.order = self.close()
                    self.pending_entry = 'LONG'
            else:
                size = int((self.broker.getcash() / self.data.close[0]) * 0.95)
                if size > 0:
                    self.order = self.buy(size=size)
                    logging.info(f"{date}: BUY {size} shares at {self.data.close[0]:.2f}")

        elif self.bearish_cross(usi_prev, usi_val):
            if self.position:
                if self.position.size > 0:  # Long position active
                    logging.info(f"{date}: CLOSING LONG POSITION BEFORE GOING SHORT")
                    self.order = self.close()
                    self.pending_entry = 'SHORT'
            else:
                size = int((self.broker.getcash() / self.data.close[0]) * 0.95)
                if size > 0:
                    self.order = self.sell(size=size)
                    logging.info(f"{date}: SELL {size} shares at {self.data.close[0]:.2f}")


class BuyAndHold(bt.Strategy):
    params = (
        ('allocation', 1.0),  # Allocate 100% of the available cash to buy and hold (adjust as needed)
    )

    def __init__(self):
        pass  # No need for indicators in Buy-and-Hold strategy

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        # Check if we already have a position (buy once and hold)
        if not self.position:  # If not in a position
            cash = self.broker.getcash()  # Get available cash
            price = self.data.close[0]  # Current price of the asset
            size = (cash * self.params.allocation) // price  # Buy with the allocated cash
            self.buy(size=size)  # Execute the buy order with calculated size
            logging.info(f"{current_date}: BUY {size} shares at {price:.2f}")

#######################################
# Backtest Runner
#######################################
def run_backtest(strategy_class, data_feed, symbol_name, cash=10000, commission=0.001):
    cerebro = bt.Cerebro(runonce=True, preload=True)
    cerebro.addstrategy(strategy_class)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95) #prevent partial fills
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    logging.info(f"Running {strategy_class.__name__} Strategy...")
    result = cerebro.run()
    
    strat = result[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_drawdown_duration = drawdown.get('maxdrawdownperiod', 'N/A')

    logging.info(f"Returns Analysis {strategy_class.__name__}:")
    logging.info("\n%s", returns)

    # Print results
    print(f"\nResults for {symbol_name} using {strategy_class.__name__}:")
    print(f"  Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    print(f"  Total Return: {returns.get('rtot', 0)*100:.2f}%")
    print(f"  Avg Daily Return: {returns.get('ravg', 0)*100:.2f}%")
    print(f"  Avg Annual Return: {((1+returns.get('ravg', 0))**252 - 1)*100:.2f}%")
    print(f"  Max Drawdown: {drawdown.get('drawdown', 0)*100:.2f}%")
    print(f"  Max Drawdown Duration: {max_drawdown_duration}")

    logging.info("Generating plot...")
    figs = cerebro.plot(style='candle')
    
    return figs

if __name__ == '__main__':
    # Configuration
    cash = 10000
    commission = 0.001
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file_path = os.path.join(script_dir, f"stock_data.xlsx")
    sheet_name = 'StockData'  # Sheet name containing stock data
    symbol_name = "Stock"  # Default symbol name for output display

    # Load stock data from Excel
    data = load_stock_data_from_excel(excel_file_path, sheet_name)
    
    if data.empty:
        logging.error(f"No valid data found in {excel_file_path}. Exiting.")
        exit(1)
    
    # If the Excel has a Symbol column, use it to get the symbol name
    if 'Symbol' in data.columns:
        # Get the most common symbol (in case there are multiple)
        symbol_name = data['Symbol'].mode()[0]
        # Remove the Symbol column as it's not needed for backtesting
        data = data.drop('Symbol', axis=1)
    
    # Create backtrader data feed
    data_feed = bt.feeds.PandasData(dataname=data)
    
    print(f"\n*********************************************")
    print(f"*************** {symbol_name} USI CROSS *******************")
    print(f"*********************************************")
    usi_cross_figs = run_backtest(
        strategy_class=USICrossStrategy, 
        data_feed=data_feed,
        symbol_name=symbol_name,
        cash=cash, 
        commission=commission
    )
    
    print(f"\n*********************************************")
    print(f"************* {symbol_name} BUY AND HOLD ******************")
    print(f"*********************************************")
    buy_hold_figs = run_backtest(
        strategy_class=BuyAndHold, 
        data_feed=data_feed,
        symbol_name=symbol_name,
        cash=cash, 
        commission=commission
    )
    
    print("\nBacktesting complete. Results printed above.")