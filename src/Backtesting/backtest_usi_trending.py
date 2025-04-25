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

# Set up detailed logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('USI_Backtest')

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
            logger.error(f"Missing required columns in Excel file: {missing_columns}")
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
        
        logger.info(f"Loaded {len(df)} data points from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading stock data from Excel: {e}")
        return pd.DataFrame()  # Return empty DataFrame

#####################################
# USI Indicator wrapped for BT using imported functions
#####################################
class USIIndicatorBT(bt.Indicator):
    lines = ('usi_signal',)
    params = dict_to_params(USI_DEFAULTS)

    def __init__(self):
        self.addminperiod(self.p.period + 2*self.p.smoothing_period)
        
        # Calculate USI
        try:
            # Get a snapshot of current data for debugging
            self.data_sample = data.head()
            logger.info(f"Calculating USI with period={self.p.period}, smoothing={self.p.smoothing_period}")
            
            # Fix for the divide by zero warning
            self.usi_df = calculate_usi(
                df=data,  # This is a global variable set in __main__
                length=self.p.period,
                window=self.p.smoothing_period
            )
            
            # Check for NaN or invalid values
            nan_count = np.isnan(self.usi_df).sum()
            if nan_count > 0:
                logger.warning(f"USI calculation contains {nan_count} NaN values")
                
            # Log sample of calculated values
            logger.info(f"USI first 5 values: {self.usi_df[:5]}")
            logger.info(f"USI last 5 values: {self.usi_df[-5:]}")
            
        except Exception as e:
            logger.error(f"Error in USI calculation: {e}")
            # Create a default array of zeros to prevent complete failure
            self.usi_df = np.zeros(len(data))

    # Assign indicator values to backtrader
    def once(self, start, end):
        try:
            for i in range(min(self.data.buflen(), len(self.usi_df))):
                # Use .iloc for positional indexing to fix the FutureWarning
                self.lines.usi_signal[i] = self.usi_df.iloc[i] if hasattr(self.usi_df, 'iloc') else self.usi_df[i]
        except Exception as e:
            logger.error(f"Error in USI signal assignment: {e}")


#######################################
# CycleDetector + USI Strategy Combined
#######################################
class USICrossWithTrendStrategy(bt.Strategy):
    params = dict_to_params(USI_DEFAULTS)

    def __init__(self):
        self.usi = USIIndicatorBT(self.data, 
                                  period=self.p.period,
                                  smoothing_period=self.p.smoothing_period)
        self.usi_signal = self.usi.usi_signal
        self.order = None
        self.pending_entry = None
        self.trend_data = self.data.trend_status  # Custom column for trend
        
        # Track trade history
        self.trade_count = 0
        self.last_signal = None
        
        # Debug counters
        self.trending_days = 0
        self.non_trending_days = 0
        self.bull_cross_count = 0
        self.bear_cross_count = 0

    def bullish_cross(self, prev_bar, current_bar):
        result = prev_bar < 0 and current_bar >= 0
        if result:
            self.bull_cross_count += 1
        return result

    def bearish_cross(self, prev_bar, current_bar):
        result = prev_bar > 0 and current_bar <= 0
        if result:
            self.bear_cross_count += 1
        return result

    def notify_order(self, order):
        date = self.data.datetime.date(0)
        if order.status in [order.Completed]:
            self.trade_count += 1
            if order.isbuy():
                logger.info(f"{date}: BUY EXECUTED at {order.executed.price:.2f}, size={order.executed.size}, value={order.executed.value:.2f}")
            elif order.issell():
                logger.info(f"{date}: SELL EXECUTED at {order.executed.price:.2f}, size={order.executed.size}, value={order.executed.value:.2f}")
        elif order.status in [order.Margin, order.Rejected]:
            logger.warning(f"{date}: Order Failed - {order.Status[order.status]}")
        self.order = None

    def next(self):
        # Get current date for logging
        current_date = self.data.datetime.date(0)
        
        # Skip if we already have a pending order
        if self.order:
            return

        # Get current indicator values
        usi_val = self.usi_signal[0]
        usi_prev = self.usi_signal[-1] if len(self.usi_signal) > 1 else 0
        trend_status = self.trend_data[0]
        
        # Count trend days for statistics
        if trend_status == 1:
            self.trending_days += 1
        else:
            self.non_trending_days += 1
        
        # Log current state every 10 bars
        if len(self) % 10 == 0 or self.bullish_cross(usi_prev, usi_val) or self.bearish_cross(usi_prev, usi_val):
            logger.info(f"{current_date} - Price: {self.data.close[0]:.2f}, USI: {usi_val:.4f}, USI Prev: {usi_prev:.4f}, Trend: {'Trending' if trend_status == 1 else 'Non-Trending'}, Position: {self.position.size}")

        # Enhanced logging for potential signals
        if usi_prev != usi_val:
            logger.debug(f"{current_date} - USI changed from {usi_prev:.4f} to {usi_val:.4f}")
            
        # Detect crossovers with detailed logging
        if self.bullish_cross(usi_prev, usi_val):
            logger.info(f"{current_date} - BULLISH CROSS DETECTED: USI crossed from {usi_prev:.4f} to {usi_val:.4f}")
            self.last_signal = "BULLISH"
            
        if self.bearish_cross(usi_prev, usi_val):
            logger.info(f"{current_date} - BEARISH CROSS DETECTED: USI crossed from {usi_prev:.4f} to {usi_val:.4f}")
            self.last_signal = "BEARISH"

        # Strategy logic with detailed logging
        if trend_status == 1:  # Trending
            if self.bullish_cross(usi_prev, usi_val):
                logger.info(f"{current_date} - SIGNAL: Bullish cross during trending period. Current position: {self.position.size}")
                if self.position.size <= 0:
                    if self.position.size < 0:
                        logger.info(f"{current_date} - ACTION: Closing short position before going long")
                        self.order = self.close()
                    else:
                        # Calculate position size
                        size = int((self.broker.getcash() / self.data.close[0]) * 0.95)
                        logger.info(f"{current_date} - ACTION: Opening long position with {size} shares")
                        self.order = self.buy(size=size)

            elif self.bearish_cross(usi_prev, usi_val):
                logger.info(f"{current_date} - SIGNAL: Bearish cross during trending period. Current position: {self.position.size}")
                if self.position.size >= 0:
                    if self.position.size > 0:
                        logger.info(f"{current_date} - ACTION: Closing long position before going short")
                        self.order = self.close()
                    else:
                        # Calculate position size
                        size = int((self.broker.getcash() / self.data.close[0]) * 0.95)
                        logger.info(f"{current_date} - ACTION: Opening short position with {size} shares")
                        self.order = self.sell(size=size)
            else:
                logger.debug(f"{current_date} - No signal during trending period. Maintaining current position.")

        else:  # Non-Trending → Always short
            if self.position.size >= 0:
                if self.position.size > 0:
                    logger.info(f"{current_date} - ACTION: Closing long position during non-trending period")
                    self.order = self.close()
                else:
                    # Calculate position size for short
                    size = int((self.broker.getcash() / self.data.close[0]) * 0.95)
                    logger.info(f"{current_date} - ACTION: Opening short position during non-trending period with {size} shares")
                    self.order = self.sell(size=size)
            else:
                logger.debug(f"{current_date} - Already short during non-trending period, maintaining position")

    def stop(self):
        """Called when backtest is complete to report statistics"""
        total_days = self.trending_days + self.non_trending_days
        logger.info("=" * 50)
        logger.info("STRATEGY STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total trading days: {total_days}")
        logger.info(f"Trending days: {self.trending_days} ({self.trending_days/total_days*100:.1f}%)")
        logger.info(f"Non-trending days: {self.non_trending_days} ({self.non_trending_days/total_days*100:.1f}%)")
        logger.info(f"Bullish crosses: {self.bull_cross_count}")
        logger.info(f"Bearish crosses: {self.bear_cross_count}")
        logger.info(f"Total trades executed: {self.trade_count}")
        logger.info("=" * 50)


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
            size = int((cash * self.params.allocation) / price)  # Buy with the allocated cash
            self.buy(size=size)  # Execute the buy order with calculated size
            logger.info(f"{current_date}: BUY {size} shares at {price:.2f}")

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
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    logger.info(f"Running {strategy_class.__name__} Strategy...")
    result = cerebro.run()
    
    strat = result[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    
    max_drawdown_duration = drawdown.get('maxdrawdownperiod', 'N/A')

    logger.info(f"Returns Analysis {strategy_class.__name__}:")
    logger.info("\n%s", returns)

    # Print trade statistics
    logger.info(f"Trade Analysis {strategy_class.__name__}:")
    total_trades = trades.get('total', {}).get('total', 0)
    won_trades = trades.get('won', {}).get('total', 0)
    lost_trades = trades.get('lost', {}).get('total', 0)
    win_rate = won_trades / total_trades * 100 if total_trades > 0 else 0
    
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Won Trades: {won_trades} ({win_rate:.1f}%)")
    logger.info(f"Lost Trades: {lost_trades}")

    # Print results
    print(f"\nResults for {symbol_name} using {strategy_class.__name__}:")
    print(f"  Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    print(f"  Total Return: {returns.get('rtot', 0)*100:.2f}%")
    print(f"  Avg Daily Return: {returns.get('ravg', 0)*100:.4f}%")
    print(f"  Avg Annual Return: {((1+returns.get('ravg', 0))**252 - 1)*100:.2f}%")
    print(f"  Max Drawdown: {drawdown.get('max', {}).get('drawdown', 0)*100:.2f}%")
    print(f"  Max Drawdown Duration: {max_drawdown_duration}")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")

    logger.info("Generating plot...")
    figs = cerebro.plot(style='candle')
    
    return figs

if __name__ == "__main__":
    import logging
    import pandas as pd
    from src.Indicators.cycle_trend_detection import CycleDetector

    # Logging setup - more detailed for debugging
    logger = logging.getLogger('USI_Backtest')
    logger.setLevel(logging.INFO)
    
    # Add file handler for persistent logs
    fh = logging.FileHandler('usi_backtest_debug.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # === Load Excel Data ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file_path = os.path.join(script_dir, f"stock_data.xlsx")
    sheet_name = 'StockData'  # Sheet name containing stock data
    symbol_name = "Stock"  # Default symbol name for output display
    
    # Load stock data from Excel
    data = load_stock_data_from_excel(excel_file_path, sheet_name)

    logger.info(f"Data index: {data.index}")
    logger.info(f"Data columns: {data.columns}")
    logger.info(f"Data shape: {data.shape}")

    if 'Symbol' in data.columns:
        symbol_name = data['Symbol'].mode()[0]
        data.drop(columns=['Symbol'], inplace=True)
    else:
        symbol_name = sheet_name

    # === Run Cycle Detector ===
    cycle_detector = CycleDetector(upper_bound=48, lower_bound=10)
    close_prices = data['Close'].tolist()  # If `data` is a DataFrame, otherwise handle as Series
    
    logger.info("Running cycle detection...")
    dominant_cycles = cycle_detector.detect_cycles(close_prices)
    trend_labels = cycle_detector.classify_trend(dominant_cycles)
    
    # Log trend classification statistics
    trending_count = trend_labels.count("Trending")
    total_count = len(trend_labels)
    trending_pct = trending_count / total_count * 100
    
    logger.info(f"Trend classification: {trending_count} out of {total_count} periods are trending ({trending_pct:.1f}%)")
    
    # Convert trend labels to binary
    trend_labels_series = pd.Series(trend_labels)
    data['trend_status'] = (trend_labels_series == "Trending").astype(int)
    
    # Log first and last few trend values
    logger.info(f"First 10 trend values: {data['trend_status'].head(10).tolist()}")
    logger.info(f"Last 10 trend values: {data['trend_status'].tail(10).tolist()}")

    # === Custom Data Feed with Trend Line ===
    class CustomData(bt.feeds.PandasData):
        lines = ('trend_status',)
        params = (('trend_status', -1),)

    data_feed = CustomData(dataname=data)

    # === Run Backtest ===
    cash = 100000
    commission = 0.001  # 0.1%

    print(f"\n********* {symbol_name} USI + TREND STRATEGY *********")
    figs = run_backtest(
        strategy_class=USICrossWithTrendStrategy,
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