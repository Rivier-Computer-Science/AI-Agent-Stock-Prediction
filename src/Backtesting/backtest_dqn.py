
 import os
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
plt.switch_backend("agg")

import streamlit as st
from datetime import datetime
import logging
import backtrader as bt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
from deap import base, creator, tools, algorithms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Data_Retrieval.data_fetcher import DataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Data Preprocessor for ML compatibility
class DataPreprocessor:
    def preprocess(self, df):
        df = df.copy()
        df.fillna(method='ffill', inplace=True)
        df['Returns'] = df['Close'].pct_change()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df = (df - df.mean()) / df.std()
        df.fillna(0, inplace=True)
        return df

    def calculate_rsi(self, prices, period):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

# DQN Strategy Generator
class DQNStrategy(bt.Strategy):
    params = (
        ('period', 14),
        ('allocation', 1.0),
        ('state_size', 10),
        ('action_size', 3),
        ('gamma', 0.95),
        ('epsilon', 1.0),
        ('epsilon_min', 0.01),
        ('epsilon_decay', 0.995),
        ('learning_rate', 0.001),
    )

    def __init__(self):
        self.trade_log = []
        self.model = self.build_dqn_model()
        self.memory = deque(maxlen=2000)
        self.preprocessor = DataPreprocessor()
        self.data_df = pd.DataFrame({
            'Close': [self.data.close[i] for i in range(-self.p.period, 0)],
            'High': [self.data.high[i] for i in range(-self.p.period, 0)],
            'Low': [self.data.low[i] for i in range(-self.p.period, 0)],
        })
        self.data_processed = self.preprocessor.preprocess(self.data_df)

    def build_dqn_model(self):
        model = Sequential([
            Dense(24, input_dim=self.p.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.p.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate))
        return model

    def get_state(self):
        return np.array(self.data_processed.iloc[-self.p.state_size:][['Close', 'RSI']].values).flatten()

    def act(self, state):
        if random.random() <= self.p.epsilon:
            return random.randrange(self.p.action_size)
        return np.argmax(self.model.predict(state.reshape(1, -1))[0])

    def replay(self):
        if len(self.memory) < 32:
            return
        batch = random.sample(self.memory, 32)
        states = np.array([t[0] for t in batch])
        next_states = np.array([t[3] for t in batch])
        targets = self.model.predict(states)
        next_qs = self.model.predict(next_states)
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            target = reward if done else reward + self.p.gamma * np.max(next_qs[i])
            targets[i][action] = target
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.p.epsilon > self.p.epsilon_min:
            self.p.epsilon *= self.p.epsilon_decay

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        state = self.get_state()
        action = self.act(state)
        reward = self.data.close[0] - self.data.close[-1] if action == 1 else 0
        done = len(self.datas[0]) - 1 == self.data.buflen()
        next_state = self.get_state()
        self.memory.append((state, action, reward, next_state, done))
        self.replay()

        if not self.position:
            if action == 1:
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * self.p.allocation) // price)
                if size > 0:
                    self.buy(size=size)
                    msg = f"{current_date}: BUY {size} shares at {price:.2f} (DQN)"
                    self.trade_log.append(msg)
                    logging.info(msg)
        else:
            if action == 2:
                size = self.position.size
                price = self.data.close[0]
                self.sell(size=size)
                msg = f"{current_date}: SELL {size} shares at {price:.2f} (DQN)"
                self.trade_log.append(msg)
                logging.info(msg)

# DQN Strategy Wrapper to fix state size mismatch
class DQNStrategyWrapper(DQNStrategy):
    def get_state(self):
        state = super().get_state()
        expected_size = self.p.state_size
        if len(state) > expected_size:
            state = state[:expected_size]
        elif len(state) < expected_size:
            state = np.pad(state, (0, expected_size - len(state)), mode='constant')
        return state

# GA Strategy Generator
class GAStrategy(bt.Strategy):
    params = (
        ('period', 14),
        ('allocation', 1.0),
        ('population_size', 50),
        ('generations', 20),
    )

    def __init__(self):
        self.trade_log = []
        self.preprocessor = DataPreprocessor()
        self.data_df = pd.DataFrame({
            'Close': [self.data.close[i] for i in range(-self.p.period, 0)],
            'High': [self.data.high[i] for i in range(-self.p.period, 0)],
            'Low': [self.data.low[i] for i in range(-self.p.period, 0)],
        })
        self.data_processed = self.preprocessor.preprocess(self.data_df)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=2)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.population = self.toolbox.population(n=self.p.population_size)
        self.optimize()

    def evaluate(self, individual):
        rsi_buy, rsi_sell = individual
        returns = []
        for i in range(1, len(self.data_processed)):
            rsi = self.data_processed['RSI'].iloc[i]
            price = self.data_processed['Close'].iloc[i]
            if rsi < rsi_buy:
                returns.append(price - self.data_processed['Close'].iloc[i-1])
            elif rsi > rsi_sell:
                returns.append(0)
        return sum(returns),

    def optimize(self):
        for gen in range(self.p.generations):
            offspring = algorithms.varAnd(self.population, self.toolbox, cxpb=0.7, mutpb=0.3)
            fits = list(map(self.toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit
            self.population = self.toolbox.select(offspring, len(self.population))
        self.best_individual = tools.selBest(self.population, 1)[0]

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        rsi = self.data_processed['RSI'].iloc[-1]
        rsi_buy, rsi_sell = self.best_individual

        if not self.position:
            if rsi < rsi_buy:
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = int((cash * self.p.allocation) // price)
                if size > 0:
                    self.buy(size=size)
                    msg = f"{current_date}: BUY {size} shares at {price:.2f} (GA)"
                    self.trade_log.append(msg)
                    logging.info(msg)
        else:
            if rsi > rsi_sell:
                size = self.position.size
                price = self.data.close[0]
                self.sell(size=size)
                msg = f"{current_date}: SELL {size | shares at {price:.2f} (GA)"
                self.trade_log.append(msg)
                logging.info(msg)

# Synthetic Data Generator for Validation
def generate_synthetic_data(start_date, end_date, initial_price=100):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    prices = [initial_price]
    for _ in range(1, len(dates)):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))
    df = pd.DataFrame({
        'Close': prices,
        'High': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'Open': prices,
        'Volume': [1000000 * np.random.uniform(0.5, 1.5) for _ in prices]
    }, index=dates)
    return df

def run_backtest(strategy_class, data_feed, cash=10000, commission=0.001):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)
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
    perf_summary = {
        "Sharpe Ratio": sharpe.get('sharperatio', 0),
        "Total Return": returns.get('rtot', 0),
        "Avg Daily Return": returns.get('ravg', 0),
        "Avg Annual Return": ((1 + returns.get('ravg', 0)) ** 252 - 1),
        "Max Drawdown": drawdown.drawdown,
        "Max Drawdown Duration": max_drawdown_duration
    }
    figs = cerebro.plot(iplot=False, show=False)
    fig = figs[0][0]
    return perf_summary, strat.trade_log, fig

def main():
    st.title("ML Strategy Backtest")
    st.sidebar.header("Backtest Parameters")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1).date())
    end_date = st.sidebar.date_input("End Date", value=datetime.today().date())
    initial_cash = st.sidebar.number_input("Initial Cash", value=10000)
    commission = st.sidebar.number_input("Commission", value=0.001, step=0.0001)
    period = st.sidebar.number_input("Period", value=14, step=1)
    model_type = st.sidebar.selectbox("Model Type", ["DQN", "GA"])
    use_synthetic = st.sidebar.checkbox("Use Synthetic Data", value=False)

    if st.sidebar.button("Run Backtest"):
        st.write("Fetching data...")
        if use_synthetic:
            data = generate_synthetic_data(start_date, end_date)
        else:
            data = DataFetcher().get_stock_data(symbol=ticker, start_date=start_date, end_date=end_date)
        data_feed = bt.feeds.PandasData(dataname=data, fromdate=start_date, todate=end_date)
        st.write("Running backtest. Please wait...")
        strategy_class = {'DQN': DQNStrategyWrapper, 'GA': GAStrategy}[model_type]
        perf_summary, trade_log, fig = run_backtest(
            strategy_class=strategy_class,
            data_feed=data_feed,
            cash=initial_cash,
            commission=commission
        )
        st.subheader("Performance Summary")
        st.write(f"**Sharpe Ratio:** {perf_summary['Sharpe Ratio']:.2f}")
        st.write(f"**Total Return:** {perf_summary['Total Return']*100:.2f}%")
        st.write(f"**Avg Daily Return:** {perf_summary['Avg Daily Return']*100:.2f}%")
        st.write(f"**Avg Annual Return:** {perf_summary['Avg Annual Return']*100:.2f}%")
        st.write(f"**Max Drawdown:** {perf_summary['Max Drawdown']*100:.2f}%")
        st.write(f"**Max Drawdown Duration:** {perf_summary['Max Drawdown Duration']}")
        st.subheader("Trade Log")
        if trade_log:
            for t in trade_log:
                st.write(t)
        else:
            st.write("No trades executed.")
        st.subheader("Backtest Chart")
        st.pyplot(fig)

if __name__ == '__main__':
    main()
