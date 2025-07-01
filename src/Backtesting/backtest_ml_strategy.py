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
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
from deap import base, creator, tools, algorithms
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
import warnings
import uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.Data_Retrieval.data_fetcher import DataFetcher
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
class DataPreprocessor:
    def preprocess(self, df):
        df = df.copy()
        df.fillna(method='ffill', inplace=True)
        df['Returns'] = df['Close'].pct_change()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df = (df - df.mean(numeric_only=True)) / df.std(numeric_only=True)
        df.fillna(0, inplace=True)
        return df
    def calculate_rsi(self, prices, period):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
        return 100 - (100 / (1 + rs))
class DQNStrategy(bt.Strategy):
    params = (
        ('period', 14),
        ('allocation', 0.3),
        ('state_size', 20),
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
        self.data_processed = {}
        self.actions_history = {data._name: [] for data in self.datas}
        try:
            for data in self.datas:
                df = pd.DataFrame({
                    'Close': [data.close[i] for i in range(-self.p.period, 0)],
                    'High': [data.high[i] for i in range(-self.p.period, 0)],
                    'Low': [data.low[i] for i in range(-self.p.period, 0)],
                })
                self.data_processed[data._name] = self.preprocessor.preprocess(df)
        except Exception as e:
            logging.error(f"Data preprocessing error: {str(e)}")
            st.error(f"Error preprocessing data: {str(e)}. Please check data integrity.")
    def build_dqn_model(self):
        model = Sequential([
            Dense(32, input_dim=self.p.state_size, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.p.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate))
        return model
    def get_state(self, data):
        df = self.data_processed[data._name]
        features = df[['Close', 'RSI', 'MA20', 'Volatility']].iloc[-self.p.state_size//4:].values.flatten()
        expected_size = self.p.state_size
        if len(features) > expected_size:
            features = features[:expected_size]
        elif len(features) < expected_size:
            features = np.pad(features, (0, expected_size - len(features)), mode='constant')
        return features
    def act(self, state):
        if random.random() <= self.p.epsilon:
            return random.randrange(self.p.action_size)
        try:
            return np.argmax(self.model.predict(state.reshape(1, -1), verbose=0)[0])
        except Exception as e:
            logging.error(f"Model prediction error: {str(e)}")
            return random.randrange(self.p.action_size)
    def replay(self):
        if len(self.memory) < 32:
            return
        try:
            batch = random.sample(self.memory, 32)
            states = np.array([t[0] for t in batch])
            next_states = np.array([t[3] for t in batch])
            targets = self.model.predict(states, verbose=0)
            next_qs = self.model.predict(next_states, verbose=0)
            for i, (state, action, reward, next_state, done) in enumerate(batch):
                target = reward if done else reward + self.p.gamma * np.max(next_qs[i])
                targets[i][action] = target
            self.model.fit(states, targets, epochs=1, verbose=0)
            if self.p.epsilon > self.p.epsilon_min:
                self.p.epsilon *= self.p.epsilon_decay
        except Exception as e:
            logging.error(f"Replay error: {str(e)}")
            st.warning("Issue during model training. Continuing with current model.")
    def extract_rules(self, data_name):
        try:
            X = np.array([self.get_state(data) for data in self.datas if data._name == data_name])
            y = np.array(self.actions_history[data_name][-len(X):]) if self.actions_history[data_name] else np.zeros(len(X))
            if len(X) == 0 or len(y) == 0:
                return "No rules extracted: insufficient data."
            tree = DecisionTreeClassifier(max_depth=3)
            tree.fit(X, y)
            output = StringIO()
            from sklearn.tree import export_text
            tree_text = export_text(tree, feature_names=['Feature'+str(i) for i in range(X.shape[1])])
            print(tree_text, file=output)
            return output.getvalue()
        except Exception as e:
            logging.error(f"Rule extraction error: {str(e)}")
            return f"Error extracting rules: {str(e)}"
    def next(self):
        for data in self.datas:
            try:
                current_date = data.datetime.date(0)
                state = self.get_state(data)
                action = self.act(state)
                self.actions_history[data._name].append(action)
                reward = data.close[0] - data.close[-1] if action == 1 else 0
                done = len(data) - 1 == data.buflen()
                next_state = self.get_state(data)
                self.memory.append((state, action, reward, next_state, done))
                self.replay()
                if not self.getposition(data):
                    if action == 1:
                        cash = self.broker.getcash()
                        price = data.close[0]
                        size = int((cash * self.p.allocation) // price)
                        if size > 0:
                            self.buy(data=data, size=size)
                            msg = f"{current_date}: BUY {size} shares of {data._name} at {price:.2f} (DQN)"
                            self.trade_log.append(msg)
                            logging.info(msg)
                else:
                    if action == 2:
                        size = self.getposition(data).size
                        price = data.close[0]
                        self.sell(data=data, size=size)
                        msg = f"{current_date}: SELL {size} shares of {data._name} at {price:.2f} (DQN)"
                        self.trade_log.append(msg)
                        logging.info(msg)
            except Exception as e:
                logging.error(f"Strategy execution error: {str(e)}")
                st.warning(f"Error executing strategy for {data._name}: {str(e)}")
class GAStrategy(bt.Strategy):
    params = (
        ('period', 14),
        ('allocation', 0.3),
        ('population_size', 50),
        ('generations', 20),
    )
    def __init__(self):
        self.trade_log = []
        self.preprocessor = DataPreprocessor()
        self.data_processed = {}
        try:
            for data in self.datas:
                df = pd.DataFrame({
                    'Close': [data.close[i] for i in range(-self.p.period, 0)],
                    'High': [data.high[i] for i in range(-self.p.period, 0)],
                    'Low': [data.low[i] for i in range(-self.p.period, 0)],
                })
                self.data_processed[data._name] = self.preprocessor.preprocess(df)
        except Exception as e:
            logging.error(f"Data preprocessing error: {str(e)}")
            st.error(f"Error preprocessing data: {str(e)}. Please check data integrity.")
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=4)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.population = self.toolbox.population(n=self.p.population_size)
        self.best_individuals = {data._name: None for data in self.datas}
        self.fitness_history = {data._name: [] for data in self.datas}
        self.optimize()
    def evaluate(self, individual, data_name):
        rsi_buy, rsi_sell, ma_buy, vol_threshold = individual
        try:
            returns = []
            data_processed = self.data_processed[data_name]
            for i in range(1, len(data_processed)):
                rsi = data_processed['RSI'].iloc[i]
                ma = data_processed['MA20'].iloc[i]
                vol = data_processed['Volatility'].iloc[i]
                price = data_processed['Close'].iloc[i]
                if rsi < rsi_buy and ma > 0 and vol < vol_threshold:
                    returns.append(price - data_processed['Close'].iloc[i-1])
                elif rsi > rsi_sell:
                    returns.append(0)
            return sum(returns) if returns else 0,
        except Exception as e:
            logging.error(f"Evaluation error: {str(e)}")
            return 0,
    def optimize(self):
        try:
            for data in self.datas:
                for gen in range(self.p.generations):
                    offspring = algorithms.varAnd(self.population, self.toolbox, cxpb=0.7, mutpb=0.3)
                    fits = list(map(lambda ind: self.toolbox.evaluate(ind, data._name), offspring))
                    for ind, fit in zip(offspring, fits):
                        ind.fitness.values = fit
                    self.population = self.toolbox.select(offspring, len(self.population))
                    self.fitness_history[data._name].append(max([ind.fitness.values[0] for ind in self.population]))
                self.best_individuals[data._name] = tools.selBest(self.population, 1)[0]
        except Exception as e:
            logging.error(f"Optimization error: {str(e)}")
            st.warning(f"Optimization failed for {data._name}: {str(e)}")
    def extract_rules(self, data_name):
        try:
            if self.best_individuals[data_name] is None:
                return "No rules extracted: optimization failed."
            rsi_buy, rsi_sell, ma_buy, vol_threshold = self.best_individuals[data_name]
            stability = np.std(self.fitness_history[data_name][-5:]) if len(self.fitness_history[data_name]) >= 5 else 0
            rules = f"""
Buy Rule: RSI < {rsi_buy:.2f} AND MA20 > 0 AND Volatility < {vol_threshold:.2f}
Sell Rule: RSI > {rsi_sell:.2f}
Stability Metric (Std of last 5 gen fitness): {stability:.4f}
            """
            return rules
        except Exception as e:
            logging.error(f"Rule extraction error: {str(e)}")
            return f"Error extracting rules: {str(e)}"
    def next(self):
        for data in self.datas:
            try:
                current_date = data.datetime.date(0)
                df = self.data_processed[data._name]
                rsi = df['RSI'].iloc[-1]
                ma = df['MA20'].iloc[-1]
                vol = df['Volatility'].iloc[-1]
                rsi_buy, rsi_sell, ma_buy, vol_threshold = self.best_individuals[data._name]
                if not self.getposition(data):
                    if rsi < rsi_buy and ma > 0 and vol < vol_threshold:
                        cash = self.broker.getcash()
                        price = data.close[0]
                        size = int((cash * self.p.allocation) // price)
                        if size > 0:
                            self.buy(data=data, size=size)
                            msg = f"{current_date}: BUY {size} shares of {data._name} at {price:.2f} (GA)"
                            self.trade_log.append(msg)
                            logging.info(msg)
                else:
                    if rsi > rsi_sell:
                        size = self.getposition(data).size
                        price = data.close[0]
                        self.sell(data=data, size=size)
                        msg = f"{current_date}: SELL {size} shares of {data._name} at {price:.2f} (GA)"
                        self.trade_log.append(msg)
                        logging.info(msg)
            except Exception as e:
                logging.error(f"Strategy execution error: {str(e)}")
                st.warning(f"Error executing strategy for {data._name}: {str(e)}")
def generate_synthetic_data(start_date, end_date, initial_price=100, ticker="SYNTH"):
    try:
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
        df.name = ticker
        return df
    except Exception as e:
        logging.error(f"Synthetic data generation error: {str(e)}")
        st.error(f"Error generating synthetic data: {str(e)}")
        return pd.DataFrame()
def export_strategy(strategy_class, perf_summary, trade_log, rules, ticker_list, filename_prefix="strategy"):
    try:
        unique_id = str(uuid.uuid4())[:8]
        py_content = f"""
import backtrader as bt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
from deap import base, creator, tools, algorithms

class DataPreprocessor:
    def preprocess(self, df):
        df = df.copy()
        df.fillna(method='ffill', inplace=True)
        df['Returns'] = df['Close'].pct_change()
        df['RSI'] = self.calculate_rsi(df['Close'], 14)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df = (df - df.mean(numeric_only=True)) / df.std(numeric_only=True)
        df.fillna(0, inplace=True)
        return df
    def calculate_rsi(self, prices, period):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
        return 100 - (100 / (1 + rs))

class {strategy_class.__name__}(bt.Strategy):
    params = {strategy_class.params}
    def __init__(self):
        self.trade_log = []
        self.preprocessor = DataPreprocessor()
        self.data_processed = {{data._name: self.preprocessor.preprocess(pd.DataFrame({{
            'Close': [data.close[i] for i in range(-self.p.period, 0)],
            'High': [data.high[i] for i in range(-self.p.period, 0)],
            'Low': [data.low[i] for i in range(-self.p.period, 0)],
        }})) for data in self.datas}}
        {'self.model = self.build_dqn_model()' if strategy_class.__name__ == 'DQNStrategy' else ''}
        {'self.memory = deque(maxlen=2000)' if strategy_class.__name__ == 'DQNStrategy' else ''}
        {'self.actions_history = {data._name: [] for data in self.datas}' if strategy_class.__name__ == 'DQNStrategy' else ''}
        {'creator.create("FitnessMax", base.Fitness, weights=(1.0,)); creator.create("Individual", list, fitness=creator.FitnessMax); self.toolbox = base.Toolbox(); self.toolbox.register("attr_float", random.uniform, 0, 1); self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=4); self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual); self.toolbox.register("evaluate", self.evaluate); self.toolbox.register("mate", tools.cxBlend, alpha=0.5); self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2); self.toolbox.register("select", tools.selTournament, tournsize=3); self.population = self.toolbox.population(n=self.p.population_size); self.best_individuals = {data._name: None for data in self.datas}; self.fitness_history = {data._name: [] for data in self.datas}; self.optimize()' if strategy_class.__name__ == 'GAStrategy' else ''}
    {'def build_dqn_model(self): model = Sequential([Dense(32, input_dim=self.p.state_size, activation="relu"), Dense(32, activation="relu"), Dense(self.p.action_size, activation="linear")]); model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.p.learning_rate)); return model' if strategy_class.__name__ == 'DQNStrategy' else ''}
    {'def get_state(self, data): df = self.data_processed[data._name]; features = df[["Close", "RSI", "MA20", "Volatility"]].iloc[-self.p.state_size//4:].values.flatten(); expected_size = self.p.state_size; if len(features) > expected_size: features = features[:expected_size]; elif len(features) < expected_size: features = np.pad(features, (0, expected_size - len(features)), mode="constant"); return features' if strategy_class.__name__ == 'DQNStrategy' else ''}
    {'def act(self, state): if random.random() <= self.p.epsilon: return random.randrange(self.p.action_size); return np.argmax(self.model.predict(state.reshape(1, -1), verbose=0)[0])' if strategy_class.__name__ == 'DQNStrategy' else ''}
    {'def replay(self): if len(self.memory) < 32: return; batch = random.sample(self.memory, 32); states = np.array([t[0] for t in batch]); next_states = np.array([t[3] for t in batch]); targets = self.model.predict(states, verbose=0); next_qs = self.model.predict(next_states, verbose=0); for i, (state, action, reward, next_state, done) in enumerate(batch): target = reward if done else reward + self.p.gamma * np.max(next_qs[i]); targets[i][action] = target; self.model.fit(states, targets, epochs=1, verbose=0); if self.p.epsilon > self.p.epsilon_min: self.p.epsilon *= self.p.epsilon_decay' if strategy_class.__name__ == 'DQNStrategy' else ''}
    {'def evaluate(self, individual, data_name): rsi_buy, rsi_sell, ma_buy, vol_threshold = individual; returns = []; data_processed = self.data_processed[data_name]; for i in range(1, len(data_processed)): rsi = data_processed["RSI"].iloc[i]; ma = data_processed["MA20"].iloc[i]; vol = data_processed["Volatility"].iloc[i]; price = data_processed["Close"].iloc[i]; if rsi < rsi_buy and ma > 0 and vol < vol_threshold: returns.append(price - data_processed["Close"].iloc[i-1]); elif rsi > rsi_sell: returns.append(0); return sum(returns) if returns else 0,' if strategy_class.__name__ == 'GAStrategy' else ''}
    {'def optimize(self): for data in self.datas: for gen in range(self.p.generations): offspring = algorithms.varAnd(self.population, self.toolbox, cxpb=0.7, mutpb=0.3); fits = list(map(lambda ind: self.toolbox.evaluate(ind, data._name), offspring)); for ind, fit in zip(offspring, fits): ind.fitness.values = fit; self.population = self.toolbox.select(offspring, len(self.population)); self.fitness_history[data._name].append(max([ind.fitness.values[0] for ind in self.population])); self.best_individuals[data._name] = tools.selBest(self.population, 1)[0]' if strategy_class.__name__ == 'GAStrategy' else ''}
    def next(self):
        for data in self.datas:
            current_date = data.datetime.date(0)
            df = self.data_processed[data._name]
            rsi = df['RSI'].iloc[-1]
            {'self.actions_history[data._name].append(self.act(self.get_state(data)))' if strategy_class.__name__ == 'DQNStrategy' else ''}
            {'ma = df["MA20"].iloc[-1]; vol = df["Volatility"].iloc[-1]; rsi_buy, rsi_sell, ma_buy, vol_threshold = self.best_individuals[data._name]' if strategy_class.__name__ == 'GAStrategy' else ''}
            {'action = self.act(self.get_state(data)); reward = data.close[0] - data.close[-1] if action == 1 else 0; done = len(data) - 1 == data.buflen(); next_state = self.get_state(data); self.memory.append((self.get_state(data), action, reward, next_state, done)); self.replay()' if strategy_class.__name__ == 'DQNStrategy' else ''}
            if not self.getposition(data):
                {'if action == 1:' if strategy_class.__name__ == 'DQNStrategy' else 'if rsi < rsi_buy and ma > 0 and vol < vol_threshold:'}
                    cash = self.broker.getcash()
                    price = data.close[0]
                    size = int((cash * self.p.allocation) // price)
                    if size > 0:
                        self.buy(data=data, size=size)
                        msg = f"{{current_date}}: BUY {{size}} shares of {{data._name}} at {{price:.2f}} ({strategy_class.__name__[:2]})"
                        self.trade_log.append(msg)
            else:
                {'if action == 2:' if strategy_class.__name__ == 'DQNStrategy' else 'if rsi > rsi_sell:'}
                    size = self.getposition(data).size
                    price = data.close[0]
                    self.sell(data=data, size=size)
                    msg = f"{{current_date}}: SELL {{size}} shares of {{data._name}} at {{price:.2f}} ({strategy_class.__name__[:2]})"
                    self.trade_log.append(msg)
    {'def extract_rules(self, data_name): X = np.array([self.get_state(data) for data in self.datas if data._name == data_name]); y = np.array(self.actions_history[data_name][-len(X):]) if self.actions_history[data_name] else np.zeros(len(X)); if len(X) == 0 or len(y) == 0: return "No rules extracted: insufficient data."; from sklearn.tree import DecisionTreeClassifier; tree = DecisionTreeClassifier(max_depth=3); tree.fit(X, y); from io import StringIO; output = StringIO(); from sklearn.tree import export_text; tree_text = export_text(tree, feature_names=["Feature"+str(i) for i in range(X.shape[1])]); print(tree_text, file=output); return output.getvalue()' if strategy_class.__name__ == 'DQNStrategy' else ''}
    {'def extract_rules(self, data_name): if self.best_individuals[data_name] is None: return "No rules extracted: optimization failed."; rsi_buy, rsi_sell, ma_buy, vol_threshold = self.best_individuals[data_name]; stability = np.std(self.fitness_history[data_name][-5:]) if len(self.fitness_history[data_name]) >= 5 else 0; rules = f"""Buy Rule: RSI < {{rsi_buy:.2f}} AND MA20 > 0 AND Volatility < {{vol_threshold:.2f}}\\nSell Rule: RSI > {{rsi_sell:.2f}}\\nStability Metric (Std of last 5 gen fitness): {{stability:.4f}}"""; return rules' if strategy_class.__name__ == 'GAStrategy' else ''}
"""
        with open(f"{filename_prefix}_{strategy_class.__name__}_{unique_id}.py", "w") as f:
            f.write(py_content)
        latex_content = f"""
\\documentclass{{article}}
\\usepackage{{geometry}}
\\usepackage{{amsmath}}
\\usepackage{{booktabs}}
\\usepackage{{parskip}}
\\usepackage{{fancyhdr}}
\\usepackage{{lastpage}}
\\geometry{{margin=1in}}
\\pagestyle{{fancy}}
\\fancyhf{{}}
\\rhead{{Strategy Report: {strategy_class.__name__}}}
\\lhead{{{', '.join(ticker_list)}}}
\\rfoot{{Page \\thepage\\ of \\pageref{{LastPage}}}}
\\begin{{document}}
\\title{{Trading Strategy Report: {strategy_class.__name__}}}
\\author{{Generated by ML Strategy Backtest}}
\\date{{\\today}}
\\maketitle
\\section{{Performance Summary}}
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|r|}}
\\hline
Metric & Value \\\\
\\hline
Sharpe Ratio & {perf_summary['Sharpe Ratio']:.2f} \\\\
Total Return & {perf_summary['Total Return']*100:.2f}\\% \\\\
Avg Daily Return & {perf_summary['Avg Daily Return']*100:.2f}\\% \\\\
Avg Annual Return & {perf_summary['Avg Annual Return']*100:.2f}\\% \\\\
Max Drawdown & {perf_summary['Max Drawdown']:.2f}\\% \\\\
Max Drawdown Duration & {perf_summary['Max Drawdown Duration']} \\\\
\\hline
\\end{{tabular}}
\\caption{{Performance Metrics}}
\\end{{table}}
\\section{{Trading Rules}}
{rules.replace('<', '$<$').replace('>', '$>$')}
\\section{{Trade Log}}
\\begin{{itemize}}
{' '.join([f'\\item {{t}}' for t in trade_log[:50]])}
\\end{{itemize}}
\\end{{document}}
"""
        with open(f"{filename_prefix}_{strategy_class.__name__}_{unique_id}.tex", "w") as f:
            f.write(latex_content)
        st.success(f"Exported strategy to {filename_prefix}_{strategy_class.__name__}_{unique_id}.py and .tex")
    except Exception as e:
        logging.error(f"Export error: {str(e)}")
        st.error(f"Error exporting strategy: {str(e)}")
def run_backtest(strategy_class, data_feeds, cash=10000, commission=0.001):
    try:
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class)
        for data_feed in data_feeds:
            cerebro.adddata(data_feed)
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.01)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        logging.info(f"Running {strategy_class.__name__} Strategy...")
        result = cerebro.run()
        strat = result[0]
        sharpe = strat.analyzers.sharpe.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        perf_summary = {
            "Sharpe Ratio": sharpe.get('sharperatio', 0) if sharpe.get('sharperatio') is not None else 0,
            "Total Return": returns.get('rtot', 0),
            "Avg Daily Return": returns.get('ravg', 0),
            "Avg Annual Return": ((1 + returns.get('ravg', 0)) ** 252 - 1),
            "Max Drawdown": drawdown.get('maxdrawdown', 0),
            "Max Drawdown Duration": drawdown.get('maxdrawdownperiod', 'N/A'),
            "Total Trades": trades.get('total', {}).get('total', 0)
        }
        figs = cerebro.plot(iplot=False, show=False)
        fig = figs[0][0] if figs else plt.figure()
        rules = {}
        for data in data_feeds:
            rules[data._name] = strat.extract_rules(data._name)
        if perf_summary["Sharpe Ratio"] < 0:
            st.warning("Negative Sharpe Ratio detected. Strategy may be underperforming.")
        if perf_summary["Total Trades"] == 0:
            st.warning("No trades executed. Check data or strategy parameters.")
        return perf_summary, strat.trade_log, fig, rules
    except Exception as e:
        logging.error(f"Backtest error: {str(e)}")
        st.error(f"Error running backtest: {str(e)}. Please verify inputs and data.")
        return {}, [], plt.figure(), {}
def main():
    st.set_page_config(page_title="ML Strategy Backtest", layout="wide")
    st.title("Machine Learning Strategy Backtest")
    st.sidebar.header("Backtest Parameters")
    tickers = st.sidebar.text_input("Tickers (comma-separated)", value="SPY,AAPL,MSFT", help="Enter valid stock tickers")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1).date(), help="Select start date for data")
    end_date = st.sidebar.date_input("End Date", value=datetime.today().date(), help="Select end date for data")
    initial_cash = st.sidebar.number_input("Initial Cash", value=10000, min_value=1000, help="Set initial portfolio cash")
    commission = st.sidebar.number_input("Commission", value=0.001, min_value=0.0, step=0.0001, help="Set broker commission rate")
    model_type = st.sidebar.selectbox("Model Type", ["DQN", "GA"], help="Choose between Deep Q-Network or Genetic Algorithm")
    use_synthetic = st.sidebar.checkbox("Use Synthetic Data", value=False, help="Generate synthetic data instead of fetching real data")
    objective = st.sidebar.selectbox("Performance Objective", ["Maximize Sharpe Ratio", "Minimize Max Drawdown"], help="Select optimization goal")
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return
    if st.sidebar.button("Run Backtest"):
        with st.spinner("Validating inputs and fetching data..."):
            ticker_list = [t.strip() for t in tickers.split(",")]
            if not ticker_list or any(not t for t in ticker_list):
                st.error("Please enter valid tickers.")
                return
            data_feeds = []
            if use_synthetic:
                for ticker in ticker_list:
                    data = generate_synthetic_data(start_date, end_date, ticker=ticker)
                    if data.empty:
                        st.error(f"Failed to generate synthetic data for {ticker}.")
                        return
                    data_feed = bt.feeds.PandasData(dataname=data, fromdate=start_date, todate=end_date, name=ticker)
                    data_feeds.append(data_feed)
            else:
                try:
                    fetcher = DataFetcher()
                    for ticker in ticker_list:
                        data = fetcher.get_stock_data(symbol=ticker, start_date=start_date, end_date=end_date)
                        if data.empty:
                            st.error(f"No data retrieved for {ticker}. Please check ticker or date range.")
                            return
                        if len(data) < 30:
                            st.warning(f"Data for {ticker} has only {len(data)} rows. Consider extending date range.")
                        data_feed = bt.feeds.PandasData(dataname=data, fromdate=start_date, todate=end_date, name=ticker)
                        data_feeds.append(data_feed)
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}. Please check network or API access.")
                    return
        with st.spinner("Running backtest..."):
            strategy_class = {'DQN': DQNStrategy, 'GA': GAStrategy}[model_type]
            try:
                perf_summary, trade_log, fig, rules = run_backtest(
                    strategy_class=strategy_class,
                    data_feeds=data_feeds,
                    cash=initial_cash,
                    commission=commission
                )
                if not perf_summary:
                    st.error("Backtest failed to produce results.")
                    return
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}. Please check inputs or contact support.")
                return
        st.subheader("Performance Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sharpe Ratio", f"{perf_summary['Sharpe Ratio']:.2f}")
            st.metric("Total Return", f"{perf_summary['Total Return']*100:.2f}%")
            st.metric("Avg Daily Return", f"{perf_summary['Avg Daily Return']*100:.2f}%")
        with col2:
            st.metric("Avg Annual Return", f"{perf_summary['Avg Annual Return']*100:.2f}%")
            st.metric("Max Drawdown", f"{perf_summary['Max Drawdown']:.2f}%")
            st.metric("Total Trades", perf_summary['Total Trades'])
        st.subheader("Trading Rules")
        for ticker, rule in rules.items():
            with st.expander(f"Rules for {ticker}"):
                st.text(rule)
        st.subheader("Trade Log")
        if trade_log:
            trade_df = pd.DataFrame({"Trade": trade_log})
            st.dataframe(trade_df, use_container_width=True)
        else:
            st.info("No trades executed during the backtest.")
        st.subheader("Backtest Chart")
        st.pyplot(fig)
        st.subheader("Model Convergence")
        if model_type == "GA":
            fig_conv = plt.figure()
            for ticker in ticker_list:
                if ticker in strategy_class().fitness_history:
                    plt.plot(strategy_class().fitness_history[ticker], label=ticker)
            plt.xlabel("Generation")
            plt.ylabel("Fitness (Sum of Returns)")
            plt.title("GA Fitness Evolution")
            plt.legend()
            plt.grid(True)
            st.pyplot(fig_conv)
        st.subheader("Export Strategy")
        if st.button("Export Strategy"):
            with st.spinner("Exporting strategy..."):
                export_strategy(strategy_class, perf_summary, trade_log, '\n'.join([f"{t}: {rules[t]}" for t in rules]), ticker_list)
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    main()