"""
integration_benchmark.py

Separate script implementing:

XGB.7.1 Integration test: full pipeline from data fetch → model inference → Crew AI DecisionAgent.
XGB.7.2 Benchmark end-to-end latency and report per-step timings, with a check against the 5s target.

Usage (example):
    python integration_benchmark.py --ticker AAPL --start 2025-03-01 --end 2025-04-01

Requires that the following files exist in working directory:
    - best_model.pt          (pretrained CNN-Attention-LSTM)
    - xgb_model.joblib       (trained XGBoost model)
"""

import argparse
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBRegressor
import joblib
from langchain_openai import ChatOpenAI
from crewai import Agent  # assuming crewai is installed

# -------------------------------
# Reuse essential model definitions (duplicated/minimal for standalone use)
# -------------------------------

class CNNAttentionLSTM(nn.Module):
    def __init__(self, num_features: int,
                 cnn_channels: int = 32,
                 lstm_hidden: int = 64,
                 attn_heads: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(embed_dim=cnn_channels,
                                          num_heads=attn_heads,
                                          batch_first=True)
        self.lstm = nn.LSTM(input_size=cnn_channels,
                            hidden_size=lstm_hidden,
                            batch_first=True)
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        out, (h, _) = self.lstm(x)
        emb = h[-1]
        return self.head(emb), emb

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DecisionAgent using Crew AI LLM (no heuristic fallback here per requirement)
class DecisionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        # Agent object kept for integration metadata
        self.agent = Agent(
            llm=self.llm,
            role='Decision Agent',
            goal='Given xgb_prob, decide whether to BUY, SELL, or HOLD using thresholds and output JSON only.',
            backstory='Maps probability to discrete recommendation with thresholds.',
            verbose=False,
            tools=[]
        )

    def create_prompt(self, inference_json: str) -> str:
        return f"""You are given a JSON string with a field "xgb_prob".
Apply the following logic:
  - If xgb_prob > 0.6 output {{"recommendation":"BUY"}}
  - If xgb_prob < 0.4 output {{"recommendation":"SELL"}}
  - Otherwise output {{"recommendation":"HOLD"}}
Return only the JSON object with the correct recommendation.
Input: {inference_json}"""

    def get_decision(self, inference_json: str) -> dict:
        prompt = self.create_prompt(inference_json)
        # Directly call the underlying LLM interface
        if hasattr(self.llm, "predict"):
            raw = self.llm.predict(prompt)
        else:
            raw = self.llm(prompt)  # fallback if interface differs
        try:
            decision = json.loads(raw.strip())
        except Exception as e:
            raise RuntimeError(f"DecisionAgent returned invalid JSON: {raw}") from e
        return decision

# -------------------------------
# Pipeline utility functions
# -------------------------------

def fetch_ohlcv(tickers, start, end, interval='1d'):
    df = yf.download(tickers, start=start, end=end,
                     interval=interval, group_by='ticker', auto_adjust=False)
    df_list = []
    for t in tickers:
        df_t = df[t].copy()
        df_t['Ticker'] = t
        df_list.append(df_t)
    out = pd.concat(df_list)
    out.reset_index(inplace=True)
    out.set_index(['Ticker', 'Date'], inplace=True)
    return out

def preprocess_data(df, method='ffill'):
    if method in ['ffill', 'bfill']:
        df = df.groupby(level=0).apply(lambda x: x.fillna(method=method))
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError("method must be 'ffill','bfill',or 'drop'")
    return df.dropna()

def normalize_features(df, features):
    df_norm = df.copy()
    from sklearn.preprocessing import StandardScaler
    for t, group in df_norm.groupby(level=0):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(group[features])
        df_norm.loc[group.index, features] = scaled
    return df_norm

def slice_windows(df, features, window_size, stride=1):
    X, y = [], []
    for t in df.index.get_level_values(0).unique():
        df_t = df.loc[t]
        vals = df_t[features].values
        closes = df_t['Close'].values
        for i in range(0, len(df_t) - window_size, stride):
            X.append(vals[i:i+window_size])
            ret = (closes[i+window_size] - closes[i+window_size-1]) / closes[i+window_size-1]
            y.append(ret)
    return np.array(X), np.array(y)

def train_val_test_split(X, y, val_ratio=0.2, test_ratio=0.1, shuffle=False):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    t_size = int(n * test_ratio)
    v_size = int(n * val_ratio)
    return {
        'train': (X[idx[:-t_size-v_size]], y[idx[:-t_size-v_size]]),
        'val':   (X[idx[-t_size-v_size:-t_size]], y[idx[-t_size-v_size:-t_size]]),
        'test':  (X[idx[-t_size:]], y[idx[-t_size:]])
    }

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------------
# End-to-end execution + integration test / benchmark
# -------------------------------

def load_models(cnn_lstm_path='best_model.pt', xgb_path='xgb_model.joblib', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load deep model
    cnn_lstm = CNNAttentionLSTM(num_features=5)
    cnn_lstm.load_state_dict(torch.load(cnn_lstm_path, map_location=device))
    cnn_lstm.to(device)
    cnn_lstm.eval()
    # load xgboost
    xgb_model = joblib.load(xgb_path)
    return cnn_lstm, xgb_model, device

def inference_pipeline_once(ticker, start, end, cnn_lstm, xgb_model, decision_agent, device):
    """
    Runs full pipeline for latest window and returns structured output plus per-stage timings.
    """
    timings = {}
    t0 = time.perf_counter()

    # 1. Fetch
    df = fetch_ohlcv([ticker], start, end)
    timings['fetch'] = time.perf_counter() - t0

    # 2. Preprocess + normalize + windowing
    t1 = time.perf_counter()
    df = preprocess_data(df)
    df = normalize_features(df, ['Open', 'High', 'Low', 'Close', 'Volume'])
    X, y = slice_windows(df, ['Open', 'High', 'Low', 'Close', 'Volume'], window_size=20)
    if len(X) == 0:
        raise RuntimeError("Not enough data to form a window.")
    latest_window = X[-1]
    latest_date = df.index.get_level_values('Date')[-1].strftime('%Y-%m-%d')
    timings['preprocess_window'] = time.perf_counter() - t1

    # 3. Model inference (cnn_lstm + xgb)
    t2 = time.perf_counter()
    with torch.no_grad():
        x_tensor = torch.from_numpy(latest_window).float().unsqueeze(0).to(device)
        cnn_score, emb = cnn_lstm(x_tensor)
        emb_np = emb.cpu().numpy()
        xgb_raw = xgb_model.predict(emb_np)
        xgb_prob = float(sigmoid(xgb_raw[0]))
        cnn_lstm_score = cnn_score.squeeze(0).cpu().item()
    timings['model_inference'] = time.perf_counter() - t2

    # 4. DecisionAgent
    t3 = time.perf_counter()
    inference_json = json.dumps({
        "date": latest_date,
        "cnn_lstm_score": cnn_lstm_score,
        "xgb_prob": xgb_prob
    })
    decision = decision_agent.get_decision(inference_json)
    timings['decision_agent'] = time.perf_counter() - t3

    total = time.perf_counter() - t0
    timings['total'] = total

    output = {
        "inference": {
            "date": latest_date,
            "cnn_lstm_score": cnn_lstm_score,
            "xgb_prob": xgb_prob
        },
        "decision": decision
    }
    return output, timings

# Integration test for validity
def test_integration_decision_validity(ticker, start, end, cnn_lstm_path='best_model.pt', xgb_path='xgb_model.joblib'):
    cnn_lstm, xgb_model, device = load_models(cnn_lstm_path, xgb_path)
    decision_agent = DecisionAgent()
    output, timings = inference_pipeline_once(ticker, start, end, cnn_lstm, xgb_model, decision_agent, device)
    assert 'decision' in output, "No decision key in integration output"
    recommendation = output['decision'].get('recommendation')
    assert recommendation in {"BUY", "SELL", "HOLD"}, f"Invalid recommendation: {recommendation}"
    print("[XGB.7.1] Integration test passed. Decision:", recommendation)
    print("[XGB.7.1] Timings:", {k: f"{v:.3f}s" for k,v in timings.items()})
    return output, timings

# Benchmark test for latency
def test_latency_threshold(ticker, start, end, runs=3, threshold=5.0, cnn_lstm_path='best_model.pt', xgb_path='xgb_model.joblib'):
    cnn_lstm, xgb_model, device = load_models(cnn_lstm_path, xgb_path)
    decision_agent = DecisionAgent()

    all_totals = []
    all_stage_breakdowns = []
    # warm-up
    _ , _ = inference_pipeline_once(ticker, start, end, cnn_lstm, xgb_model, decision_agent, device)
    for i in range(runs):
        _, timings = inference_pipeline_once(ticker, start, end, cnn_lstm, xgb_model, decision_agent, device)
        all_totals.append(timings['total'])
        all_stage_breakdowns.append(timings)
        print(f"[XGB.7.2] Run {i+1} total latency: {timings['total']:.3f}s")

    median = np.median(all_totals)
    mean = np.mean(all_totals)
    print(f"[XGB.7.2] Latency summary over {runs} runs: mean={mean:.3f}s, median={median:.3f}s")
    if median > threshold:
        print(f"[XGB.7.2] WARNING: median latency {median:.3f}s exceeds target {threshold}s.")
    else:
        print(f"[XGB.7.2] SUCCESS: median latency {median:.3f}s within target.")

    # Detailed breakdown of the best run (lowest total)
    best_idx = int(np.argmin(all_totals))
    best_timings = all_stage_breakdowns[best_idx]
    print("[XGB.7.2] Breakdown of best run:", {k: f"{v:.3f}s" for k,v in best_timings.items()})
    return all_totals, best_timings

# -------------------------------
# CLI entrypoint
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integration test and benchmark for XGB.7")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, e.g., AAPL")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--threshold", type=float, default=5.0, help="Latency threshold in seconds")
    args = parser.parse_args()

    try:
        # Run integration validity test
        test_integration_decision_validity(args.ticker, args.start, args.end)

        # Run latency benchmark
        test_latency_threshold(args.ticker, args.start, args.end, runs=args.runs, threshold=args.threshold)
    except AssertionError as ae:
        print("Integration test assertion failed:", ae)
    except Exception as e:
        print("Error during integration/benchmark:", e)
