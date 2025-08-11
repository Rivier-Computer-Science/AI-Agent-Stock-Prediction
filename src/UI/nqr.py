# ------------------------------------------------------------------------------------
# READ ME (comments only; NO code changes):
# - This script demonstrates an end-to-end toy pipeline:
#     1) Fetch quarterly financials via yfinance.
#     2) Compute a few simple ratios per quarter.
#     3) Train quick models (FNN + RandomForest) on synthetic data
#     4) Build a feature vector from the most recent quarter in your date window.
#     5) Run inference with the trained models.
#     6) Ask the CrewAIDecisionAgent (LLM) to apply a threshold/majority rule and return JSON.
#
# ------------------------------------------------------------------------------------

######################################
# Config / verbosity
######################################
VERBOSE = False  # set True to see debug logs for ratios, etc.

def _vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

######################################
# NQR.1 Ratio Data Retrieval & Preparation
######################################
import yfinance as yf
import sys
import os
import numpy as np
import pandas as pd
import joblib
import time
import pickle
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

# OpenAI (required by your CrewAIDecisionAgent implementation)
import openai  # NOTE: This assumes the legacy SDK (< 1.0). See the header notes above.

# Optional ANFIS import (if available)
try:
    from anfis import ANFISNet
except ImportError:
    ANFISNet = None


# ----------------- Small utilities -----------------
def _to_float_or_zero(x):
    """Robustly coerce value to finite float or 0.0."""
    # Try float(); if NaN/inf or raises, return 0.0.
    try:
        val = float(x)
        if np.isfinite(val):
            return val
        return 0.0
    except Exception:
        return 0.0


def _parse_date(s: str) -> datetime:
    """Parse 'YYYY-MM-DD' into datetime (raises on invalid)."""
    # This ensures the user’s date input is validated early.
    return datetime.strptime(s.strip(), "%Y-%m-%d")


# ----------------- Data fetching -----------------
class DataFetch:
    """
    Fetch quarterly financial statements for a given ticker via yfinance.
    NOTE: yfinance quarterly statements don't accept date ranges; we fetch
    recent quarters and filter later by user-provided dates.
    """
    def __init__(self, num_quarters: int = 16):
        self.num_quarters = num_quarters  # Increase if you need a longer lookback window.

    def fetch_income_statement(self, ticker: str) -> list:
        tk = yf.Ticker(ticker)
        # quarterly_financials returns a wide DF; transpose for row-per-quarter.
        inc_df = tk.quarterly_financials.transpose().sort_index(ascending=False)
        top_inc = inc_df.head(self.num_quarters)
        records = []
        for idx, row in top_inc.iterrows():
            records.append({
                "fiscalDateEnding": idx.strftime("%Y-%m-%d"),
                "totalRevenue": row.get("Total Revenue", None),  # May be missing for some tickers
                "netIncome": row.get("Net Income", None),        # May be missing for some tickers
            })
        if not records:
            # If you hit this, the ticker may not expose quarterly data via yfinance.
            raise ValueError(f"No income statement data found for {ticker}")
        return records

    def fetch_balance_sheet(self, ticker: str) -> list:
        tk = yf.Ticker(ticker)
        bs_df = tk.quarterly_balance_sheet.transpose().sort_index(ascending=False)
        top_bs = bs_df.head(self.num_quarters)
        records = []
        for idx, row in top_bs.iterrows():
            records.append({
                "fiscalDateEnding": idx.strftime("%Y-%m-%d"),
                "totalShareholderEquity": row.get("Total Stockholder Equity", None),  # Could be None
                "totalLiabilities": row.get("Total Liab", None),                      # Could be None
            })
        if not records:
            raise ValueError(f"No balance sheet data found for {ticker}")
        return records


# ----------------- Ratio calculation -----------------
class RatioCalc:
    """Compute financial ratios (ROE, Debt/Equity, Net Profit Margin)."""
    def compute_ratios(self, income_reports: list, balance_reports: list) -> list:
        # This zips income & balance by order (most recent to older). If the two
        # statements are not aligned perfectly in dates, you may want to match by
        # fiscalDateEnding explicitly (merge by date) in a future improvement.
        ratios = []
        for inc, bal in zip(income_reports, balance_reports):
            net_inc = _to_float_or_zero(inc.get('netIncome', 0))
            rev = _to_float_or_zero(inc.get('totalRevenue', 0))
            equity = _to_float_or_zero(bal.get('totalShareholderEquity', 0))
            liabilities = _to_float_or_zero(bal.get('totalLiabilities', 0))

            # Guard against zero divisors; zero is used when line items are missing.
            roe = (net_inc / equity) if equity else 0.0
            debt_equity = (liabilities / equity) if equity else 0.0
            net_profit_margin = (net_inc / rev) if rev else 0.0

            ratios.append({
                'fiscalDateEnding': inc.get('fiscalDateEnding'),
                'roe': float(roe),
                'debt_equity': float(debt_equity),
                'net_profit_margin': float(net_profit_margin),
            })
        return ratios

    def filter_by_date(self, ratios: list, start_date: str, end_date: str) -> list:
        # Keep only rows whose fiscalDateEnding falls in [start_date, end_date].
        # Sorted newest-first for easy "latest" indexing.
        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date)
        out = []
        for r in ratios:
            try:
                dt = _parse_date(str(r.get('fiscalDateEnding', '1900-01-01')))
            except Exception:
                # If a date is malformed/missing, we skip it.
                continue
            if start_dt <= dt <= end_dt:
                out.append(r)
        out.sort(key=lambda x: x['fiscalDateEnding'], reverse=True)
        return out


######################################
# NQR.2 Model Training Pipeline
######################################
class ModelTrain:
    """
    Train Feed-Forward NN, Random Forest, and ANFIS models.
    (Demo uses synthetic labels since we don't have ground-truth targets here.)
    """
    class FeedForwardNN(nn.Module):
        def __init__(self, input_dim, hidden_dim=16):
            super(ModelTrain.FeedForwardNN, self).__init__()
            # A tiny 2-layer network with Sigmoid for binary classification demo.
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.network(x)

    def train_fnn(self, X, y, epochs=20, lr=0.001):
        # Simple train/validation split for demo purposes.
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        model = ModelTrain.FeedForwardNN(X.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        # Quick validation accuracy on binarized outputs.
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t)
            preds_cls = (preds.numpy() >= 0.5).astype(int)
            val_acc = accuracy_score(y_val, preds_cls)

        return model, val_acc

    def train_rf(self, X, y):
        # Small grid just to show hyperparameter tuning in the example.
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, None]}
        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_rf = grid.best_estimator_  # correct single underscore (common typo)
        preds = best_rf.predict(X_val)
        val_acc = accuracy_score(y_val, preds)
        return best_rf, val_acc

    def train_anfis(self, X, y):
        # Optional path; many environments won't have ANFIS installed.
        if ANFISNet is None:
            raise ImportError("ANFISNet library not installed.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        anfis = ANFISNet(input_dim=X.shape[1])
        anfis.fit(X_train, y_train)
        preds = anfis.predict(X_val)
        preds_cls = (preds >= 0.5).astype(int)
        val_acc = accuracy_score(y_val, preds_cls)
        return anfis, val_acc

    def compare_models(self, acc_fnn, acc_rf, acc_anfis):
        # Just a convenience function to summarize which validation accuracy is largest.
        best = max([(acc_fnn, 'FNN'), (acc_rf, 'RF'), ((acc_anfis or 0), 'ANFIS')])[1]
        return {'fnn_acc': acc_fnn, 'rf_acc': acc_rf, 'anfis_acc': acc_anfis, 'best_model': best}


######################################
# NQR.3 Inference API Development
######################################
class ModelInferAgent:
    """
    Load trained models and perform inference, outputting JSON.
    Uses a state_dict checkpoint for FNN to be compatible with PyTorch 2.6 weights_only safety.
    """
    def __init__(self, fnn_path, rf_path, anfis_path=None):
        # Load FNN (state_dict) with weights_only=True (PyTorch 2.6 default)
        # We saved {"state_dict": ..., "meta": {"input_dim": 3, "hidden_dim": 16}}
        # so we can reconstruct the module and load weights safely.
        ckpt = torch.load(fnn_path, map_location='cpu', weights_only=True)
        meta = ckpt.get('meta', {})
        input_dim = int(meta.get('input_dim', 3))
        hidden_dim = int(meta.get('hidden_dim', 16))

        self.fnn = ModelTrain.FeedForwardNN(input_dim=input_dim, hidden_dim=hidden_dim)
        self.fnn.load_state_dict(ckpt['state_dict'])
        self.fnn.eval()

        # Load RF from joblib
        self.rf = joblib.load(rf_path)

        # Optional ANFIS via pickle (only if you trained & saved it)
        self.anfis = None
        if anfis_path:
            try:
                with open(anfis_path, 'rb') as f:
                    self.anfis = pickle.load(f)
            except Exception:
                # If loading fails, we proceed without ANFIS rather than crashing.
                self.anfis = None  # tolerate missing/invalid ANFIS

    def infer(self, features: list) -> dict:
        # Features expected shape: (3,) matching [ROE, Debt/Equity, Net Profit Margin]
        start = time.time()
        x = np.array(features, dtype=float).reshape(1, -1)
        with torch.no_grad():
            fnn_prob = float(self.fnn(torch.tensor(x, dtype=torch.float32)).item())
        rf_prob = float(self.rf.predict_proba(x)[0, 1])
        anfis_prob = float(self.anfis.predict(x)[0]) if self.anfis else None
        latency = time.time() - start
        if latency > 1.0:
            print(f"[WARN] Inference latency {latency:.3f}s exceeds 1s")
        return {"fnn_prob": fnn_prob, "rf_prob": rf_prob, "anfis_prob": anfis_prob}


######################################
# NQR.4 Crew AI Decision Agent (YOUR VERSION)
######################################
class CrewAIDecisionAgent:
    """
    NQR.4 Crew AI Decision Agent using GPT-4o for majority-vote or threshold rules.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o",
        threshold: float = 0.5,
        role: str = "financial decision-maker",
        goal: str = "Apply majority-vote or threshold rules to decide trade action.",
        backstory: str = "You are a Crew AI agent specialized in combining multiple model probabilities into a single buy/hold/sell decision."
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.role = role
        self.goal = goal
        self.backstory = backstory

    def decide(self, fnn_prob: float, rf_prob: float, anfis_prob: float = None) -> dict:
        """
        NQR.4.1 & NQR.4.2:
        - Build system + user messages embedding role, goal, backstory.
        - Instruct GPT-4o to count buy/sell signals via threshold and apply majority vote.
        - Return JSON { decision, explanation }.
        """
        system_msg = {
            "role": "system",
            "content": (
                f"Backstory: {self.backstory}\n"
                f"Role: {self.role}\n"
                f"Goal: {self.goal}"
            )
        }

        user_prompt = (
            f"You are given the following model probabilities:\n"
            f"- FNN: {fnn_prob}\n"
            f"- RF: {rf_prob}\n"
            f"- ANFIS: {anfis_prob if anfis_prob is not None else 'N/A'}\n\n"
            f"Use threshold = {self.threshold}.\n"
            f"Count each probability > threshold as a BUY signal, < threshold as a SELL signal.\n"
            f"Apply majority vote:\n"
            f"- If 2 or more BUY signals → decision = BUY\n"
            f"- If 2 or more SELL signals → decision = SELL\n"
            f"- Otherwise → decision = HOLD\n\n"
            f"Respond in strict JSON format:\n"
            f'{{"decision": "<BUY|SELL|HOLD>", "explanation": "your reasoning here"}}'
        )
        user_msg = {"role": "user", "content": user_prompt}

        # NOTE:
        # - This uses the legacy SDK method. Ensure your `openai` package is < 1.0.
        # - Ensure OPENAI_API_KEY is set in your environment.
        # - If the model name is not accessible to your key, this will error at runtime.
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[system_msg, user_msg],
            temperature=0.0
        )
        content = response.choices[0].message.content

        try:
            # Strictly try to parse as JSON first; if the model returns plain text,
            # we fall back to putting it in the "decision" field verbatim.
            result = json.loads(content)
        except json.JSONDecodeError:
            result = {"decision": content.strip(), "explanation": ""}

        return result


######################################
# MAIN
######################################
def main():
    """
    Interactive CLI:
      1) Ask user for ticker, start date, end date (YYYY-MM-DD).
      2) Fetch quarterly financials; compute ratios; filter by date.
      3) Build a 3-feature vector from the MOST RECENT record in range:
         [ROE, Debt/Equity, Net Profit Margin].
      4) Train demo models (FNN + RF) on synthetic data just to complete the pipeline.
      5) Run inference and aggregate probabilities.
      6) Ask Crew AI agent (OpenAI) for final decision with explanation.
    """
    # --- 1) Prompt user input
    try:
        ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
    except EOFError:
        # Happens if stdin is closed or you run non-interactively without piping input.
        print("[ERROR] No input provided for ticker.")
        sys.exit(1)

    if not ticker:
        print("[ERROR] Empty ticker.")
        sys.exit(1)

    try:
        start_date = input("Enter START date (YYYY-MM-DD): ").strip()
        end_date = input("Enter END date   (YYYY-MM-DD): ").strip()
        _ = _parse_date(start_date)
        _ = _parse_date(end_date)
    except EOFError:
        print("[ERROR] No input provided for dates.")
        sys.exit(1)
    except Exception as e:
        # If the date is in the wrong format, we bail out early.
        print(f"[ERROR] Invalid date format: {e}")
        sys.exit(1)

    print("\n[STEP] Fetching financial statements...")
    df = DataFetch(num_quarters=16)
    try:
        income_reports = df.fetch_income_statement(ticker)
        balance_reports = df.fetch_balance_sheet(ticker)
    except Exception as e:
        # Network errors, invalid ticker symbols, or missing data can land here.
        print(f"[ERROR] Failed to fetch financial data for {ticker}: {e}")
        sys.exit(1)

    # --- 2) Compute & filter ratios
    rc = RatioCalc()
    ratios_all = rc.compute_ratios(income_reports, balance_reports)
    ratios = rc.filter_by_date(ratios_all, start_date, end_date)

    # If nothing falls in range, silently fall back to most recent (no debug prints)
    if not ratios:
        # This fallback helps ensure the pipeline runs even when the date window is empty.
        ratios = sorted(ratios_all, key=lambda r: r['fiscalDateEnding'], reverse=True)
        _vprint("[WARN] No quarterly statements within your date range; using most recent available data instead.")

    # Optional debug output (hidden by default)
    _vprint(f"[INFO] Found {len(ratios)} quarterly ratio rows (most recent first).")
    for r in (ratios[:5] if VERBOSE else []):
        _vprint("   ", r)

    # --- 3) Build features from the MOST RECENT record
    # If `ratios` is empty here (shouldn’t be), indexing [0] would crash; the fallback above avoids that.
    latest = ratios[0]
    feature_vector = [
        _to_float_or_zero(latest.get('roe', 0.0)),
        _to_float_or_zero(latest.get('debt_equity', 0.0)),
        _to_float_or_zero(latest.get('net_profit_margin', 0.0)),
    ]
    print(f"\n[STEP] Feature vector (latest quarter in range): {feature_vector}  "
          f"(date={latest.get('fiscalDateEnding')})")

    # --- 4) Train models on synthetic data (3 features)
    print("\n[STEP] Training models (FNN + RF) on synthetic data...")
    rng = np.random.default_rng(42)
    X_demo = rng.normal(size=(200, 3)).astype(np.float32)
    y_demo = (rng.random(size=(200,)) > 0.5).astype(int)

    trainer = ModelTrain()
    fnn_model, fnn_acc = trainer.train_fnn(X_demo, y_demo, epochs=15, lr=0.003)
    rf_model, rf_acc = trainer.train_rf(X_demo, y_demo)

    # Optional ANFIS
    anfis_model, anfis_acc = None, None
    if ANFISNet:
        try:
            anfis_model, anfis_acc = trainer.train_anfis(X_demo, y_demo)
        except Exception as e:
            # ANFIS libs vary; if training fails, we quietly proceed without it.
            _vprint(f"[WARN] ANFIS training failed/disabled: {e}")
            anfis_model, anfis_acc = None, None

    comp = trainer.compare_models(fnn_acc, rf_acc, anfis_acc)
    print(f"[INFO] Validation accuracies → {comp}")

    # --- Save models (FNN via state_dict; RF via joblib; ANFIS via pickle if present)
    # We save a minimal checkpoint to be compatible with PyTorch 2.6 `weights_only=True`.
    ckpt = {
        "state_dict": fnn_model.state_dict(),
        "meta": {"input_dim": 3, "hidden_dim": 16}
    }
    torch.save(ckpt, 'fnn_model.pt')  # PyTorch 2.6-safe
    joblib.dump(rf_model, 'rf_model.pkl')
    if anfis_model is not None:
        with open('anfis_model.pkl', 'wb') as f:
            pickle.dump(anfis_model, f)
        anfis_path = 'anfis_model.pkl'
    else:
        anfis_path = None

    # --- 5) Run inference
    print("\n[STEP] Running inference on feature vector...")
    infer_agent = ModelInferAgent('fnn_model.pt', 'rf_model.pkl', anfis_path)
    result = infer_agent.infer(feature_vector)
    print(f"[INFO] Model probabilities → {result}")

    # --- 6) Ask Crew AI decision agent (OpenAI) using your class
    # NOTE: Ensure your OPENAI_API_KEY is set and you have access to the specified model.
    print("\n[STEP] Getting final decision from Crew AI agent...")
    crew_decider = CrewAIDecisionAgent(threshold=0.5)
    decision = crew_decider.decide(
        fnn_prob=result["fnn_prob"],
        rf_prob=result["rf_prob"],
        anfis_prob=result.get("anfis_prob", None)
    )

    print("\n===== FINAL DECISION =====")
    print(f"Decision: {decision['decision']}")
    # If you want to show the explanation, uncomment the next line.
    # (Left commented per your instruction not to print extra lines.)
    #print(f"Explanation: {decision['explanation']}")
    print("==========================\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C.
        print("\n[INFO] Aborted by user.")
    except Exception as e:
        # Any unexpected exceptions get caught here to avoid ugly tracebacks in CLI usage.
        print(f"[ERROR] Unhandled exception: {e}")
        sys.exit(1)
