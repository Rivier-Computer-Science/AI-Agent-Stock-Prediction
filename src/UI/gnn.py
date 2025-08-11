import os
import sys
from typing import List, Optional, Tuple, Dict, Any
import json
import time
from textwrap import dedent

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Optional OpenAI import for Crew-style agent integration (works with new or legacy SDKs)
try:
    from openai import OpenAI  # new SDK
    _HAS_OPENAI_NEW = True
except Exception:
    _HAS_OPENAI_NEW = False

try:
    import openai  # legacy SDK
    _HAS_OPENAI_LEGACY = True
except Exception:
    _HAS_OPENAI_LEGACY = False

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Get script directory to resolve relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------------------------------------------
# Utility: robust JSON extraction (handles stray text around it)
# -------------------------------------------------------------
def _extract_first_json_object(text: str) -> Dict[str, Any]:
    # Attempt to parse JSON directly; fallback to extracting first {...} block if needed.
    """
    Try to parse `text` as JSON; if that fails, extract the first {...} block.
    Raises ValueError if no valid JSON object can be found.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            for i in range(end, start, -1):
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    continue
    raise ValueError("No valid JSON object found in LLM output.")


# -------------------------------------------------------------
# Utility: LLM caller (JSON-mode if available) for DecisionAgent
# -------------------------------------------------------------
def _call_llm_json(prompt: str, model: Optional[str] = None) -> str:
    # Call OpenAI LLM preferring JSON-mode; gracefully fallback between SDKs.
    """
    Calls an LLM and returns the assistant text (JSON-only if supported).
    Prefers the new OpenAI SDK; falls back to legacy; raises on failure.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if _HAS_OPENAI_NEW:
        try:
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if not _HAS_OPENAI_LEGACY:
                raise RuntimeError(f"[GNN.6] LLM call failed (new SDK): {e}") from e

    if _HAS_OPENAI_LEGACY:
        try:
            if not getattr(openai, "api_key", None):
                openai.api_key = os.getenv("OPENAI_API_KEY")
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            raise RuntimeError(f"[GNN.6] LLM call failed (legacy SDK): {e}") from e

    raise RuntimeError("[GNN.6] OpenAI SDK not available. Install 'openai' and set OPENAI_API_KEY.")


# --------------------------------------------
# GNN.1 Price & Returns Ingestion (8 ph)
# --------------------------------------------
class PriceLoader:
    """
    Fetches OHLC data for a universe of tickers and computes daily returns.
    Persists price and returns matrices as NumPy and PyTorch tensors.
    """
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        data_dir: str = "data",
    ):
        # Initialize price loader with tickers, date range, and storage paths.
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = os.path.join(SCRIPT_DIR, data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

        self.ohlc_df: Optional[pd.DataFrame] = None
        self.returns_df: Optional[pd.DataFrame] = None

    def fetch_ohlc_data(self) -> pd.DataFrame:
        # Download and shape adjusted close prices into a clean DataFrame.
        raw = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            adj_close = raw["Adj Close"].copy()
        else:
            adj_close = raw[["Adj Close"]].rename(columns={"Adj Close": self.tickers[0]})
        adj_close = adj_close.reindex(columns=self.tickers)
        adj_close.ffill(inplace=True)
        adj_close.dropna(how="any", inplace=True)
        self.ohlc_df = adj_close
        return adj_close

    def compute_daily_returns(self) -> pd.DataFrame:
        # Compute daily percentage returns from adjusted close prices.
        if self.ohlc_df is None:
            raise RuntimeError("OHLC data not yet fetched. Call fetch_ohlc_data() first.")
        prices = self.ohlc_df
        returns = prices.pct_change().iloc[1:].copy()
        returns.dropna(how="any", inplace=True)
        self.returns_df = returns
        return returns

    def persist_tensors(self) -> Tuple[str, str, str, str]:
        # Persist prices/returns as NumPy arrays and PyTorch tensors.
        if self.ohlc_df is None or self.returns_df is None:
            raise RuntimeError(
                "DataFrames missing. Ensure fetch_ohlc_data() and compute_daily_returns() were called."
            )
        prices_np = self.ohlc_df.values.astype(np.float32)
        returns_np = self.returns_df.values.astype(np.float32)
        prices_pt = torch.from_numpy(prices_np)
        returns_pt = torch.from_numpy(returns_np)
        prices_np_path = os.path.join(self.data_dir, "prices.npy")
        prices_pt_path = os.path.join(self.data_dir, "prices.pt")
        returns_np_path = os.path.join(self.data_dir, "returns.npy")
        returns_pt_path = os.path.join(self.data_dir, "returns.pt")
        np.save(prices_np_path, prices_np)
        torch.save(prices_pt, prices_pt_path)
        np.save(returns_np_path, returns_np)
        torch.save(returns_pt, returns_pt_path)
        return prices_np_path, prices_pt_path, returns_np_path, returns_pt_path


# --------------------------------------------
# GNN.2 Graph Construction (9 ph)
# --------------------------------------------
class GraphBuilder:
    """
    Computes N-day rolling Pearson correlations on returns data, thresholds them,
    and saves adjacency matrices (NumPy and PyTorch) per date.
    """
    def __init__(
        self,
        returns_df: pd.DataFrame,
        window_size: int,
        threshold: float,
        output_dir: str = "adjacency_data",
    ):
        # Initialize rolling-correlation graph builder and ensure output directory.
        self.returns_df = returns_df.copy()
        self.window_size = window_size
        self.threshold = threshold
        self.output_dir = os.path.join(SCRIPT_DIR, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def build_graphs(self) -> None:
        # Build and store thresholded adjacency matrices for each rolling window end-date.
        dates = list(self.returns_df.index)
        for idx in range(self.window_size - 1, len(dates)):
            window_slice = self.returns_df.iloc[idx - self.window_size + 1 : idx + 1]
            corr_mat = window_slice.corr().values.astype(np.float32)
            adj_np = (np.abs(corr_mat) > self.threshold).astype(np.float32)
            np.fill_diagonal(adj_np, 0.0)
            adj_pt = torch.from_numpy(adj_np)
            date_str = dates[idx].strftime("%Y-%m-%d")
            np_path = os.path.join(self.output_dir, f"adjacency_{date_str}.npy")
            pt_path = os.path.join(self.output_dir, f"adjacency_{date_str}.pt")
            np.save(np_path, adj_np)
            torch.save(adj_pt, pt_path)


# --------------------------------------------
# GNN.3 Teacher GNN Development (9 ph)
# --------------------------------------------
class TeacherGNN(nn.Module):
    """
    Temporal GNN combining a GCN-like neighborhood aggregate (via adj @ x)
    with a GRU for node-level sequence modeling.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        gru_hidden_dim: int,
        output_dim: int,
    ):
        # Define a simple GCN→GRU→Classifier architecture for node-level prediction.
        super().__init__()
        self.gcn = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)
        self.classifier = nn.Linear(gru_hidden_dim, output_dim)

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Propagate via adjacency aggregation, run GRU over time, and classify last hidden state.
        """
        features: (seq_len, N)      – sequence of per-node features
        adj:      (N, N)            – adjacency matrix (binary or weighted)
        returns:  (N, output_dim)   – per-node logits
        """
        seq_len, N = features.shape
        x = features.unsqueeze(-1)  # (seq_len, N, 1)
        hidden_seq = []
        for t in range(seq_len):
            h = adj @ x[t]          # (N, 1) neighborhood aggregate
            h = self.gcn(h)         # (N, hidden_dim)
            hidden_seq.append(h)
        hidden_seq = torch.stack(hidden_seq, dim=0).permute(1, 0, 2)  # (N, seq_len, hidden_dim)
        out, _ = self.gru(hidden_seq)                                 # (N, seq_len, gru_hidden_dim)
        last = out[:, -1, :]                                          # (N, gru_hidden_dim)
        logits = self.classifier(last)                                # (N, output_dim)
        return logits


class TeacherTrain:
    """
    Training loop for Teacher GNN on adjacency and return sequences.
    """
    def __init__(
        self,
        returns_df: pd.DataFrame,
        adj_dir: str,
        delta: int,
        model_dir: str,
        lr: float = 1e-3,
        epochs: int = 10,
    ):
        # Configure teacher training, model/optim/loss, device, and storage.
        self.returns_df = returns_df.copy()
        self.adj_dir = os.path.join(SCRIPT_DIR, adj_dir)
        self.delta = delta
        self.model_dir = os.path.join(SCRIPT_DIR, model_dir)
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = TeacherGNN(input_dim=1, hidden_dim=32, gru_hidden_dim=64, output_dim=2).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        self.train_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.val_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def prepare_data(self) -> None:
        # Assemble (feature sequence, adjacency, label) tuples and create train/val split.
        dates = list(self.returns_df.index)
        feats, adjs, labels = [], [], []
        for idx in range(self.delta, len(dates) - 1):
            seq_start = idx - self.delta
            seq_end = idx + self.delta
            feat_seq = self.returns_df.iloc[seq_start : seq_end + 1].values.astype(np.float32)  # (2*delta+1, N)
            date_str = dates[idx].strftime("%Y-%m-%d")
            adj_path = os.path.join(self.adj_dir, f"adjacency_{date_str}.npy")
            if not os.path.exists(adj_path):
                continue
            adj_np = np.load(adj_path).astype(np.float32)  # (N, N)
            next_ret = self.returns_df.iloc[idx + 1].values
            label = (next_ret > 0).astype(int)             # (N,)
            feats.append(feat_seq)
            adjs.append(adj_np)
            labels.append(label)
        if not feats:
            raise RuntimeError("No training data; check adjacency files.")
        split = int(0.8 * len(feats))
        self.train_data = list(zip(feats[:split], adjs[:split], labels[:split]))
        self.val_data = list(zip(feats[split:], adjs[split:], labels[split:]))

    def train(self) -> None:
        # Train the teacher for multiple epochs and save the best validation checkpoint.
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for feat_np, adj_np, label_np in self.train_data:
                feat = torch.tensor(feat_np, device=self.device)
                adj = torch.tensor(adj_np, device=self.device)
                label = torch.tensor(label_np, device=self.device)
                logits = self.model(feat, adj)
                loss = self.criterion(logits, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)
            train_acc = correct / total

            self.model.eval()
            val_corr, val_tot = 0, 0
            with torch.no_grad():
                for feat_np, adj_np, label_np in self.val_data:
                    feat = torch.tensor(feat_np, device=self.device)
                    adj = torch.tensor(adj_np, device=self.device)
                    label = torch.tensor(label_np, device=self.device)
                    logits = self.model(feat, adj)
                    preds = logits.argmax(dim=1)
                    val_corr += (preds == label).sum().item()
                    val_tot += label.size(0)
            val_acc = val_corr / val_tot

            print(f"Epoch {epoch}/{self.epochs} - Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                ckpt = os.path.join(self.model_dir, "best_teacher.pth")
                torch.save(self.model.state_dict(), ckpt)
                print(f"Saved best teacher model to {ckpt}")


# --------------------------------------------
# GNN.4 Student GNN & Distillation (7 ph)
# --------------------------------------------
class StudentGNN(TeacherGNN):
    """
    Student model: same as teacher but no future inputs (uses only features up to t).
    """
    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Reuse TeacherGNN forward pass for the student model.
        return super().forward(features, adj)


class StudentTrain:
    """
    Train student with distillation loss (KL + classification).
    """
    def __init__(
        self,
        returns_df: pd.DataFrame,
        adj_dir: str,
        delta: int,
        teacher_ckpt: str,
        model_dir: str,
        lr: float = 1e-3,
        epochs: int = 10,
        alpha: float = 0.5,
        temperature: float = 2.0,
    ):
        # Configure student training with KD, load teacher, set losses/optim, and storage.
        self.returns_df = returns_df.copy()
        self.adj_dir = os.path.join(SCRIPT_DIR, adj_dir)
        self.delta = delta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = TeacherGNN(1, 32, 64, 2).to(self.device)
        self.teacher.load_state_dict(torch.load(teacher_ckpt, map_location=self.device))
        self.teacher.eval()
        self.model = StudentGNN(1, 32, 64, 2).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.alpha = alpha
        self.T = temperature
        self.epochs = epochs
        self.model_dir = os.path.join(SCRIPT_DIR, model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.train_data, self.val_data = [], []

    def prepare_data(self):
        # Build student sequences up to current time (no future), labels, and split.
        dates = list(self.returns_df.index)
        feats, adjs, labels = [], [], []
        for idx in range(self.delta, len(dates) - 1):
            seq_start = idx - self.delta
            seq_end = idx + 1  # student sees only up to current (end exclusive)
            feat_seq = self.returns_df.iloc[seq_start : seq_end].values.astype(np.float32)  # (delta, N)
            date_str = dates[idx].strftime("%Y-%m-%d")
            adj_path = os.path.join(self.adj_dir, f"adjacency_{date_str}.npy")
            if not os.path.exists(adj_path):
                continue
            adj_np = np.load(adj_path).astype(np.float32)
            next_ret = self.returns_df.iloc[idx + 1].values
            label = (next_ret > 0).astype(int)
            feats.append(feat_seq)
            adjs.append(adj_np)
            labels.append(label)
        split = int(0.8 * len(feats))
        self.train_data = list(zip(feats[:split], adjs[:split], labels[:split]))
        self.val_data = list(zip(feats[split:], adjs[split:], labels[split:]))

    def train(self):
        # Train the student with KD (KL + CE), evaluate, and save the best checkpoint.
        best_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            self.model.train(); total_loss = 0.0; correct = 0; total = 0
            for feat_np, adj_np, label_np in self.train_data:
                feat = torch.tensor(feat_np, device=self.device)
                adj = torch.tensor(adj_np, device=self.device)
                label = torch.tensor(label_np, device=self.device)
                s_logits = self.model(feat, adj)
                with torch.no_grad():
                    t_logits = self.teacher(feat, adj)
                loss_ce = self.ce(s_logits, label)
                p_s = F.log_softmax(s_logits / self.T, dim=1)
                p_t = F.softmax(t_logits / self.T, dim=1)
                loss_kl = self.kl(p_s, p_t) * (self.T * self.T)
                loss = self.alpha * loss_kl + (1 - self.alpha) * loss_ce
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                total_loss += loss.item()
                preds = s_logits.argmax(dim=1)
                correct += (preds == label).sum().item(); total += label.size(0)
            train_acc = correct / total

            self.model.eval(); v_corr = 0; v_tot = 0
            with torch.no_grad():
                for feat_np, adj_np, label_np in self.val_data:
                    feat = torch.tensor(feat_np, device=self.device)
                    adj = torch.tensor(adj_np, device=self.device)
                    label = torch.tensor(label_np, device=self.device)
                    logits = self.model(feat, adj)
                    preds = logits.argmax(dim=1)
                    v_corr += (preds == label).sum().item(); v_tot += label.size(0)
            val_acc = v_corr / v_tot

            print(f"Epoch {epoch}/{self.epochs} - Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                path = os.path.join(self.model_dir, "best_student.pth")
                torch.save(self.model.state_dict(), path)
                print(f"Saved best student model to {path}")


# --------------------------------------------
# GNN.5 Inference Wrapper (6 ph)
# --------------------------------------------
class Inference:
    """
    Loads a distilled student model and runs inference on today's data.
    (Note: This is not a Crew AI agent; 'Agent' naming is reserved for Crew AI.)
    """
    def __init__(self, model_path: str, tickers: List[str], delta: int):
        # Load trained student model and cached returns for runtime inference.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StudentGNN(1, 32, 64, 2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tickers = tickers
        self.delta = delta
        # load returns
        self.returns = np.load(os.path.join(SCRIPT_DIR, "data", "returns.npy"))

    def run(self) -> List[Dict[str, Any]]:
        # Produce latest per-ticker upward probabilities using most recent adjacency.
        # prepare last sequence
        seq = self.returns[-self.delta:, :]

        # pick only .npy adjacency files
        adj_dir = os.path.join(SCRIPT_DIR, "adjacency_data")
        npy_files = [f for f in os.listdir(adj_dir)
                     if f.startswith("adjacency_") and f.endswith(".npy")]
        if not npy_files:
            raise RuntimeError(f"No .npy files found in {adj_dir}")
        npy_files.sort()
        latest = npy_files[-1]
        adj_np = np.load(os.path.join(adj_dir, latest)).astype(np.float32)

        feat = torch.tensor(seq, device=self.device)
        adj_t = torch.tensor(adj_np, device=self.device)
        logits = self.model(feat, adj_t)
        probs = F.softmax(logits, dim=1)[:, 1]  # probability up
        return [{"stock": ticker, "student_prob": float(p)}
                for ticker, p in zip(self.tickers, probs.tolist())]


# =========================================================
# GNN.6 Crew AI Integration (4 ph)  #716
# ---------------------------------------------------------
# GNN.6.1 DecisionAgent:
#   - Reads {student_prob} and returns {"recommendation":"BUY"/"SELL"/"HOLD"}.
#   - The agent decides autonomously (no thresholds passed from code).
#   - JSON-only output enforced; robust parsing with fallback.
#
# GNN.6.2 Test chaining Inference → DecisionAgent end-to-end.
# =========================================================
class DecisionAgent:
    """
    Crew-style Decision Agent.
    - Accepts 'student_prob' (likelihood of upward move).
    - Autonomously decides BUY/SELL/HOLD (no thresholds provided by code).
    - Calls an LLM to produce a JSON-only response for integration.
    - If the LLM fails, we use a conservative local fallback rule.
    """
    def __init__(self, model: Optional[str] = None):
        # Prepare Crew-style decision agent and choose LLM model for calls.
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _build_prompt(self, student_prob: float) -> str:
        # Construct instruction for autonomous decision with JSON-only response.
        """
            Instruct the agent to choose a recommendation without explicit thresholds.
            The guidance is qualitative to keep the decision autonomous.
        """
        return dedent(f"""
            You are a trading decision microservice.
            Input: an object with a single numeric field 'student_prob' in [0,1],
            representing the probability that a stock's price goes UP in the next period.

            Decide autonomously (do NOT reveal your internal thresholds):
              - Return {{"recommendation":"BUY"}} if the probability indicates a clear upside edge.
              - Return {{"recommendation":"SELL"}} if the probability indicates a clear downside edge.
              - Otherwise return {{"recommendation":"HOLD"}} when uncertainty is high or the edge is weak.

            Respond with only a JSON object and nothing else.
            Input: {{"student_prob": {student_prob:.6f}}}
        """).strip()

    def _local_fallback(self, p: float) -> str:
        # Provide a conservative local decision if LLM path fails or returns invalid JSON.
        """
        Conservative deterministic fallback used only if the LLM path fails.
        (Kept internal to maintain 'agent decides on its own' contract externally.)
        """
        if p >= 0.65:
            return "BUY"
        if p <= 0.35:
            return "SELL"
        return "HOLD"

    def get_decision(self, student_prob: float) -> Dict[str, str]:
        # Query the LLM, parse JSON, and return {'recommendation': ...} with fallback on error.
        """
        Query LLM for the decision; robustly parse JSON.
        Falls back to local rule if anything goes wrong.
        """
        prompt = self._build_prompt(student_prob)
        try:
            raw = _call_llm_json(prompt, model=self.model)
            obj = _extract_first_json_object(raw)
            rec = str(obj.get("recommendation", "")).upper().strip()
            if rec not in {"BUY", "SELL", "HOLD"}:
                raise ValueError("Invalid 'recommendation' value.")
            print(f"[GNN.6.1] DecisionAgent JSON -> {obj}")
            return {"recommendation": rec}
        except Exception as e:
            rec = self._local_fallback(student_prob)
            print(f"[GNN.6.1] DecisionAgent fell back to local rule ({e}); rec={rec}")
            return {"recommendation": rec}


def chain_inference_to_decision(
    preds: List[Dict[str, Any]],
    decision_agent: DecisionAgent
) -> List[Dict[str, Any]]:
    # Chain raw predictions into decisions by calling DecisionAgent for each item.
    """
    GNN.6.2: End-to-end chaining function.
    Takes predictions from Inference and augments each item with a Crew decision.
    """
    enriched: List[Dict[str, Any]] = []
    for item in preds:
        p = float(item["student_prob"])
        decision = decision_agent.get_decision(p)
        enriched.append({**item, **decision})
    print("[GNN.6.2] Chaining complete.")
    return enriched


# --------------------------------------------
# Main Script
# --------------------------------------------
if __name__ == "__main__":
    # 1) Price Loading
    universe = input("Enter tickers separated by commas: ").upper().split(",")
    universe = [t.strip() for t in universe if t.strip()]
    loader = PriceLoader(universe, "2020-01-01", "2023-12-31", data_dir="data")
    prices = loader.fetch_ohlc_data()
    returns = loader.compute_daily_returns()
    loader.persist_tensors()

    # 2) Graphs
    window = int(input("Enter rolling window size: "))
    thresh = float(input("Enter correlation threshold: "))
    gb = GraphBuilder(returns, window, thresh, output_dir="adjacency_data")
    gb.build_graphs()

    # 3) Teacher Train
    delta = int(input("Enter teacher delta: "))
    epochs_t = int(input("Enter teacher epochs: "))
    lr_t = float(input("Enter teacher learning rate: "))
    tch = TeacherTrain(returns, "adjacency_data",
                       delta,
                       model_dir="teacher_model",
                       lr=lr_t,
                       epochs=epochs_t)
    tch.prepare_data()
    tch.train()

    # 4) Student Train & Distillation
    epochs_s = int(input("Enter student epochs: "))
    tch_ckpt = os.path.join(SCRIPT_DIR, "teacher_model", "best_teacher.pth")
    stu = StudentTrain(returns,
                       "adjacency_data",
                       delta,
                       teacher_ckpt=tch_ckpt,
                       model_dir="student_model",
                       lr=lr_t,
                       epochs=epochs_s)
    stu.prepare_data()
    stu.train()

    # 5) Inference (not a Crew AI agent)
    infer = Inference(os.path.join(SCRIPT_DIR, "student_model", "best_student.pth"),
                      universe,
                      delta)
    t0 = time.time()
    raw_result = infer.run()
    duration = time.time() - t0

    # 6) Crew AI DecisionAgent & chaining (end-to-end test)
    #    - Agent decides autonomously; no thresholds passed from code.
    decision_agent = DecisionAgent(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    final_result = chain_inference_to_decision(raw_result, decision_agent)

    # Pretty-print final JSON including recommendations
    print(json.dumps({"inference_time": duration, "predictions": final_result}, indent=2))
