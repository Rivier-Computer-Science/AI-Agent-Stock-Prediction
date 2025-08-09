import os
import re
import sys
import json
import math
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings

# --- CrewAI / LLM ---
# CrewAI orchestrates LLM-backed agents & tasks; LangChain's ChatOpenAI is the LLM client.
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# TRF end-to-end overview
# -----------------------------------------------------------------------------
# TRF.1 DataHandler:
#   - Downloads OHLC data using yfinance
#   - Cleans/imputes and normalizes Close prices
#   - Builds sliding windows and labels
#
# TRF.2 TransformerModel:
#   - Small Transformer encoder producing 2-class logits for "down"/"up"
#
# TRF.3 EmbeddingsExtractor:
#   - Uses the trained Transformer to emit logits per window and saves them to CSV
#
# TRF.4 XGBoostTrainer:
#   - Trains an XGBoost classifier on the Transformer logits via GridSearchCV
#
# TRF.6 CrewAIDecisionAgent:
#   - Calls a CrewAI LLM agent to map xgb_prob -> BUY/SELL/HOLD (no fallback)
#
# TRF.5 InferencePipeline:
#   - Loads Transformer weights + XGBoost model
#   - Computes transformer_score and xgb_prob for a latest window
#   - Invokes CrewAI agent for the recommendation
#   - Prints the result and force-exits the process immediately (as requested)
# -----------------------------------------------------------------------------


# #########################
# 1. TRF.1: DataHandler
# #########################
class DataHandler:
    """
    Handles data ingestion and preprocessing for OHLCV data.
    """
    def __init__(self, ticker, start_date, end_date, window_size=20):
        # Parameters for the asset universe and temporal range
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        # Number of past timesteps in each sliding window
        self.window_size = window_size
        # Standard z-score scaler for the Close column
        self.scaler = StandardScaler()

    def fetch_ohlcv(self):
        """Fetch Close prices for the given ticker and date range."""
        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        # Keep only Close to keep model simple and avoid feature leakage
        return df[['Close']].dropna()

    def impute_and_align(self, df):
        """Fill missing data and ensure datetime index."""
        df = df.ffill().bfill()  # forward/backward fill any small gaps
        df.index = pd.to_datetime(df.index)
        return df

    def normalize(self, df):
        """Apply z-score normalization to the Close price."""
        values = df.values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)
        return pd.DataFrame(scaled, index=df.index, columns=df.columns)

    def create_windows(self, df, test_size=0.2, val_size=0.1):
        """
        Build windows and labels (1 if next-day price up, else 0),
        then split into train/val/test without shuffling to preserve temporal order.
        """
        X, y = [], []
        # Iterate over all possible windows: [i, i+window_size)
        for i in range(len(df) - self.window_size - 1):
            window = df.iloc[i : i + self.window_size].values
            # Label compares next day's Close vs current last Close in the window
            label = 1 if df.iloc[i + self.window_size + 1, 0] > df.iloc[i + self.window_size, 0] else 0
            X.append(window)
            y.append(label)
        X = np.stack(X)
        y = np.array(y)

        # Temporal split: train vs (val+test), then split (val,test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + val_size, shuffle=False
        )
        val_rel = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_rel, shuffle=False
        )
        return X_train, X_val, X_test, y_train, y_val, y_test


# #########################
# 2. TRF.2: TransformerModel
# #########################
class TransformerModel(nn.Module):
    """
    Transformer encoder + classifier for windowed data.
    """
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # Project input features (here: 1 feature -> d_model)
        self.input_proj = nn.Linear(feature_dim, d_model)
        # Single encoder layer configuration
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        # Stack of identical encoder layers
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        # Final classifier to 2 classes (down/up)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: [batch, window, feature]
        x = self.input_proj(x)            # [batch, window, d_model]
        x = x.permute(1, 0, 2)            # Transformer expects [seq_len, batch, d_model]
        out = self.encoder(x)             # [window, batch, d_model]
        pooled = out.mean(dim=0)          # Mean-pool across the temporal dimension
        logits = self.classifier(pooled)  # [batch, 2]
        return logits

    def train_model(self, train_loader, val_loader=None,
                    epochs=20, lr=1e-4, patience=3, ckpt_path='transformer.pt'):
        """Training loop with optional early stopping and checkpoint."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        best_val = float('inf')
        wait = 0

        for epoch in range(1, epochs + 1):
            # Explicitly use Module.train() to enable dropout, etc.
            super().train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}, train loss: {total_loss/len(train_loader):.4f}")

            if val_loader:
                # Eval mode disables dropout, uses running stats for batchnorm
                super().eval()
                val_loss = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        val_loss += loss_fn(self(xb), yb).item()
                val_loss /= len(val_loader)
                print(f"Epoch {epoch}, val loss: {val_loss:.4f}")

                # Early stopping on best validation loss
                if val_loss < best_val:
                    best_val = val_loss
                    wait = 0
                    torch.save(self.state_dict(), ckpt_path)
                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping.")
                        break

        # Load best checkpointed weights for downstream usage
        self.load_state_dict(torch.load(ckpt_path))


# #########################
# 3. TRF.3: EmbeddingsExtractor
# #########################
class EmbeddingsExtractor:
    """
    Extracts embeddings from a trained TransformerModel.
    """
    def __init__(self, model):
        # Keep the model in eval mode for deterministic outputs
        self.model = model.eval()

    def extract_embeddings(self, loader, output_csv='embeddings.csv'):
        """Generate embeddings (logits) and save with labels."""
        rows = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                emb = self.model(xb).cpu().numpy()  # logits for each sample in the batch
                for vec, label in zip(emb, yb.numpy()):
                    rows.append(list(vec) + [int(label)])
        if not rows:
            # Helpful fail-fast if upstream windowing emitted nothing
            raise RuntimeError("No embeddings produced. Check your data/windowing.")
        cols = [f'emb_{i}' for i in range(len(rows[0]) - 1)] + ['label']
        pd.DataFrame(rows, columns=cols).to_csv(output_csv, index=False)
        print(f"Embeddings saved to {output_csv}")

    def validate_embeddings(self, csv_path, emb_dim):
        """Check CSV has emb_dim cols + label."""
        df = pd.read_csv(csv_path)
        assert df.shape[1] == emb_dim + 1, f"Expected {emb_dim+1} cols, got {df.shape[1]}"
        assert set(df['label'].unique()).issubset({0, 1}), "Labels must be 0 or 1"
        print("Embedding file validated.")


# #########################
# 4. TRF.4: XGBoostTrainer
# #########################
class XGBoostTrainer:
    """
    Loads embeddings, grid-searches hyperparameters, trains and saves XGBoost model.
    """
    def __init__(self, emb_csv='embeddings.csv'):
        self.emb_csv = emb_csv

    def train(self, output_model='xgb_model.joblib'):
        # Read the Transformer logits + labels
        df = pd.read_csv(self.emb_csv)
        X = df.drop('label', axis=1).values
        y = df['label'].values

        # Temporal split respected (shuffle=False earlier), GridSearch then CV on train
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Small hyperparameter grid for speed; expand if needed
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        grid = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', verbose=1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        joblib.dump(best_model, output_model)
        print(f"Saved XGBoost model to {output_model}")
        print(f"Best hyperparameters: {grid.best_params_}")
        return best_model


# #########################
# 6. TRF.6: CrewAI Decision Agent (LLM-only, no fallback)
# #########################
class CrewAIDecisionAgent:
    """
    Uses a CrewAI Agent (LLM-backed) to decide BUY/SELL/HOLD based on xgb_prob.
    No rule-based fallback: if the LLM isn't available or doesn't return valid JSON, this raises.
    """
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0, verbose: bool = False):
        # Require OpenAI credentials; the user explicitly asked for no fallback
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. This implementation requires an LLM via CrewAI (no fallback)."
            )
        # Construct the LLM and an Agent with strict JSON-only output behavior
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.agent = Agent(
            role="Trading Decision Agent",
            goal="Return a clear BUY, SELL, or HOLD recommendation for a stock based on xgb_prob.",
            backstory=(
                "You specialize in converting probabilistic model outputs into actionable trade decisions. "
                "You always answer strictly in JSON."
            ),
            verbose=verbose,
            allow_delegation=False,
            llm=self.llm,
        )

    def _task_for(self, stock: str, prob: float) -> Task:
        # Validate input probability; ensure the LLM sees a clean numeric value
        if prob is None or isinstance(prob, float) and (math.isnan(prob) or math.isinf(prob)):
            raise ValueError("xgb_prob must be a finite float.")

        # Prompt instructs the LLM to emit exactly one JSON object with the key "recommendation"
        prompt = (
            "You are a trading decision assistant.\n"
            f"Ticker: {stock}\n"
            f"xgb_prob (probability of an upward move): {prob:.6f}\n\n"
            "Instructions:\n"
            "1) Decide one of: BUY, SELL, HOLD.\n"
            "2) Respond with EXACTLY a compact JSON object on one line, no extra text, like:\n"
            '{"recommendation":"BUY"}\n'
            "3) Valid values are only BUY, SELL, HOLD (uppercase)."
        )
        return Task(
            description=prompt,
            expected_output='A single-line JSON object with key "recommendation" and value BUY/SELL/HOLD.',
            agent=self.agent,
        )

    def _extract_json(self, text: str) -> dict:
        # Be defensive: some LLM chains may add traces; pull the first {...} block
        m = re.search(r'\{.*\}', text.strip(), flags=re.DOTALL)
        if not m:
            raise RuntimeError("CrewAI agent did not return JSON.")
        try:
            return json.loads(m.group(0))
        except Exception as e:
            raise RuntimeError(f"Invalid JSON from CrewAI agent: {e}")

    def decide(self, items):
        """
        items: dict or list of dicts with keys:
          { "stock": <str>, "xgb_prob": <float> }
        returns: list of {stock, xgb_prob, recommendation}
        """
        if isinstance(items, dict):
            items = [items]
        outputs = []
        for it in items:
            stock = it.get("stock")
            prob = it.get("xgb_prob")
            if stock is None:
                raise ValueError("Missing 'stock' in decision item.")
            # Build a single Task, run a single-agent Crew sequentially
            task = self._task_for(stock, float(prob))
            crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential)
            res = crew.kickoff()
            text = str(res)
            # Parse LLM output as strict JSON with top-level key "recommendation"
            obj = self._extract_json(text)
            rec = obj.get("recommendation", "").upper()
            if rec not in {"BUY", "SELL", "HOLD"}:
                raise RuntimeError("CrewAI agent returned invalid recommendation.")
            outputs.append({"stock": stock, "xgb_prob": float(prob), "recommendation": rec})
        return outputs


# #########################
# 5. TRF.5: InferencePipeline
# #########################
class InferencePipeline:
    """
    Loads Transformer and XGBoost models, computes scores for a new window, asks CrewAI agent for recommendation,
    and emits JSON including the recommendation. The program terminates immediately after printing.
    """
    def __init__(self, transformer_path='transformer.pt', xgb_path='xgb_model.joblib',
                 window_size=20, model_name: str = "gpt-4o", temperature: float = 0.0):
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer = None
        self.transformer_path = transformer_path
        # Load the trained XGBoost model (fitted on Transformer logits)
        self.xgb = joblib.load(xgb_path)
        self.window_size = window_size
        # Keep a local scaler if needed (not used in inference path directly)
        self.scaler = StandardScaler()
        # CrewAI decision agent (no fallback)
        self.decision_agent = CrewAIDecisionAgent(model_name=model_name, temperature=temperature)

    def load_transformer(self, feature_dim):
        # Instantiate the Transformer and load its trained weights
        model = TransformerModel(feature_dim)
        model.load_state_dict(torch.load(self.transformer_path, map_location=self.device))
        self.transformer = model.to(self.device).eval()

    def predict(self, ticker, start_date, end_date):
        # Prepare normalized data for the requested period
        handler = DataHandler(ticker, start_date, end_date, self.window_size)
        df = handler.fetch_ohlcv()
        df = handler.impute_and_align(df)
        df_norm = handler.normalize(df)

        # Build latest window for inference
        window = df_norm.values[-self.window_size:]
        feature_dim = window.shape[1]
        if self.transformer is None:
            self.load_transformer(feature_dim)

        # Torch forward pass to get Transformer logits and softmax probability
        xb = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.transformer(xb).cpu().numpy()
            probs = F.softmax(torch.tensor(logits), dim=1).numpy()[0]

        # transformer_score: P(up) from Transformer
        transformer_score = float(probs[1])
        # emb: logits are the features for the XGBoost meta-classifier
        emb = logits
        # xgb_prob: meta probability of "up" based on Transformer logits
        xgb_prob = float(self.xgb.predict_proba(emb)[0][1])

        # Ask CrewAI agent for the decision (no fallback; will raise on error)
        rec_obj = self.decision_agent.decide({"stock": ticker, "xgb_prob": xgb_prob})[0]
        recommendation = rec_obj["recommendation"]

        # Build a compact, machine-readable record
        result = {
            'date': end_date,
            'ticker': ticker,
            'transformer_score': transformer_score,
            'xgb_prob': xgb_prob,
            'recommendation': recommendation
        }

        # Print the CrewAI decision for humans and JSON for machines, then exit immediately
        print(f"CrewAI Decision: {recommendation}")
        print(json.dumps(result))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # force-stop the process immediately after printing (prevents lingering telemetry threads)


# #########################
# Main execution for TRF.1 - TRF.6
# #########################
if __name__ == '__main__':
    # Simple CLI inputs for interactive runs
    ticker = input("Enter stock ticker (e.g. AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date   = input("Enter end date (YYYY-MM-DD): ")

    # 1) Data ingestion & windowing
    handler = DataHandler(ticker, start_date, end_date)
    df = handler.fetch_ohlcv()
    df = handler.impute_and_align(df)
    df_norm = handler.normalize(df)
    X_train, X_val, X_test, y_train, y_val, y_test = handler.create_windows(df_norm)

    # Wrap arrays into PyTorch DataLoaders
    train_ld = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ), batch_size=32, shuffle=True)
    val_ld = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    ), batch_size=32)

    # 2) Transformer training (with early stopping + checkpointing)
    feature_dim = X_train.shape[2]
    transformer = TransformerModel(feature_dim)
    transformer.train_model(train_ld, val_ld)

    # 3) Embedding extraction from Transformer and basic CSV validation
    extractor = EmbeddingsExtractor(transformer)
    extractor.extract_embeddings(train_ld, output_csv='embeddings.csv')
    extractor.validate_embeddings('embeddings.csv', emb_dim=2)

    # 4) Train XGBoost meta-classifier on Transformer logits
    xgb_trainer = XGBoostTrainer('embeddings.csv')
    xgb_trainer.train(output_model='xgb_model.joblib')

    # 5 & 6) Inference + CrewAI decision (program exits inside predict)
    infer = InferencePipeline(
        'transformer.pt',
        'xgb_model.joblib',
        window_size=handler.window_size,
        model_name="gpt-4o",
        temperature=0.0
    )
    infer.predict(ticker, start_date, end_date)
