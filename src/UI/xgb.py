import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import os
import json
from textwrap import dedent

# PyTorch imports for CNN-Attention-LSTM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# XGBoost & Scikit-Learn imports for fine-tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# Crew AI / LLM imports (reference-style)
from crewai import Agent, Task
from langchain_openai import ChatOpenAI

# === Utility to call LLM flexibly ===
def call_llm(llm, prompt: str) -> str:
    """
    Invoke the LLM interface; handles common naming conventions.
    Returns the raw text output.
    """
    if hasattr(llm, "predict"):
        return llm.predict(prompt)
    if callable(llm):
        return llm(prompt)
    raise AttributeError("LLM object has no callable interface for prediction.")

# === XGB.1 Data Ingestion & Windowing ===
def fetch_ohlcv(tickers: List[str], start: str, end: str,
                interval: str = '1d') -> pd.DataFrame:
    """
    XGB.1.1: Download OHLCV data for given tickers from yfinance.
    Returns a DataFrame indexed by [Ticker, Date].
    """
    data = yf.download(tickers, start=start, end=end,
                       interval=interval, group_by='ticker',
                       auto_adjust=False)
    df_list = []
    for t in tickers:
        df_t = data[t].copy()
        df_t['Ticker'] = t
        df_list.append(df_t)
    df = pd.concat(df_list)
    df.reset_index(inplace=True)
    df.set_index(['Ticker', 'Date'], inplace=True)
    print(f"[XGB.1.1] Fetched OHLCV for {tickers}, shape: {df.shape}")
    return df

def preprocess_data(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    XGB.1.2: Impute or drop missing data.
    Supports forward-fill, backward-fill, or dropping.
    """
    if method in ['ffill', 'bfill']:
        df = df.groupby(level=0).apply(lambda x: x.fillna(method=method))
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError("method must be 'ffill', 'bfill', or 'drop'")
    df = df.dropna()
    print(f"[XGB.1.2] Preprocessed data with method={method}, remaining NaNs: {df.isna().sum().sum()}")
    return df

def normalize_features(df: pd.DataFrame,
                       features: List[str]) -> pd.DataFrame:
    """
    XGB.1.3: Z-score normalize the specified features per ticker.
    """
    df_norm = df.copy()
    for t, group in df_norm.groupby(level=0):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(group[features])
        df_norm.loc[group.index, features] = scaled
    print(f"[XGB.1.3] Normalized features: {features}")
    return df_norm

def slice_windows(df: pd.DataFrame, features: List[str],
                  window_size: int, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    XGB.1.4: Slice time series into overlapping windows and compute next-day return label.
    """
    X, y = [], []
    for t in df.index.get_level_values(0).unique():
        df_t = df.loc[t]
        vals = df_t[features].values
        closes = df_t['Close'].values
        for i in range(0, len(df_t) - window_size, stride):
            X.append(vals[i:i+window_size])
            ret = (closes[i+window_size] - closes[i+window_size-1]) / closes[i+window_size-1]
            y.append(ret)
    X_arr, y_arr = np.array(X), np.array(y)
    print(f"[XGB.1.4] Sliced into windows: X shape {X_arr.shape}, y shape {y_arr.shape}")
    return X_arr, y_arr

def train_val_test_split(
    X: np.ndarray, y: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    shuffle: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split into train/val/test without leakage. Optionally shuffle.
    """
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    t_size = int(n * test_ratio)
    v_size = int(n * val_ratio)
    splits = {
        'train': (X[idx[:-t_size-v_size]], y[idx[:-t_size-v_size]]),
        'val':   (X[idx[-t_size-v_size:-t_size]], y[idx[-t_size-v_size:-t_size]]),
        'test':  (X[idx[-t_size:]], y[idx[-t_size:]])
    }
    print(f"[XGB.1.4] Split sizes -> train: {splits['train'][0].shape[0]}, val: {splits['val'][0].shape[0]}, test: {splits['test'][0].shape[0]}")
    return splits

class WindowDataset(Dataset):
    """
    Wrapper to provide torch Dataset interface for windowed data.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === XGB.2 CNN-Attention-LSTM Pretraining ===
class CNNAttentionLSTM(nn.Module):
    """
    XGB.2.1 / XGB.2.2 / XGB.2.3: 1D CNN encoder + self-attention + LSTM decoder with regression head.
    """
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
        x = x.transpose(1, 2)                  # [batch, features, time]
        x = self.cnn(x)                        # CNN feature extraction
        x = x.transpose(1, 2)                  # Prepare for attention
        x, _ = self.attn(x, x, x)              # Self-attention over time
        out, (h, _) = self.lstm(x)             # LSTM decoding
        emb = h[-1]                            # Final hidden state embedding
        return self.head(emb), emb             # Regression output and embedding

def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 50,
    patience: int = 5,
    ckpt_path: str = 'best_model.pt'
) -> nn.Module:
    """
    XGB.2.4 / XGB.2.5: Training loop with MSE loss, checkpointing, early stopping, and metric logging.
    """
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    wait = 0
    model.to(device)

    for epoch in range(1, epochs+1):
        model.train()
        tr_losses = []
        for Xb, yb in dataloaders['train']:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in dataloaders['val']:
                Xb, yb = Xb.to(device), yb.to(device)
                pred, _ = model(Xb)
                val_losses.append(criterion(pred, yb).item())
        tr_avg, val_avg = np.mean(tr_losses), np.mean(val_losses)
        print(f"[XGB.2.4] Epoch {epoch:02d} | Train {tr_avg:.4f} | Val {val_avg:.4f}")
        if val_avg < best_val:
            best_val = val_avg
            torch.save(model.state_dict(), ckpt_path)
            print(f"[XGB.2.5] Saved new best model at epoch {epoch} with val loss {val_avg:.4f}")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("[XGB.2.4] Early stopping.")
                break

    model.load_state_dict(torch.load(ckpt_path))
    print("[XGB.2] Pretraining complete. Best model loaded.")
    return model

# === XGB.3 Embedding Extraction ===
def freeze_backbone(model: nn.Module):
    """
    XGB.3.1: Freeze CNN+Attention+LSTM backbone so that gradients aren't tracked.
    """
    for p in model.cnn.parameters():  p.requires_grad = False
    for p in model.attn.parameters(): p.requires_grad = False
    for p in model.lstm.parameters(): p.requires_grad = False
    print("[XGB.3.1] Backbone frozen.")

def extract_and_save_embeddings(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    output_file: str = 'embeddings_labels.csv'
):
    """
    XGB.3.2 / XGB.3.3: Forward all windows, extract embeddings, and persist with labels.
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    embs, labs = [], []
    model.to(device).eval()
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            _, e = model(Xb)
            embs.append(e.cpu().numpy())
            labs.append(yb.numpy())
    embs = np.vstack(embs)
    labs = np.concatenate(labs)
    cols = [f"emb_{i}" for i in range(embs.shape[1])]
    df_out = pd.DataFrame(embs, columns=cols)
    df_out['label'] = labs
    df_out.to_csv(output_file, index=False)
    print(f"[XGB.3.2/3.3] Saved embeddings and labels to {output_file}")

# === XGB.4 XGBoost Fine-tuning ===
# XGB.4.1 Load embeddings + labels into a pandas DataFrame (#652)
def load_embeddings_labels(path: str = 'embeddings_labels.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[XGB.4.1] Loaded embeddings DataFrame shape: {df.shape}")
    return df

# XGB.4.2 Perform train/validation split (#653)
def split_embeddings(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df.drop('label', axis=1).values
    y = df['label'].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=random_state)
    print(f"[XGB.4.2] Split embeddings -> train: {X_train.shape[0]}, val: {X_val.shape[0]}")
    return X_train, X_val, y_train, y_val

# XGB.4.3 Grid-search n_estimators, max_depth, learning_rate (#654)
def tune_xgboost(
    X_train: np.ndarray, y_train: np.ndarray
) -> GridSearchCV:
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb = XGBRegressor(objective='reg:squarederror', verbosity=0)
    grid = GridSearchCV(
        xgb, param_grid, cv=5,
        scoring='neg_mean_squared_error',
        verbose=2, n_jobs=-1,
        error_score='raise'
    )
    grid.fit(X_train, y_train)
    print(f"[XGB.4.3] Grid search complete. Best params: {grid.best_params_}")
    return grid

# XGB.4.4 Train best estimator; evaluate and save the model (#655)
def train_and_save_xgb(
    grid: GridSearchCV,
    X_val: np.ndarray, y_val: np.ndarray,
    output_path: str = 'xgb_model.joblib'
):
    best = grid.best_estimator_
    preds = best.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    print(f"[XGB.4.4] Validation MSE: {mse:.4f}, R2: {r2:.4f}")
    joblib.dump(best, output_path)
    print(f"[XGB.4.4] Saved XGBoost model to {output_path}")

# === XGB.5 Inference Wrapper ===
# XGB.5.1 Load both deep model and XGBoost model (#656)
def load_models(cnn_lstm_path: str = 'best_model.pt', xgb_path: str = 'xgb_model.joblib', device=None):
    """
    Load pretrained CNN-Attention-LSTM and XGBoost models for inference.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_lstm = CNNAttentionLSTM(num_features=5)
    cnn_lstm.load_state_dict(torch.load(cnn_lstm_path, map_location=device))
    cnn_lstm.to(device)
    cnn_lstm.eval()
    xgb_model = joblib.load(xgb_path)
    print(f"[XGB.5.1] Loaded CNN-LSTM and XGBoost models.")
    return cnn_lstm, xgb_model, device

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# XGB.5.2 Given latest window compute scores (#657)
def infer_window(
    cnn_lstm: nn.Module,
    xgb_model,
    window: np.ndarray,
    scaler: StandardScaler = None,
    device=None
) -> Dict:
    """
    Compute cnn_lstm_score and xgb_prob for a single window.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if scaler is not None:
        window = scaler.transform(window)
    x_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)
    with torch.no_grad():
        cnn_score, _ = cnn_lstm(x_tensor)
        cnn_score_val = cnn_score.squeeze(0).cpu().item()
        _, emb = cnn_lstm(x_tensor)
        emb_np = emb.cpu().numpy()
        xgb_raw = xgb_model.predict(emb_np)
        xgb_prob = float(sigmoid(xgb_raw[0]))
    print("[XGB.5.2] Computed cnn_lstm_score and xgb_prob.")
    return {
        "cnn_lstm_score": cnn_score_val,
        "xgb_prob": xgb_prob
    }

# XGB.5.3 Emit JSON (#658)
def make_inference_json(date_str: str, scores: Dict) -> str:
    """
    Package the inference results into JSON.
    """
    out = {
        "date": date_str,
        "cnn_lstm_score": scores["cnn_lstm_score"],
        "xgb_prob": scores["xgb_prob"]
    }
    json_str = json.dumps(out)
    print(f"[XGB.5.3] Emitted inference JSON: {json_str}")
    return json_str

# === XGB.6 Crew AI Agent Development ===
# Shared LLM model for agents
gpt_model = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o"
)

# XGB.6.1 DecisionAgent (#659)
class DecisionAgent:
    def __init__(self):
        # Agent object kept for integration metadata; LLM used directly for inference
        self.agent = Agent(
            llm=gpt_model,
            role='Decision Agent',
            goal='Given xgb_prob, decide whether to BUY, SELL, or HOLD using thresholds and output JSON only.',
            backstory='This agent maps probability to trading recommendation based on fixed thresholds.',
            verbose=False,
            tools=[]
        )
        self.llm = gpt_model

    def create_prompt(self, inference_json: str) -> str:
        """
        Build prompt to feed into the LLM for decision logic.
        """
        return dedent(f"""
            You are given a JSON string with a field "xgb_prob".
            Apply the following logic:
              - If xgb_prob > 0.6 output {{"recommendation":"BUY"}}
              - If xgb_prob < 0.4 output {{"recommendation":"SELL"}}
              - Otherwise output {{"recommendation":"HOLD"}}
            Return only the JSON object with the correct recommendation.
            Input: {inference_json}
        """).strip()

    def get_decision(self, inference_json: str) -> Dict:
        """
        Query LLM to get the decision; parse JSON output.
        """
        prompt = self.create_prompt(inference_json)
        raw = call_llm(self.llm, prompt)
        try:
            recommendation = json.loads(raw.strip())
        except Exception as e:
            print("[XGB.6.1] Failed to parse decision agent output:", raw, "error:", e)
            raise RuntimeError("DecisionAgent returned invalid JSON") from e
        print(f"[XGB.6.1] DecisionAgent output: {recommendation}")
        return recommendation

# XGB.6.2 ExplainAgent (#660)
class ExplainAgent:
    def __init__(self):
        self.agent = Agent(
            llm=gpt_model,
            role='Explain Agent',
            goal='Given the inference JSON and feature importances, provide a concise rationale for the recommendation.',
            backstory='This agent explains why the model made the recommendation by referencing top feature importances.',
            verbose=False,
            tools=[]
        )
        self.llm = gpt_model

    def create_prompt(self, inference_json: str, feature_importances: np.ndarray, feature_names: List[str]) -> str:
        """
        Build prompt describing top feature importances and asking for a rationale.
        """
        idxs = np.argsort(np.abs(feature_importances))[::-1][:3]
        top_features = [(feature_names[i], float(feature_importances[i])) for i in idxs]
        feats_desc = ", ".join([f"{name} ({imp:.3f})" for name, imp in top_features])
        prompt = dedent(f"""
            You are given:
              1. Inference JSON: {inference_json}
              2. Feature importances from the XGBoost model.
            The top drivers are: {feats_desc}.
            Based on these and the xgb_prob in the JSON, give a brief explanation (1-2 sentences) of why the recommendation was made.
            Keep it concise and mention the most influential features.
        """).strip()
        return prompt

    def get_explanation(self, inference_json: str, feature_importances: np.ndarray, feature_names: List[str]) -> str:
        """
        Query LLM to get the rationale explanation.
        """
        prompt = self.create_prompt(inference_json, feature_importances, feature_names)
        raw = call_llm(self.llm, prompt)
        explanation = raw.strip()
        print(f"[XGB.6.2] ExplainAgent output: {explanation}")
        return explanation

# XGB.6.3 End-to-end test chaining InferenceAgent → DecisionAgent → ExplainAgent (#661)
def end_to_end_test(
    ticker: str,
    start: str,
    end: str,
    cnn_lstm_path: str = 'best_model.pt',
    xgb_path: str = 'xgb_model.joblib'
):
    print("[XGB.6.3] Starting end-to-end test.")
    # 1. Prepare data & latest window
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    win_size = 20
    df = fetch_ohlcv([ticker], start, end)
    df = preprocess_data(df)
    df = normalize_features(df, features)
    X, y = slice_windows(df, features, win_size)
    if len(X) == 0:
        raise RuntimeError("Not enough data to form a window.")
    latest_window = X[-1]
    latest_date = df.index.get_level_values('Date')[-1].strftime('%Y-%m-%d')

    # 2. Load models
    cnn_lstm, xgb_model, device = load_models(cnn_lstm_path, xgb_path)

    # 3. InferenceAgent: compute scores
    scores = infer_window(cnn_lstm, xgb_model, latest_window, scaler=None, device=device)
    inference_json = make_inference_json(latest_date, scores)

    # 4. DecisionAgent via Crew AI LLM
    decision_agent = DecisionAgent()
    recommendation = decision_agent.get_decision(inference_json)

    # 5. ExplainAgent via Crew AI LLM
    feat_importances = getattr(xgb_model, "feature_importances_", None)
    if feat_importances is None:
        feat_importances = np.zeros(1)
    feature_names = [f"emb_{i}" for i in range(len(feat_importances))]
    explain_agent = ExplainAgent()
    explanation = explain_agent.get_explanation(inference_json, feat_importances, feature_names)

    # 6. Consolidate and output
    output = {
        "inference": json.loads(inference_json),
        "decision": recommendation,
        "explanation": explanation
    }
    print("[XGB.6.3] End-to-end output:", json.dumps(output, indent=2))
    return output

# === Main Script ===
if __name__ == '__main__':
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    start = input("Start date (YYYY-MM-DD): ").strip()
    end   = input("End date   (YYYY-MM-DD): ").strip()
    try:
        pd.to_datetime(start); pd.to_datetime(end)
    except:
        print("Bad date format."); exit(1)

    features = ['Open','High','Low','Close','Volume']
    win_size = 20

    # XGB.1 pipeline
    df = fetch_ohlcv([ticker], start, end)
    df = preprocess_data(df)
    df = normalize_features(df, features)
    X, y = slice_windows(df, features, win_size)
    splits = train_val_test_split(X, y, val_ratio=0.2,
                                  test_ratio=0.1, shuffle=True)

    batch = 64
    dl = {k: DataLoader(WindowDataset(*splits[k]),
                        batch_size=batch,
                        shuffle=(k=='train'))
          for k in splits}
    print("[Main] DataLoaders prepared.")

    # XGB.2 pretraining
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNAttentionLSTM(num_features=len(features))
    model = train_model(model, dl, dev)

    # XGB.3 embedding extraction
    freeze_backbone(model)
    X_all = np.vstack([splits[k][0] for k in splits])
    y_all = np.concatenate([splits[k][1] for k in splits])
    full_ds = WindowDataset(X_all, y_all)
    extract_and_save_embeddings(model, full_ds, dev)

    # XGB.4 fine-tuning
    df_emb = load_embeddings_labels('embeddings_labels.csv')
    X_train, X_val, y_train, y_val = split_embeddings(df_emb)
    grid = tune_xgboost(X_train, y_train)
    train_and_save_xgb(grid, X_val, y_val)

    # XGB.5/XGB.6 example end-to-end run
    try:
        end_to_end_test(ticker, start, end,
                        cnn_lstm_path='best_model.pt',
                        xgb_path='xgb_model.joblib')
    except Exception as e:
        print("End-to-end test failed:", str(e))
