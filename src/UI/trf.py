import os
import json
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
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# #########################
# 1. TRF.1: DataHandler
# #########################
class DataHandler:
    """
    Handles data ingestion and preprocessing for OHLCV data.
    """
    def __init__(self, ticker, start_date, end_date, window_size=20):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.scaler = StandardScaler()

    def fetch_ohlcv(self):
        """Fetch Close prices for the given ticker and date range."""
        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        return df[['Close']].dropna()

    def impute_and_align(self, df):
        """Fill missing data and ensure datetime index."""
        df = df.ffill().bfill()
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
        then split into train/val/test.
        """
        X, y = [], []
        for i in range(len(df) - self.window_size - 1):
            window = df.iloc[i : i + self.window_size].values
            label = 1 if df.iloc[i + self.window_size + 1, 0] > df.iloc[i + self.window_size, 0] else 0
            X.append(window)
            y.append(label)
        X = np.stack(X)
        y = np.array(y)

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
        self.input_proj = nn.Linear(feature_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        # x: [batch, window, feature]
        x = self.input_proj(x)            # [batch, window, d_model]
        x = x.permute(1, 0, 2)            # [window, batch, d_model]
        out = self.encoder(x)             # [window, batch, d_model]
        pooled = out.mean(dim=0)          # [batch, d_model]
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
            # set module to training mode
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
                # set module to eval mode
                super().eval()
                val_loss = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        val_loss += loss_fn(self(xb), yb).item()
                val_loss /= len(val_loader)
                print(f"Epoch {epoch}, val loss: {val_loss:.4f}")

                if val_loss < best_val:
                    best_val = val_loss
                    wait = 0
                    torch.save(self.state_dict(), ckpt_path)
                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping.")
                        break

        # load best model
        self.load_state_dict(torch.load(ckpt_path))


# #########################
# 3. TRF.3: EmbeddingsExtractor
# #########################
class EmbeddingsExtractor:
    """
    Extracts embeddings from a trained TransformerModel.
    """
    def __init__(self, model):
        self.model = model.eval()

    def extract_embeddings(self, loader, output_csv='embeddings.csv'):
        """Generate embeddings (logits) and save with labels."""
        rows = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                emb = self.model(xb).cpu().numpy()
                for vec, label in zip(emb, yb.numpy()):
                    rows.append(list(vec) + [int(label)])
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
        df = pd.read_csv(self.emb_csv)
        X = df.drop('label', axis=1).values
        y = df['label'].values
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

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
# 5. TRF.5: InferencePipeline
# #########################
class InferencePipeline:
    """
    Loads Transformer and XGBoost models, computes scores for a new window, emits JSON.
    """
    def __init__(self, transformer_path='transformer.pt', xgb_path='xgb_model.joblib',
                 window_size=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer = None
        self.transformer_path = transformer_path
        self.xgb = joblib.load(xgb_path)
        self.window_size = window_size
        self.scaler = StandardScaler()

    def load_transformer(self, feature_dim):
        model = TransformerModel(feature_dim)
        model.load_state_dict(torch.load(self.transformer_path, map_location=self.device))
        self.transformer = model.to(self.device).eval()

    def predict(self, ticker, start_date, end_date):
        handler = DataHandler(ticker, start_date, end_date, self.window_size)
        df = handler.fetch_ohlcv()
        df = handler.impute_and_align(df)
        df_norm = handler.normalize(df)

        window = df_norm.values[-self.window_size:]
        feature_dim = window.shape[1]
        if self.transformer is None:
            self.load_transformer(feature_dim)

        xb = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.transformer(xb).cpu().numpy()
            probs = F.softmax(torch.tensor(logits), dim=1).numpy()[0]

        transformer_score = float(probs[1])
        emb = logits
        xgb_prob = float(self.xgb.predict_proba(emb)[0][1])

        result = {
            'date': end_date,
            'transformer_score': transformer_score,
            'xgb_prob': xgb_prob
        }
        print(json.dumps(result))
        return result


# #########################
# Main execution for TRF.1 - TRF.5
# #########################
if __name__ == '__main__':
    ticker = input("Enter stock ticker (e.g. AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date   = input("Enter end date (YYYY-MM-DD): ")

    # 1. Data ingestion & windowing
    handler = DataHandler(ticker, start_date, end_date)
    df = handler.fetch_ohlcv()
    df = handler.impute_and_align(df)
    df_norm = handler.normalize(df)
    X_train, X_val, X_test, y_train, y_val, y_test = handler.create_windows(df_norm)

    train_ld = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ), batch_size=32, shuffle=True)
    val_ld = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    ), batch_size=32)

    # 2. Transformer training
    feature_dim = X_train.shape[2]
    transformer = TransformerModel(feature_dim)
    transformer.train_model(train_ld, val_ld)

    # 3. Embedding extraction & validation
    extractor = EmbeddingsExtractor(transformer)
    extractor.extract_embeddings(train_ld, output_csv='embeddings.csv')
    extractor.validate_embeddings('embeddings.csv', emb_dim=2)

    # 4. XGBoost training
    xgb_trainer = XGBoostTrainer('embeddings.csv')
    xgb_trainer.train(output_model='xgb_model.joblib')

    # 5. (Optional) Inference example
    infer = InferencePipeline('transformer.pt', 'xgb_model.joblib', window_size=handler.window_size)
    infer.predict(ticker, start_date, end_date)
