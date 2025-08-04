import os
import sys
import json
import time
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

# Optional LLM support
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# -----------------------------
# SNIF.1 Return Data Ingestion
# -----------------------------
class ReturnFetcher:
    def __init__(self, source_client=None):
        # Use provided client or default to yfinance
        self.client = source_client or yf

    def fetch_ohlcv(self, tickers, start, end):
        # Download OHLCV data for the given tickers and date range
        return self.client.download(
            tickers, start=start, end=end,
            group_by='ticker', auto_adjust=False, progress=False
        )

    def compute_returns(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        # Extract close prices, handling both single and multi-index formats, then compute percent change
        if isinstance(ohlcv.columns, pd.MultiIndex):
            close = ohlcv.xs('Close', axis=1, level=1)
        else:
            close = ohlcv['Close']
        return close.pct_change().dropna(how='all')

    def clean_and_align(self, returns: pd.DataFrame, max_nan_pct: float = 0.5) -> pd.DataFrame:
        # Drop rows with too many NaNs, then forward/backward fill to align series
        thresh = int((1 - max_nan_pct) * returns.shape[1])
        cleaned = returns.dropna(axis=0, thresh=thresh)
        return cleaned.ffill().bfill()

# -----------------------------
# SNIF.2 Autoencoder Training
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # Encoder: compress input to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(True),
            nn.Linear(128, latent_dim), nn.ReLU(True)
        )
        # Decoder: reconstruct input from latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(True),
            nn.Linear(128, input_dim), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Full autoencoding pass: encode then decode
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Only encode (extract representation)
        return self.encoder(x)

class AutoencoderTrainer:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 lr: float = 1e-3, batch_size: int = 64,
                 epochs: int = 50, device: str = None):
        # Determine compute device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize autoencoder model
        self.model = Autoencoder(input_dim, latent_dim).to(self.device)
        # Loss and optimizer
        self.crit = nn.MSELoss()
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs

    def _prepare_loaders(self, data: torch.Tensor, val_split=0.2):
        # Split into training and validation
        ds = TensorDataset(data)
        val_size = int(len(ds) * val_split)
        train_ds, val_ds = random_split(ds, [len(ds)-val_size, val_size])
        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=self.batch_size)
        )

    def train(self, data: torch.Tensor, checkpoint_dir: str = 'checkpoints/autoencoder'):
        # Train the autoencoder and save best encoder weights based on validation loss
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_loss = float('inf')
        best_path = os.path.join(checkpoint_dir, 'best_encoder.pth')
        train_loader, val_loader = self._prepare_loaders(data.to(self.device))
        for ep in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0
            for (batch,) in train_loader:
                self.opt.zero_grad()
                recon = self.model(batch.to(self.device))
                loss = self.crit(recon, batch.to(self.device))
                loss.backward()
                self.opt.step()
                train_loss += loss.item() * batch.size(0)
            train_loss /= len(train_loader.dataset)
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for (batch,) in val_loader:
                    val_loss += self.crit(self.model(batch.to(self.device)), batch.to(self.device)).item() * batch.size(0)
            val_loss /= len(val_loader.dataset)
            print(f'Epoch {ep}/{self.epochs}: Train {train_loss:.4f}, Val {val_loss:.4f}')
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.encoder.state_dict(), best_path)
        print('SNIF.2 complete: encoder saved to', best_path)

    def load_encoder(self, path: str):
        # Load saved encoder weights
        self.model.encoder.load_state_dict(torch.load(path, map_location=self.device))
        self.model.encoder.eval()

    def extract_embeddings(self, data: torch.Tensor) -> torch.Tensor:
        # Get latent embeddings without gradient tracking
        self.model.eval()
        with torch.no_grad():
            return self.model.encode(data.to(self.device)).cpu()

# -----------------------------
# SNIF.3 Topology Inference
# -----------------------------
class TopologyBuilder:
    def __init__(self, threshold=0.7, output_dir='checkpoints/topology'):
        # Ensure output directory exists for adjacency persistence
        os.makedirs(output_dir, exist_ok=True)
        self.threshold = threshold
        self.output_dir = output_dir

    def compute_similarity(self, emb: np.ndarray) -> np.ndarray:
        # Compute pairwise cosine similarity of embeddings
        return cosine_similarity(emb)

    def sparsify(self, sim: np.ndarray) -> np.ndarray:
        # Threshold similarity to create adjacency matrix and zero out self-loops
        adj = (sim >= self.threshold).astype(float)
        np.fill_diagonal(adj, 0)
        return adj

    def persist(self, adj: np.ndarray, dates: pd.DatetimeIndex):
        # Save full adjacency matrix and per-date edge weights for later inspection
        pd.DataFrame(adj, index=dates, columns=dates).to_csv(
            os.path.join(self.output_dir, 'adjacency_full.csv'))
        for i, dt in enumerate(dates):
            pd.DataFrame(adj[i], index=dates, columns=['edge_weight']).to_csv(
                os.path.join(self.output_dir, f'adjacency_{dt.date()}.csv'))

# -----------------------------
# SNIF.4 GCN+LSTM Development (fixed size mismatch)
# -----------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        # Linear transformation without bias
        self.linear = nn.Linear(in_f, out_f, bias=False)

    def forward(self, x, adj):
        # Graph convolution with symmetric normalization
        I = torch.eye(adj.size(0), device=adj.device)
        A_hat = adj + I  # add self-loops
        deg = A_hat.sum(1)
        D_inv_sqrt = torch.diag(deg.pow(-0.5))
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        return F.relu(self.linear(A_norm @ x))

class GCN_LSTM(nn.Module):
    def __init__(self, feat_dim, gcn_hidden=64, lstm_hidden=64, num_classes=3):
        super().__init__()
        # Two-layer GCN followed by LSTM and a final classifier
        self.gcn1 = GCNLayer(feat_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden)
        self.lstm = nn.LSTM(gcn_hidden, lstm_hidden, num_layers=2, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, seq: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # seq: [1, seq_len, feat_dim]
        x = seq.squeeze(0)  # Shape [seq_len, feat_dim]
        g1 = self.gcn1(x, adj)       # First GCN layer output
        g2 = self.gcn2(g1, adj)      # Second GCN layer output
        lstm_in = g2.unsqueeze(0)    # Prepare for LSTM: [1, seq_len, gcn_hidden]
        lstm_out, _ = self.lstm(lstm_in)
        logits = self.fc(lstm_out[:, -1, :])  # Use last time step for classification
        return logits

# -----------------------------
# SNIF.5 Inference
# -----------------------------
class InferenceEngine:
    def __init__(self, model_pt, enc_pth, input_dim, latent_dim, thresh=0.7):
        # Load scripted GCN+LSTM model
        self.model = torch.jit.load(model_pt)
        self.model.eval()
        # Prepare encoder trainer and load pretrained encoder
        self.encoder = AutoencoderTrainer(input_dim=input_dim, latent_dim=latent_dim)
        self.encoder.load_encoder(enc_pth)
        # Setup topology builder with threshold
        self.topo = TopologyBuilder(thresh)

    def run(self, tickers, start, end):
        # Full pipeline: fetch returns, get embeddings, build adjacency, run model, format output
        fetcher = ReturnFetcher()
        ohlcv = fetcher.fetch_ohlcv(tickers, start, end)
        returns = fetcher.compute_returns(ohlcv)
        cleaned = fetcher.clean_and_align(returns)
        emb = self.encoder.extract_embeddings(torch.tensor(cleaned.values, dtype=torch.float32)).numpy()
        adj = torch.tensor(self.topo.sparsify(self.topo.compute_similarity(emb)), dtype=torch.float32)
        seq = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, feat_dim]
        logits = self.model(seq, adj).squeeze(0)
        probs = F.softmax(logits, dim=-1)
        return json.dumps([{'stock': s, 'snif_prob': probs[i].item()} for i, s in enumerate(tickers)])

# -----------------------------
# SNIF.6 Crew AI Decision Agent (BUY/SELL only from LLM; fallback is HOLD)
# -----------------------------
class CrewAIDecisionAgent:
    def __init__(self):
        # Initialize LLM if available, otherwise fallback to no LLM (always HOLD)
        if LLM_AVAILABLE:
            self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        else:
            self.llm = None

    def _build_prompt(self, stock, prob):
        # Prompt the LLM to make a BUY/SELL/HOLD decision based on snif_prob
        return (
            f"You are a trading decision assistant. Given the stock '{stock}' with snif_prob {prob:.2f}, "
            f"decide whether to BUY, SELL, or HOLD based on your interpretation. "
            f"Respond with exactly a JSON object like "
            f"{{\"recommendation\":\"BUY\"}} or {{\"recommendation\":\"SELL\"}} or {{\"recommendation\":\"HOLD\"}}. "
            f"Do not apply any internal hard thresholds yourself; use your judgment."
        )

    def decide(self, snif_json):
        # Parse input (string or list), iterate over items and get recommendation
        if isinstance(snif_json, str):
            data = json.loads(snif_json)
        else:
            data = snif_json
        out = []
        for item in data:
            stock = item.get("stock")
            prob = item.get("snif_prob")
            recommendation = "HOLD"  # default fallback
            if stock is None or prob is None:
                out.append({"stock": stock, "snif_prob": prob, "recommendation": recommendation})
                continue
            if self.llm:
                prompt = self._build_prompt(stock, prob)
                try:
                    response = self.llm.predict(prompt)
                    parsed = json.loads(response.strip())
                    cand = parsed.get("recommendation", "").upper()
                    if cand in {"BUY", "SELL", "HOLD"}:
                        recommendation = cand
                except Exception:
                    recommendation = "HOLD"
            else:
                recommendation = "HOLD"
            out.append({"stock": stock, "snif_prob": prob, "recommendation": recommendation})
        return out


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Interactive input for tickers and date range
    tickers = input("Enter tickers (comma-separated): ").split(',')
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # SNIF.1–3: Data ingestion, compute returns, clean, get embeddings, and infer topology
    fetcher = ReturnFetcher()
    ohlcv = fetcher.fetch_ohlcv(tickers, start_date, end_date)
    returns = fetcher.compute_returns(ohlcv)
    cleaned = fetcher.clean_and_align(returns)

    # Autoencoder training / loading
    ae = AutoencoderTrainer(input_dim=cleaned.shape[1], latent_dim=16, epochs=30)
    ae.train(torch.tensor(cleaned.values, dtype=torch.float32))
    enc_path = os.path.join("checkpoints", "autoencoder", "best_encoder.pth")
    ae.load_encoder(enc_path)

    # Extract embeddings and build/persist topology adjacency
    emb = ae.extract_embeddings(torch.tensor(cleaned.values, dtype=torch.float32)).numpy()
    topo = TopologyBuilder()
    adjacency = topo.sparsify(topo.compute_similarity(emb))
    topo.persist(adjacency, cleaned.index)
    print("SNIF.3 complete.")

    # SNIF.4: Prepare or load GCN+LSTM student model, script it for inference
    latent_dim = 16
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_dir = os.path.join(root, "models")
    os.makedirs(model_dir, exist_ok=True)
    pth = os.path.join(model_dir, "snif_student.pth")
    pt = os.path.join(model_dir, "snif_student.pt")
    gcn_lstm = GCN_LSTM(feat_dim=latent_dim)
    if os.path.exists(pth):
        # Load existing weights if present
        gcn_lstm.load_state_dict(torch.load(pth, map_location="cpu"))
    else:
        # Save initial (possibly untrained) weights as placeholder
        torch.save(gcn_lstm.state_dict(), pth)
    # Script and persist the model for faster inference
    torch.jit.script(gcn_lstm).save(pt)
    print("SNIF.4 complete.")

    # SNIF.5: Run inference pipeline to get snif output probabilities
    engine = InferenceEngine(pt, enc_path, input_dim=cleaned.shape[1], latent_dim=latent_dim)
    snif_output = engine.run(tickers, start_date, end_date)
    print("SNIF.5 output:", snif_output)

    # SNIF.6: Use CrewAI decision agent to convert probabilities into recommendations
    crew_agent = CrewAIDecisionAgent()
    decisions = crew_agent.decide(snif_output)
    print("SNIF.6 recommendations:", decisions)
