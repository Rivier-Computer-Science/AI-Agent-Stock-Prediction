import os
import json
import tempfile
import torch
import numpy as np
import pandas as pd
import pytest
import torch.nn.functional as F

# Adjust this import if your module path is different
from UI.snif import (
    ReturnFetcher,
    AutoencoderTrainer,
    TopologyBuilder,
    GCN_LSTM,
    CrewAIDecisionAgent,
    InferenceEngine,
)


@pytest.fixture
def synthetic_ohlcv():
    # Create 5 days of synthetic OHLCV for 2 tickers
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    data = {}
    for ticker in ["AAA", "BBB"]:
        close = np.linspace(100, 104, len(dates)) + (0 if ticker == "AAA" else 1)
        df = pd.DataFrame({
            ("Open",): close - 0.5,
            ("High",): close + 0.5,
            ("Low",): close - 1,
            ("Close",): close,
            ("Volume",): np.full(len(dates), 1000),
        }, index=dates)
        # Build MultiIndex for columns
        df.columns = pd.MultiIndex.from_product([[ticker], ["Open", "High", "Low", "Close", "Volume"]])
        data[ticker] = df
    # Concatenate per ticker to get the format yfinance.download returns
    combined = pd.concat([data["AAA"], data["BBB"]], axis=1)
    return combined


def test_return_fetcher_compute_and_clean(synthetic_ohlcv):
    rf = ReturnFetcher()
    returns = rf.compute_returns(synthetic_ohlcv)
    # Expect (days-1) rows, 2 tickers
    assert returns.shape == (4, 2)
    # Introduce a row with NaNs beyond threshold and test clean_and_align
    returns_with_nan = returns.copy()
    returns_with_nan.iloc[1, :] = np.nan
    cleaned = rf.clean_and_align(returns_with_nan, max_nan_pct=0.4)
    # Second row dropped because 100% NaN > 40%
    assert cleaned.shape[0] == 3


def test_autoencoder_trainer_reduces_loss(tmp_path):
    # Small synthetic dataset: 10 samples, 3 features
    np.random.seed(0)
    data = torch.tensor(np.random.rand(10, 3).astype(np.float32))
    trainer = AutoencoderTrainer(input_dim=3, latent_dim=2, epochs=5, batch_size=4)
    # Train
    trainer.train(data, checkpoint_dir=str(tmp_path / "ae_ckpt"))
    # Extract embeddings
    emb = trainer.extract_embeddings(data)
    assert emb.shape == (10, 2)
    # Ensure saved encoder exists
    best = tmp_path / "ae_ckpt" / "best_encoder.pth"
    assert best.exists()


def test_topology_builder_thresholding():
    tb = TopologyBuilder(threshold=0.5, output_dir=tempfile.mkdtemp())
    # Embeddings where first and second are identical, third is different
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    sim = tb.compute_similarity(emb)
    # Diagonal should be 1
    assert np.allclose(np.diag(sim), 1.0)
    adj = tb.sparsify(sim)
    # Because threshold 0.5, first and second connect, third only to none
    assert adj[0, 1] == 1.0 and adj[1, 0] == 1.0
    assert adj[2].sum() == 0.0


def test_gcn_lstm_forward_shape():
    feat_dim = 5
    seq_len = 4
    batch_seq = torch.rand(1, seq_len, feat_dim)
    # adjacency should be (feat_dim x feat_dim)? For GCN we expect number of nodes = feat_dim
    # Here we treat seq elements as nodes; using identity adj
    adj = torch.eye(feat_dim)
    model = GCN_LSTM(feat_dim=feat_dim, gcn_hidden=8, lstm_hidden=6, num_classes=3)
    out = model(batch_seq, adj)
    assert out.shape == (1, 3)  # single batch, 3 classes


def test_crew_ai_decision_agent_fallback(monkeypatch):
    # Force no LLM available scenario by monkeypatching attribute
    from UI.snif import LLM_AVAILABLE
    monkeypatch.setattr("UI.snif.LLM_AVAILABLE", False)
    agent = CrewAIDecisionAgent()
    # Provide malformed and valid input
    sample = [{"stock": "XYZ", "snif_prob": 0.7}]
    result = agent.decide(sample)
    assert isinstance(result, list)
    assert result[0]["recommendation"] == "HOLD"


def test_crew_ai_decision_agent_with_mocked_llm(monkeypatch):
    # Simulate LLM giving a BUY recommendation
    class DummyLLM:
        def predict(self, prompt):
            return json.dumps({"recommendation": "BUY"})

    agent = CrewAIDecisionAgent()
    agent.llm = DummyLLM()
    sample = [{"stock": "XYZ", "snif_prob": 0.9}]
    result = agent.decide(sample)
    assert result[0]["recommendation"] == "BUY"


def test_inference_engine_with_mocked_components(tmp_path, monkeypatch):
    # Create synthetic OHLCV and patch ReturnFetcher to return it
    dates = pd.date_range("2025-01-01", periods=6, freq="D")
    close = np.linspace(100, 105, len(dates))
    df = pd.DataFrame({
        ("AAA", "Open"): close - 0.5,
        ("AAA", "High"): close + 0.5,
        ("AAA", "Low"): close - 1,
        ("AAA", "Close"): close,
        ("AAA", "Volume"): np.full(len(dates), 1000),
    }, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # Patch ReturnFetcher.fetch_ohlcv
    def fake_fetch(self, tickers, start, end):
        return df
    monkeypatch.setattr(ReturnFetcher, "fetch_ohlcv", fake_fetch)

    # Prepare autoencoder checkpoint
    trainer = AutoencoderTrainer(input_dim=1, latent_dim=2, epochs=2)
    # Create minimal data: returns of length 5
    rf = ReturnFetcher()
    returns = rf.compute_returns(df)
    cleaned = rf.clean_and_align(returns)
    data_tensor = torch.tensor(cleaned.values, dtype=torch.float32)
    trainer.train(data_tensor, checkpoint_dir=str(tmp_path / "ae_ckpt"))
    enc_path = os.path.join(str(tmp_path / "ae_ckpt"), "best_encoder.pth")
    trainer.load_encoder(enc_path)

    # Build a dummy GCN_LSTM model and script it
    feat_dim = 2
    model = GCN_LSTM(feat_dim=feat_dim, gcn_hidden=4, lstm_hidden=4, num_classes=3)
    scripted = torch.jit.script(model)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    pt_path = str(model_dir / "snif_student.pt")
    scripted.save(pt_path)

    # Run inference engine
    engine = InferenceEngine(pt_path, enc_path, input_dim=cleaned.shape[1], latent_dim=2)
    output_json = engine.run(["AAA"], dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d"))
    parsed = json.loads(output_json)
    assert isinstance(parsed, list)
    assert parsed[0]["stock"] == "AAA"
    assert "snif_prob" in parsed[0]
    prob = parsed[0]["snif_prob"]
    assert 0.0 <= prob <= 1.0
