import os
import json
import time
import torch
import numpy as np
import pandas as pd
import pytest

from UI.snif import (
    ReturnFetcher,
    AutoencoderTrainer,
    TopologyBuilder,
    GCN_LSTM,
    InferenceEngine,
    CrewAIDecisionAgent,
)

# A deterministic dummy LLM mock for integration decision
class DummyLLM:
    def predict(self, prompt: str):
        # Always BUY for testing
        return json.dumps({"recommendation": "BUY"})

@pytest.fixture
def synthetic_multi_ticker_ohlcv():
    # 8 days of synthetic close prices for two tickers
    dates = pd.date_range("2025-01-01", periods=8, freq="D")
    frames = []
    for ticker in ["AAA", "BBB"]:
        close = np.linspace(50, 57, len(dates)) + (0 if ticker == "AAA" else 1)
        df = pd.DataFrame({
            ("Open",): close - 0.3,
            ("High",): close + 0.3,
            ("Low",): close - 0.6,
            ("Close",): close,
            ("Volume",): np.full(len(dates), 500),
        }, index=dates)
        df.columns = pd.MultiIndex.from_product([[ticker], ["Open", "High", "Low", "Close", "Volume"]])
        frames.append(df)
    combined = pd.concat(frames, axis=1)
    return combined, dates

def test_full_pipeline_end_to_end(tmp_path, synthetic_multi_ticker_ohlcv, monkeypatch):
    # Patch fetcher to return synthetic data
    ohlcv_df, dates = synthetic_multi_ticker_ohlcv

    def fake_fetch(self, tickers, start, end):
        return ohlcv_df
    monkeypatch.setattr(ReturnFetcher, "fetch_ohlcv", fake_fetch)

    # Data ingestion
    rf = ReturnFetcher()
    returns = rf.compute_returns(ohlcv_df)
    cleaned = rf.clean_and_align(returns)

    # Autoencoder train
    ae_dir = tmp_path / "checkpoints" / "autoencoder"
    ae_trainer = AutoencoderTrainer(input_dim=cleaned.shape[1], latent_dim=4, epochs=3)
    ae_trainer.train(torch.tensor(cleaned.values, dtype=torch.float32), checkpoint_dir=str(ae_dir))
    enc_path = os.path.join(str(ae_dir), "best_encoder.pth")
    ae_trainer.load_encoder(enc_path)

    # Topology
    emb = ae_trainer.extract_embeddings(torch.tensor(cleaned.values, dtype=torch.float32)).numpy()
    topo = TopologyBuilder(threshold=0.5, output_dir=str(tmp_path / "checkpoints" / "topology"))
    adj = topo.sparsify(topo.compute_similarity(emb))
    topo.persist(adj, cleaned.index)

    # GCN+LSTM placeholder: script and save
    model = GCN_LSTM(feat_dim=4, gcn_hidden=8, lstm_hidden=6, num_classes=3)
    # Save untrained
    model_dir = tmp_path / "models"
    os.makedirs(model_dir, exist_ok=True)
    pt_path = os.path.join(str(model_dir), "snif_student.pt")
    torch.jit.script(model).save(pt_path)

    # Inference
    engine = InferenceEngine(pt_path, enc_path, input_dim=cleaned.shape[1], latent_dim=4)
    start = cleaned.index[0].strftime("%Y-%m-%d")
    end = cleaned.index[-1].strftime("%Y-%m-%d")
    snif_output = engine.run(["AAA", "BBB"], start, end)
    parsed = json.loads(snif_output)
    assert len(parsed) == 2
    for item in parsed:
        assert "stock" in item and "snif_prob" in item
        assert 0.0 <= item["snif_prob"] <= 1.0

    # Crew AI decision with mocked LLM
    crew = CrewAIDecisionAgent()
    crew.llm = DummyLLM()
    decisions = crew.decide(parsed)
    assert isinstance(decisions, list)
    for d in decisions:
        assert d["recommendation"] == "BUY"

    # Basic latency check: full pipeline should complete within a reasonable time (e.g., 5s)
    t0 = time.time()
    _ = engine.run(["AAA", "BBB"], start, end)
    t1 = time.time()
    assert (t1 - t0) < 5.0, "Pipeline latency exceeded 5 seconds"
