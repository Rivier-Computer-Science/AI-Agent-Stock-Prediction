MODULE_UNDER_TEST = "trf_pipeline"

import os
import json
import importlib
import numpy as np
import pandas as pd
import torch
import pytest

# ---------------------------------------------------------------------
# Import the module under test once. We then pull references to the
# classes we need so reads/writes are explicit (and easy to monkeypatch).
# ---------------------------------------------------------------------
mod = importlib.import_module(MODULE_UNDER_TEST)

DataHandler = mod.DataHandler
TransformerModel = mod.TransformerModel
EmbeddingsExtractor = mod.EmbeddingsExtractor
XGBoostTrainer = mod.XGBoostTrainer
CrewAIDecisionAgent = mod.CrewAIDecisionAgent
InferencePipeline = mod.InferencePipeline


# ---------------------------------------------------------------------
# Test fixtures & helpers
# ---------------------------------------------------------------------
@pytest.fixture(autouse=True)
def disable_telemetry_and_require_fake_openai(monkeypatch):
    """
    Auto-applied to every test:
      - Disable CrewAI/OpenTelemetry tracing to avoid outbound POSTs.
      - Provide a dummy OPENAI_API_KEY since the production code requires it.
    """
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    # The code raises if this is missing (no fallback), so set a dummy token.
    monkeypatch.setenv("OPENAI_API_KEY", "pytest-dummy-key")


def _make_df(n=60, start="2025-01-01", start_price=100.0, step=0.2):
    """
    Build a simple synthetic Close-price time series:
      Close_t = start_price + step * t
    That monotonic structure makes label creation deterministic (mostly ups).
    """
    idx = pd.date_range(start, periods=n, freq="D")
    close = start_price + step * np.arange(n, dtype=float)
    return pd.DataFrame({"Close": close}, index=idx)


# ---------------------------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------------------------
def test_datahandler_normalize_and_windows(monkeypatch):
    """
    Unit-test DataHandler:
      - fetch_ohlcv() is patched to avoid network and return synthetic data
      - normalize() should produce ~0-mean z-scored Close
      - create_windows() should produce (N, window, features) with labels in {0,1}
    """
    # 1) Patch yfinance.download to deterministic synthetic data (no I/O).
    def fake_download(ticker, start, end, progress=False):
        return _make_df(50)
    monkeypatch.setattr(mod.yf, "download", fake_download)

    # 2) Create handler and verify pipeline shapes/values.
    h = DataHandler("AAPL", "2025-01-01", "2025-03-30", window_size=10)
    df = h.fetch_ohlcv()
    assert "Close" in df.columns and len(df) == 50

    df = h.impute_and_align(df)
    df_norm = h.normalize(df)

    # z-score sanity check: mean should be ~0 (tolerance because of finite-sample).
    assert abs(float(df_norm["Close"].mean())) < 1e-6

    X_train, X_val, X_test, y_train, y_val, y_test = h.create_windows(df_norm)

    # Expect 3D arrays: (batch, window_size, feature_dim), feature_dim==1 for "Close".
    assert X_train.ndim == 3 and X_train.shape[2] == 1
    assert X_train.shape[1] == 10

    # Labels must be 0/1
    assert set(np.unique(y_train)).issubset({0, 1})


def test_transformer_forward_shape():
    """
    Unit-test TransformerModel forward pass:
      - Input (batch=4, window=10, feature_dim=1)
      - Output (batch=4, logits=2)
    """
    model = TransformerModel(feature_dim=1)
    x = torch.randn(4, 10, 1)
    out = model(x)
    assert out.shape == (4, 2)


def test_embeddings_extractor_writes_csv(tmp_path):
    """
    Unit-test EmbeddingsExtractor:
      - Generates logits for a few samples
      - Writes CSV with columns: emb_0, emb_1, label
      - Row count must match number of samples
    """
    # A tiny eval-only model and a small batch of random data.
    model = TransformerModel(feature_dim=1).eval()
    X = torch.randn(5, 10, 1)
    y = torch.randint(0, 2, (5,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=2
    )

    out_csv = tmp_path / "embeddings.csv"
    EmbeddingsExtractor(model).extract_embeddings(loader, output_csv=str(out_csv))

    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["emb_0", "emb_1", "label"]
    assert len(df) == 5
    assert set(df["label"].unique()).issubset({0, 1})


def test_xgb_trainer_trains_and_returns_model(tmp_path, monkeypatch):
    """
    Unit-test XGBoostTrainer.train():
      - Provide a tiny embeddings CSV
      - Patch GridSearchCV to a fast fake that sets best_estimator_
      - Ensure the returned object has predict_proba()
    """
    # 1) Create a tiny embeddings CSV with random numbers.
    n = 30
    df = pd.DataFrame({
        "emb_0": np.random.randn(n),
        "emb_1": np.random.randn(n),
        "label": np.random.randint(0, 2, size=n),
    })
    emb_csv = tmp_path / "emb.csv"
    df.to_csv(emb_csv, index=False)

    # 2) Provide a dummy "best model" that has predict_proba for downstream use.
    class DummyModel:
        def predict_proba(self, X):
            # Return [p_down, p_up] rows; here we nudge p_up > p_down.
            return np.tile([0.4, 0.6], (len(X), 1))

    # 3) Fake a minimal GridSearchCV that instantly "fits."
    class FakeGrid:
        def __init__(self, clf, param_grid, cv, scoring, verbose):
            self.best_estimator_ = DummyModel()
            self.best_params_ = {"n_estimators": 1, "max_depth": 1, "learning_rate": 0.1}
        def fit(self, X, y):
            return self

    # 4) Patch GridSearchCV and joblib.dump inside the module under test.
    monkeypatch.setattr(mod, "GridSearchCV", FakeGrid)
    monkeypatch.setattr(mod.joblib, "dump", lambda model, path: None)

    trainer = XGBoostTrainer(str(emb_csv))
    best = trainer.train(output_model=str(tmp_path / "xgb.joblib"))
    assert hasattr(best, "predict_proba")


def test_crewai_decision_agent_decide_monkeypatched(monkeypatch):
    """
    Unit-test CrewAIDecisionAgent.decide():
      - Patch ChatOpenAI/Agent/Crew so NO real LLM call happens
      - Force Crew.kickoff() to return a JSON string
      - Ensure the parsed output is normalized and returned properly
    """
    # Dummy LLM client (constructor only; never used for network).
    class DummyChat:
        def __init__(self, *a, **k): pass

    # Dummy Agent (accepts the same kwargs as real Agent).
    class DummyAgent:
        def __init__(self, *a, **k): pass

    # Fake Crew that "runs" and returns a JSON string.
    class FakeCrew:
        def __init__(self, agents, tasks, process): pass
        def kickoff(self):
            return '{"recommendation":"SELL"}'

    # Replace the actual classes in the target module with our fakes.
    monkeypatch.setattr(mod, "ChatOpenAI", DummyChat)
    monkeypatch.setattr(mod, "Agent", DummyAgent)
    monkeypatch.setattr(mod, "Crew", FakeCrew)

    agent = CrewAIDecisionAgent(model_name="gpt-4o", temperature=0.0)
    out = agent.decide({"stock": "AAPL", "xgb_prob": 0.42})
    assert out == [{"stock": "AAPL", "xgb_prob": 0.42, "recommendation": "SELL"}]


# ---------------------------------------------------------------------
# INTEGRATION TEST
# ---------------------------------------------------------------------
def test_integration_predict_prints_and_exits(monkeypatch, capsys):
    """
    End-to-end test for InferencePipeline.predict():
      - Mock yfinance.download to use synthetic prices (no network).
      - Monkeypatch joblib.load to return a fake XGB model with constant probs.
      - Monkeypatch InferencePipeline.load_transformer() to bypass disk I/O
        and use a fresh, tiny Transformer in eval mode.
      - Monkeypatch CrewAIDecisionAgent.decide() to return a fixed recommendation.
      - Monkeypatch os._exit to raise SystemExit so pytest can assert stdout.
      - Verify:
          * "CrewAI Decision: BUY" gets printed
          * The JSON payload contains the required keys/ticker and a recommendation
    """
    # 1) Replace yfinance download with deterministic synthetic data.
    def fake_download(ticker, start, end, progress=False):
        return _make_df(40)  # Enough rows to form several windows
    monkeypatch.setattr(mod.yf, "download", fake_download)

    # 2) Replace joblib.load so we don't rely on a real .joblib file on disk.
    class FakeXGBModel:
        def predict_proba(self, X):
            # Always return 70% upward probability for simplicity.
            return np.tile([0.3, 0.7], (len(X), 1))
    monkeypatch.setattr(mod.joblib, "load", lambda path: FakeXGBModel())

    # 3) Avoid reading a real transformer checkpoint; supply a fresh model instead.
    def fake_load_transformer(self, feature_dim):
        self.transformer = TransformerModel(feature_dim).eval()
    monkeypatch.setattr(InferencePipeline, "load_transformer", fake_load_transformer)

    # 4) Avoid real LLM: force "BUY" recommendation for predictable assertions.
    def fake_decide(self, items):
        if isinstance(items, dict):
            items = [items]
        outs = []
        for it in items:
            outs.append({
                "stock": it["stock"],
                "xgb_prob": float(it["xgb_prob"]),
                "recommendation": "BUY"
            })
        return outs
    monkeypatch.setattr(CrewAIDecisionAgent, "decide", fake_decide)

    # 5) Intercept os._exit so the process doesn't actually terminate the test run.
    def fake_exit(code):
        raise SystemExit(code)
    monkeypatch.setattr(mod.os, "_exit", fake_exit)

    # Execute predict() and assert we exit via SystemExit (our fake).
    pipe = InferencePipeline(
        transformer_path="transformer.pt",
        xgb_path="xgb_model.joblib",
        window_size=10,
        model_name="gpt-4o",
        temperature=0.0
    )
    with pytest.raises(SystemExit) as se:
        pipe.predict("AAPL", "2025-01-01", "2025-02-15")
    # Code may be 0 or any int (we don't assert specific code, only that it exits).
    assert isinstance(se.value.code, (int, type(None)))

    # Capture stdout and verify both the human-readable and JSON outputs exist.
    out, err = capsys.readouterr()
    assert "CrewAI Decision: BUY" in out

    # Pull the last JSON-looking line and assert required keys/values.
    json_lines = [l for l in out.strip().splitlines() if l.startswith("{") and l.endswith("}")]
    assert json_lines, "Expected a JSON payload on stdout but found none."
    payload = json.loads(json_lines[-1])
    assert payload["ticker"] == "AAPL"
    assert "transformer_score" in payload
    assert "xgb_prob" in payload
    assert payload["recommendation"] == "BUY"
