import os
import json
import types
import importlib
import importlib.util
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import torch

# ------------------------------------------------------------
# Helper: dynamically load the target module under test (MUT)
# ------------------------------------------------------------
def _load_target_module():
    """
    Try to import the GNN pipeline module to test. We support 3 strategies:
      1) If TARGET_MODULE_NAME env var is set, import by that module name.
      2) If TARGET_MODULE_PATH env var is set, import by file path.
      3) Fallback to trying 'main' and then 'gnn_main' by name.

    Usage:
      export TARGET_MODULE_PATH=/path/to/your_script.py
      or
      export TARGET_MODULE_NAME=your_module_name
    """
    name = os.environ.get("TARGET_MODULE_NAME")
    path = os.environ.get("TARGET_MODULE_PATH")

    if name:
        try:
            return importlib.import_module(name)
        except Exception as e:
            raise RuntimeError(f"Failed to import module name '{name}': {e}")

    if path:
        try:
            spec = importlib.util.spec_from_file_location("gnn_module", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            raise RuntimeError(f"Failed to import module path '{path}': {e}")

    # Fallbacks (common names)
    for candidate in ("main", "gnn_main"):
        try:
            return importlib.import_module(candidate)
        except Exception:
            continue

    raise RuntimeError(
        "Could not import the target module. "
        "Set TARGET_MODULE_NAME or TARGET_MODULE_PATH environment variable."
    )


# -------------#
#  Test Setup  #
# -------------#
@pytest.fixture(scope="session")
def mut():
    """Load the module-under-test once for all tests."""
    return _load_target_module()


@pytest.fixture(autouse=True)
def set_seed():
    """Set seeds for determinism in tests where feasible."""
    np.random.seed(7)
    torch.manual_seed(7)
    yield


# ============================================================
# Unit tests
# ============================================================

def test_extract_first_json_object_direct(mut):
    """# Unit: JSON extractor should parse a clean JSON string."""
    s = '{"recommendation":"BUY"}'
    obj = mut._extract_first_json_object(s)
    assert obj == {"recommendation": "BUY"}


def test_extract_first_json_object_embedded(mut):
    """# Unit: JSON extractor should find first {...} within noisy text."""
    s = 'some header\n\n{ "recommendation" : "HOLD" }\ntrailer'
    obj = mut._extract_first_json_object(s)
    assert obj["recommendation"] == "HOLD"


def test_price_loader_fetch_and_returns_singlecol(monkeypatch, tmp_path, mut):
    """# Unit: PriceLoader should handle single-column (non-MultiIndex) download and compute returns."""
    # --- Arrange: monkeypatch yf.download to return a simple DataFrame with 'Adj Close'
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"Adj Close": [100, 101, 103, 102, 104]}, index=dates)

    def fake_download(tickers, start, end, progress, auto_adjust, threads):
        return prices

    monkeypatch.setattr(mut.yf, "download", fake_download)
    monkeypatch.setattr(mut, "SCRIPT_DIR", str(tmp_path))  # write into temp dir

    tickers = ["AAPL"]
    loader = mut.PriceLoader(tickers, "2020-01-01", "2020-01-10", data_dir="data")
    df_prices = loader.fetch_ohlc_data()
    assert "AAPL" in df_prices.columns and df_prices.shape == (5, 1)

    df_returns = loader.compute_daily_returns()
    # 4 rows (pct_change drops first), 1 column
    assert df_returns.shape == (4, 1)
    assert not df_returns.isna().any().any()

    np_paths = loader.persist_tensors()
    # Ensure files exist
    for p in np_paths:
        assert os.path.exists(p)


def test_price_loader_fetch_multiindex(monkeypatch, tmp_path, mut):
    """# Unit: PriceLoader should handle MultiIndex column format from yfinance."""
    # --- Arrange a MultiIndex DataFrame like yfinance returns: levels ('Adj Close', ticker)
    dates = pd.date_range("2020-02-01", periods=4, freq="D")
    cols = pd.MultiIndex.from_product([["Adj Close"], ["AAPL", "MSFT"]])
    data = np.array([[100, 200], [101, 202], [102, 204], [104, 203]], dtype=float)
    df_multi = pd.DataFrame(data=data, index=dates, columns=cols)

    def fake_download(tickers, start, end, progress, auto_adjust, threads):
        return df_multi

    monkeypatch.setattr(mut.yf, "download", fake_download)
    monkeypatch.setattr(mut, "SCRIPT_DIR", str(tmp_path))
    tickers = ["AAPL", "MSFT"]
    loader = mut.PriceLoader(tickers, "2020-02-01", "2020-02-10", data_dir="data")
    df_prices = loader.fetch_ohlc_data()
    assert list(df_prices.columns) == tickers
    assert df_prices.shape == (4, 2)


def test_graph_builder_outputs(monkeypatch, tmp_path, mut):
    """# Unit: GraphBuilder should write adjacency files for each valid window end-date."""
    # --- Arrange: small returns DataFrame
    monkeypatch.setattr(mut, "SCRIPT_DIR", str(tmp_path))
    dates = pd.date_range("2020-03-01", periods=5, freq="D")
    returns_df = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=["A", "B", "C"])
    gb = mut.GraphBuilder(returns_df, window_size=3, threshold=0.0, output_dir="adjacency_data")
    gb.build_graphs()

    out_dir = tmp_path / "adjacency_data"
    files = sorted([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    # window_size=3 with 5 days => adjacency from index 2..4 => 3 files
    assert len(files) == 3
    # Check shapes
    arr = np.load(os.path.join(out_dir, files[0]))
    assert arr.shape == (3, 3)
    # diagonal should be zero
    assert np.all(np.diag(arr) == 0.0)


def test_teacher_gnn_forward_shape(mut):
    """# Unit: TeacherGNN forward should return logits with shape (N, 2)."""
    model = mut.TeacherGNN(input_dim=1, hidden_dim=8, gru_hidden_dim=16, output_dim=2)
    seq_len, N = 5, 4
    features = torch.randn(seq_len, N)
    adj = torch.eye(N)
    logits = model(features, adj)
    assert logits.shape == (N, 2)


def test_student_gnn_forward_shape(mut):
    """# Unit: StudentGNN forward reuses TeacherGNN and preserves output shape."""
    model = mut.StudentGNN(input_dim=1, hidden_dim=8, gru_hidden_dim=16, output_dim=2)
    seq_len, N = 3, 5
    features = torch.randn(seq_len, N)
    adj = torch.eye(N)
    logits = model(features, adj)
    assert logits.shape == (N, 2)


def test_decision_agent_llm_success(monkeypatch, mut):
    """# Unit: DecisionAgent should parse JSON from LLM and return recommendation."""
    # --- Mock the LLM to return a proper JSON string.
    def fake_call_llm_json(prompt: str, model: str = None) -> str:
        return json.dumps({"recommendation": "SELL"})

    monkeypatch.setattr(mut, "_call_llm_json", fake_call_llm_json)
    agent = mut.DecisionAgent(model="anything")
    result = agent.get_decision(0.1)
    assert result == {"recommendation": "SELL"}


def test_decision_agent_fallback(monkeypatch, mut):
    """# Unit: DecisionAgent should fall back to local rule when LLM fails."""
    # --- Force the LLM call to raise an error to trigger fallback path.
    def raise_call(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(mut, "_call_llm_json", raise_call)
    agent = mut.DecisionAgent(model="anything")
    # 0.8 should map to "BUY" in fallback rule
    result = agent.get_decision(0.8)
    assert result == {"recommendation": "BUY"}


# ============================================================
# Integration tests
# ============================================================

def test_inference_and_chain_integration(tmp_path, monkeypatch, mut):
    """# Integration: Build minimal FS layout to run Inference and chain to DecisionAgent."""
    # --- Redirect SCRIPT_DIR to temp path so the code writes under tmp
    monkeypatch.setattr(mut, "SCRIPT_DIR", str(tmp_path))

    # --- Create minimal 'data/returns.npy' expected by Inference
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    N = 3           # number of tickers/nodes
    delta = 2       # sequence length used in inference
    T = 5           # number of time rows in returns
    returns_np = np.random.randn(T, N).astype(np.float32)
    np.save(data_dir / "returns.npy", returns_np)

    # --- Create minimal 'adjacency_data/adjacency_YYYY-MM-DD.npy'
    adj_dir = tmp_path / "adjacency_data"
    adj_dir.mkdir(parents=True, exist_ok=True)
    latest_adj = np.eye(N, dtype=np.float32)
    np.save(adj_dir / "adjacency_2023-12-31.npy", latest_adj)

    # --- Create a dummy student model checkpoint expected by Inference
    model_dir = tmp_path / "student_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    dummy_model = mut.StudentGNN(1, 32, 64, 2)
    torch.save(dummy_model.state_dict(), model_dir / "best_student.pth")

    # --- Prepare Inference object
    tickers = ["AAA", "BBB", "CCC"]
    infer = mut.Inference(str(model_dir / "best_student.pth"), tickers, delta)

    # --- Run inference to get probabilities
    preds = infer.run()
    assert isinstance(preds, list)
    assert len(preds) == N
    for item in preds:
        assert "stock" in item and "student_prob" in item
        assert isinstance(item["student_prob"], float)

    # --- Mock DecisionAgent LLM to return HOLD for all
    def fake_call_llm_json(prompt: str, model: str = None) -> str:
        return json.dumps({"recommendation": "HOLD"})
    monkeypatch.setattr(mut, "_call_llm_json", fake_call_llm_json)

    # --- Chain predictions into decisions
    agent = mut.DecisionAgent(model="anything")
    final = mut.chain_inference_to_decision(preds, agent)

    # --- Validate enriched output
    assert len(final) == len(preds)
    for item in final:
        assert "recommendation" in item
        assert item["recommendation"] in {"BUY", "SELL", "HOLD"}


def test_price_to_graph_to_inference_integration(tmp_path, monkeypatch, mut):
    """# Integration: From synthetic PriceLoader -> GraphBuilder -> (mock) training -> Inference path."""
    # --- Redirect SCRIPT_DIR
    monkeypatch.setattr(mut, "SCRIPT_DIR", str(tmp_path))

    # --- Mock yfinance to return small MultiIndex Adj Close data for two tickers
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    tickers = ["AAA", "BBB"]
    mi_cols = pd.MultiIndex.from_product([["Adj Close"], tickers])
    values = np.array([[100, 50],
                       [101, 50.5],
                       [102, 50.0],
                       [103, 49.5],
                       [102, 49.0],
                       [104, 50.0]], dtype=float)
    df_yf = pd.DataFrame(values, index=dates, columns=mi_cols)

    def fake_download(*args, **kwargs):
        return df_yf

    monkeypatch.setattr(mut.yf, "download", fake_download)

    # --- PriceLoader
    loader = mut.PriceLoader(tickers, "2020-01-01", "2020-02-01", data_dir="data")
    prices = loader.fetch_ohlc_data()
    rets = loader.compute_daily_returns()
    loader.persist_tensors()
    assert prices.shape[1] == len(tickers)
    assert rets.shape[0] == len(dates) - 1

    # --- GraphBuilder
    gb = mut.GraphBuilder(rets, window_size=3, threshold=0.0, output_dir="adjacency_data")
    gb.build_graphs()
    adj_files = sorted([f for f in os.listdir(tmp_path / "adjacency_data") if f.endswith(".npy")])
    assert len(adj_files) >= 1

    # --- Create dummy teacher/student checkpoints (skip heavy training)
    teacher_dir = tmp_path / "teacher_model"
    student_dir = tmp_path / "student_model"
    teacher_dir.mkdir(exist_ok=True)
    student_dir.mkdir(exist_ok=True)

    # Save a random-initialized teacher model to expected path
    teacher_model = mut.TeacherGNN(1, 32, 64, 2)
    torch.save(teacher_model.state_dict(), teacher_dir / "best_teacher.pth")

    # Save a random-initialized student model to expected path
    student_model = mut.StudentGNN(1, 32, 64, 2)
    torch.save(student_model.state_dict(), student_dir / "best_student.pth")

    # --- Inference (using returns.npy and latest adjacency file already present)
    infer = mut.Inference(str(student_dir / "best_student.pth"), tickers, delta=3)
    preds = infer.run()
    assert len(preds) == len(tickers)
    for p in preds:
        assert set(p.keys()) == {"stock", "student_prob"}
        assert isinstance(p["student_prob"], float)

    # --- DecisionAgent mocked to BUY when prob > 0.5 else HOLD to ensure variability
    def fake_llm(prompt: str, model: str = None) -> str:
        # Read the student_prob value from the prompt to produce variable outputs
        start = prompt.find('"student_prob":')
        assert start != -1
        num_str = prompt[start:].split()[1].strip(",} ")
        try:
            val = float(num_str)
        except Exception:
            val = 0.0
        rec = "BUY" if val > 0.5 else "HOLD"
        return json.dumps({"recommendation": rec})

    monkeypatch.setattr(mut, "_call_llm_json", fake_llm)
    agent = mut.DecisionAgent(model="mock")
    enriched = mut.chain_inference_to_decision(preds, agent)
    assert len(enriched) == len(preds)
    assert all("recommendation" in x for x in enriched)
