# test_pipeline_agents_all.py
"""
Combined unit + integration tests for pipeline_agents.py

How to run:
    pytest -q

What these tests do:
- Mock yfinance and OpenAI so no external calls happen.
- Check pure helpers, ratio math, model training, inference.
- Exercise the CLI main() end-to-end with controlled inputs.
- Verify artifact files are created and output contains expected markers.

Requirements (install if needed):
    pip install pytest numpy pandas torch scikit-learn joblib yfinance
"""

import os
import json
import types
import builtins
import numpy as np
import pandas as pd
import pytest
import torch

# Import the system under test
import pipeline_agents as pa


# -----------------------------------------------------------------------------
# Helpers / fakes
# -----------------------------------------------------------------------------

class FakeTicker:
    """
    Mimics yfinance.Ticker object enough for our code to work.
    We provide "quarterly_financials" and "quarterly_balance_sheet"
    with columns = dates (Timestamp) and rows = line items.
    """
    def __init__(self):
        dates = [
            pd.Timestamp("2025-06-30"),
            pd.Timestamp("2025-03-31"),
            pd.Timestamp("2024-12-31"),
            pd.Timestamp("2024-09-30"),
        ]
        # Income (rows=line items, cols=dates)
        self.quarterly_financials = pd.DataFrame(
            data=[
                [120.0, 110.0, 105.0, 100.0],  # Total Revenue
                [ 30.0,  27.0,  25.0,  20.0],  # Net Income
            ],
            index=["Total Revenue", "Net Income"],
            columns=dates,
        )
        # Balance (rows=line items, cols=dates)
        self.quarterly_balance_sheet = pd.DataFrame(
            data=[
                [60.0, 58.0, 55.0, 50.0],  # Total Stockholder Equity
                [20.0, 21.0, 22.0, 23.0],  # Total Liab
            ],
            index=["Total Stockholder Equity", "Total Liab"],
            columns=dates,
        )


class FakeTickerTwoQuarters:
    """Smaller ticker (2 quarters) to simplify some unit tests."""
    def __init__(self):
        dates = [pd.Timestamp("2025-06-30"), pd.Timestamp("2025-03-31")]
        self.quarterly_financials = pd.DataFrame(
            data=[[100.0, 80.0], [10.0, 8.0]],
            index=["Total Revenue", "Net Income"],
            columns=dates,
        )
        self.quarterly_balance_sheet = pd.DataFrame(
            data=[[50.0, 40.0], [25.0, 20.0]],
            index=["Total Stockholder Equity", "Total Liab"],
            columns=dates,
        )


class FakeTickerEmpty:
    """Ticker that returns no data; used to test error handling."""
    def __init__(self):
        self.quarterly_financials = pd.DataFrame()
        self.quarterly_balance_sheet = pd.DataFrame()


class FakeOpenAIResponse:
    """Shape-compatible with openai.ChatCompletion.create return value."""
    def __init__(self, content: str):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


# -----------------------------------------------------------------------------
# Unit tests: pure helpers
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "inp,expect",
    [
        (0, 0.0),
        ("1.23", 1.23),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (None, 0.0),
        ("bad", 0.0),
    ],
)
def test__to_float_or_zero(inp, expect):
    """Converts messy inputs to finite floats or 0.0."""
    assert pa._to_float_or_zero(inp) == expect


def test__parse_date_valid_and_invalid():
    """Parses valid YYYY-MM-DD and raises for invalid."""
    assert pa._parse_date("2025-01-31").year == 2025
    with pytest.raises(ValueError):
        pa._parse_date("2025/01/31")  # wrong format
    with pytest.raises(ValueError):
        pa._parse_date("not-a-date")


# -----------------------------------------------------------------------------
# Unit tests: Data fetching
# -----------------------------------------------------------------------------

def test_datafetch_returns_records(monkeypatch):
    """Ensures DataFetch turns yfinance frames into clean lists of dicts."""
    monkeypatch.setattr(pa.yf, "Ticker", lambda _: FakeTickerTwoQuarters())
    df = pa.DataFetch(num_quarters=4)
    inc = df.fetch_income_statement("AAPL")
    bs = df.fetch_balance_sheet("AAPL")

    assert len(inc) == 2
    assert len(bs) == 2
    assert set(inc[0].keys()) == {"fiscalDateEnding", "totalRevenue", "netIncome"}
    assert set(bs[0].keys()) == {"fiscalDateEnding", "totalShareholderEquity", "totalLiabilities"}


def test_datafetch_raises_when_empty(monkeypatch):
    """If yfinance returns empty frames, DataFetch should raise ValueError."""
    monkeypatch.setattr(pa.yf, "Ticker", lambda _: FakeTickerEmpty())
    df = pa.DataFetch()
    with pytest.raises(ValueError):
        df.fetch_income_statement("AAPL")
    with pytest.raises(ValueError):
        df.fetch_balance_sheet("AAPL")


# -----------------------------------------------------------------------------
# Unit tests: Ratio calculation
# -----------------------------------------------------------------------------

def test_ratio_calc_handles_div_by_zero_and_filters():
    """
    Validates that division-by-zero cases are handled (ratios become 0),
    and date filtering is inclusive and sorted desc by date.
    """
    rc = pa.RatioCalc()
    # Crafted inputs with zeros to trigger safe divisions
    income_reports = [
        {"fiscalDateEnding": "2025-06-30", "totalRevenue": 0, "netIncome": 10},   # rev=0
        {"fiscalDateEnding": "2025-03-31", "totalRevenue": 100, "netIncome": 0},  # net=0
    ]
    balance_reports = [
        {"fiscalDateEnding": "2025-06-30", "totalShareholderEquity": 0, "totalLiabilities": 20},  # equity=0
        {"fiscalDateEnding": "2025-03-31", "totalShareholderEquity": 50, "totalLiabilities": 25},
    ]
    ratios = rc.compute_ratios(income_reports, balance_reports)
    # First row: equity=0, rev=0 -> ratios zero
    assert ratios[0]["roe"] == 0.0
    assert ratios[0]["debt_equity"] == 0.0
    assert ratios[0]["net_profit_margin"] == 0.0

    # Second row: equity>0, rev>0 -> valid math
    assert ratios[1]["roe"] == 0.0  # netIncome=0
    assert ratios[1]["debt_equity"] == 25.0 / 50.0
    assert ratios[1]["net_profit_margin"] == 0.0

    # Filter by date keeps both (inclusive)
    filt = rc.filter_by_date(ratios, "2025-03-31", "2025-06-30")
    assert [r["fiscalDateEnding"] for r in filt] == ["2025-06-30", "2025-03-31"]


# -----------------------------------------------------------------------------
# Unit tests: Training (small, fast checks)
# -----------------------------------------------------------------------------

def test_train_fnn_and_rf_return_reasonable_models():
    """We expect models to be constructed and produce sane validation accuracies."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = (rng.random(100) > 0.5).astype(int)

    trainer = pa.ModelTrain()
    fnn_model, fnn_acc = trainer.train_fnn(X, y, epochs=3, lr=0.01)
    rf_model, rf_acc = trainer.train_rf(X, y)

    assert isinstance(fnn_model, pa.ModelTrain.FeedForwardNN)
    assert 0.0 <= fnn_acc <= 1.0
    assert hasattr(rf_model, "predict_proba")
    assert 0.0 <= rf_acc <= 1.0


def test_compare_models_reports_best():
    """Quick check that compare_models returns the correct best_model key."""
    trainer = pa.ModelTrain()
    summary = trainer.compare_models(acc_fnn=0.6, acc_rf=0.55, acc_anfis=None)
    assert summary["best_model"] == "FNN"
    summary2 = trainer.compare_models(acc_fnn=0.4, acc_rf=0.7, acc_anfis=0.65)
    assert summary2["best_model"] == "RF"


def test_train_anfis_when_available(monkeypatch):
    """
    If ANFISNet is available, ModelTrain should use it.
    We fake a tiny ANFISNet to avoid external dependency.
    """
    class FakeANFIS:
        def __init__(self, input_dim):
            self.input_dim = input_dim
        def fit(self, X, y):  # no-op
            return self
        def predict(self, X):
            # Return 0.6 for all rows to simulate "mostly positive"
            return np.full(shape=(len(X),), fill_value=0.6, dtype=float)

    monkeypatch.setattr(pa, "ANFISNet", FakeANFIS)

    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3)).astype(np.float32)
    y = (rng.random(60) > 0.5).astype(int)

    trainer = pa.ModelTrain()
    model, acc = trainer.train_anfis(X, y)
    assert isinstance(model, FakeANFIS)
    assert 0.0 <= acc <= 1.0


# -----------------------------------------------------------------------------
# Unit tests: Inference agent
# -----------------------------------------------------------------------------

def test_model_infer_agent_end_to_end(tmp_path):
    """
    Create a deterministic FNN checkpoint (all-zero weights => sigmoid(0)=0.5),
    and a small RF. Then run a single inference call.
    """
    # Build and zero the FNN weights
    model = pa.ModelTrain.FeedForwardNN(input_dim=3, hidden_dim=4)
    with torch.no_grad():
        sd = model.state_dict()
        for k in sd:
            sd[k].zero_()
        model.load_state_dict(sd)

    ckpt = {"state_dict": model.state_dict(), "meta": {"input_dim": 3, "hidden_dim": 4}}
    fnn_path = tmp_path / "fnn_model.pt"
    torch.save(ckpt, fnn_path)

    # Tiny RF that can do predict_proba
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    y = (rng.random(50) > 0.5).astype(int)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, random_state=1)
    rf.fit(X, y)
    rf_path = tmp_path / "rf_model.pkl"
    import joblib as _joblib
    _joblib.dump(rf, rf_path)

    agent = pa.ModelInferAgent(str(fnn_path), str(rf_path), None)
    out = agent.infer([0.1, 0.2, 0.3])

    assert set(out.keys()) == {"fnn_prob", "rf_prob", "anfis_prob"}
    assert out["anfis_prob"] is None
    assert 0.0 <= out["fnn_prob"] <= 1.0
    assert 0.0 <= out["rf_prob"] <= 1.0


# -----------------------------------------------------------------------------
# Unit tests: Crew AI decision agent
# -----------------------------------------------------------------------------

def test_crew_ai_decision_agent_parses_json(monkeypatch):
    """
    When the model returns strict JSON, we parse it directly.
    """
    def fake_create(model, messages, temperature):
        # Optional: inspect 'messages' for threshold text
        assert any("threshold" in m.get("content", "").lower() for m in messages if m["role"] == "user")
        content = json.dumps({"decision": "BUY", "explanation": "2 of 3 above threshold"})
        return FakeOpenAIResponse(content)

    monkeypatch.setattr(pa.openai.ChatCompletion, "create", staticmethod(fake_create))

    agent = pa.CrewAIDecisionAgent(model_name="gpt-4o", threshold=0.5)
    res = agent.decide(0.6, 0.7, 0.2)

    assert res["decision"] == "BUY"
    assert "explanation" in res


def test_crew_ai_decision_agent_fallback_when_non_json(monkeypatch):
    """
    If the model returns non-JSON text, the agent should wrap it as 'decision'
    and use an empty explanation (as per implementation).
    """
    def fake_create(model, messages, temperature):
        # Return raw text instead of JSON
        return FakeOpenAIResponse("BUY")

    monkeypatch.setattr(pa.openai.ChatCompletion, "create", staticmethod(fake_create))

    agent = pa.CrewAIDecisionAgent(model_name="gpt-4o", threshold=0.5)
    res = agent.decide(0.9, 0.2, None)

    assert res["decision"] == "BUY"
    assert res["explanation"] == ""


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------

def test_main_end_to_end_with_mocks(monkeypatch, tmp_path, capsys):
    """
    Full run with valid date range (rows present).
    - Mocks input() for ticker and dates
    - Mocks yfinance and OpenAI
    - Ensures files are written and decision is printed
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Mock user input: ticker + date range
        inputs = iter(["AAPL", "2024-01-01", "2025-12-31"])
        monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

        # Patch yfinance.Ticker used in the module
        monkeypatch.setattr(pa.yf, "Ticker", lambda _: FakeTicker())

        # Patch OpenAI legacy ChatCompletion.create
        def fake_create(model, messages, temperature):
            # Provide a strict JSON response
            content = json.dumps({"decision": "HOLD", "explanation": "Mixed signals; threshold logic"})
            return FakeOpenAIResponse(content)

        monkeypatch.setattr(pa.openai.ChatCompletion, "create", staticmethod(fake_create))

        # Execute main()
        pa.main()

        # Verify stdout carries the final decision banner
        out = capsys.readouterr().out
        assert "===== FINAL DECISION =====" in out
        assert "Decision:" in out
        # Explanation should NOT be printed (line is commented in the script)
        assert "Explanation:" not in out

        # Verify model artifacts exist
        assert (tmp_path / "fnn_model.pt").exists()
        assert (tmp_path / "rf_model.pkl").exists()
        # anfis_model.pkl is optional; do not assert
    finally:
        os.chdir(cwd)


def test_main_with_empty_range_falls_back_to_recent(monkeypatch, tmp_path, capsys):
    """
    Uses a date range with no matching rows to ensure the code
    silently falls back to the most recent quarters and still completes.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Pick a very old range that won't match provided fake data
        inputs = iter(["AAPL", "2010-01-01", "2010-12-31"])
        monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

        monkeypatch.setattr(pa.yf, "Ticker", lambda _: FakeTicker())

        # Reply BUY this time to ensure different path
        def fake_create(model, messages, temperature):
            content = json.dumps({"decision": "BUY", "explanation": "2 of 3 above threshold"})
            return FakeOpenAIResponse(content)

        monkeypatch.setattr(pa.openai.ChatCompletion, "create", staticmethod(fake_create))

        pa.main()

        out = capsys.readouterr().out
        assert "===== FINAL DECISION =====" in out
        assert "Decision:" in out
        assert "Explanation:" not in out  # still commented in script

        # Artifacts created?
        assert (tmp_path / "fnn_model.pt").exists()
        assert (tmp_path / "rf_model.pkl").exists()
    finally:
        os.chdir(cwd)


def test_main_invalid_date_exits(monkeypatch, capsys):
    """
    If the user enters invalid date format, main() should print an error and exit.
    """
    inputs = iter(["AAPL", "2025/01/01", "2025-12-31"])  # invalid start date format
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    with pytest.raises(SystemExit) as ex:
        pa.main()

    assert ex.value.code == 1
    out = capsys.readouterr().out
    assert "[ERROR] Invalid date format:" in out


def test_main_empty_ticker_exits(monkeypatch, capsys):
    """
    If the user provides an empty ticker, main() should print an error and exit.
    """
    inputs = iter(["", "2025-01-01", "2025-12-31"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    with pytest.raises(SystemExit) as ex:
        pa.main()

    assert ex.value.code == 1
    out = capsys.readouterr().out
    assert "[ERROR] Empty ticker." in out


def test_main_yfinance_failure_exits(monkeypatch, capsys):
    """
    If yfinance raises during fetch, main() should print a clear error and exit(1).
    """
    # Valid inputs so we reach the fetch step
    inputs = iter(["AAPL", "2025-01-01", "2025-12-31"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    class BoomTicker:
        def __init__(self, *a, **kw):
            raise RuntimeError("network down")

    monkeypatch.setattr(pa.yf, "Ticker", BoomTicker)

    with pytest.raises(SystemExit) as ex:
        pa.main()

    assert ex.value.code == 1
    out = capsys.readouterr().out
    assert "[ERROR] Failed to fetch financial data for AAPL:" in out
