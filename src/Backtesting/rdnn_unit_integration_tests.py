# tests/test_trading.py

import pytest
import numpy as np
import pandas as pd
import torch
import json
import time
from types import SimpleNamespace

import trading
from trading import TradingEnv, fetch_ohlcv, RNNPolicyNetwork, CrewAIDecisionAgent, Inference, RecurrentPPO

class DummyDF(pd.DataFrame):
    @property
    def _constructor(self):
        return DummyDF

# -----------------------
# Unit Tests
# -----------------------

def test_get_observation_and_reset():
    data = np.arange(50, dtype=np.float32).reshape(10, 5)
    env = TradingEnv(data, window_size=5)
    obs = env.reset()
    assert obs.shape == (5, 5)
    np.testing.assert_array_equal(obs, data[0:5, :])

def test_step_and_cost():
    data = np.zeros((7,5), dtype=np.float32)
    data[:,3] = np.arange(7)  # close price
    env = TradingEnv(data, window_size=2, transaction_cost=0.0, slippage=0.0)
    env.reset()
    # Step 1: BUY at price 1
    obs, reward1, done1, info1 = env.step(2)
    assert env.position == 1
    assert pytest.approx(env.entry_price) == 1.0
    # Step 2: HOLD (action=1) at price 2: reward = pnl = 2-1 = 1
    obs, reward2, done2, info2 = env.step(1)
    assert pytest.approx(reward2) == 1.0
    assert info2["net_worth"] == pytest.approx(1.0 + 1.0)

def test_cost_computation():
    env = TradingEnv(np.zeros((10,5)), transaction_cost=0.01, slippage=0.005)
    price = 100.0
    expected = price * (0.01 + 0.005)
    assert env._cost(price) == pytest.approx(expected)

def test_fetch_ohlcv_success(monkeypatch):
    df = DummyDF({
        "Open": [1,2],
        "High": [1,2],
        "Low": [1,2],
        "Close": [1,2],
        "Volume": [10,20]
    }, index=pd.date_range("2025-01-01", periods=2))
    monkeypatch.setattr(trading.yf, "download", lambda *args, **kwargs: df)
    arr = fetch_ohlcv("FAKE", "2025-01-01", "2025-01-03")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,5)
    np.testing.assert_array_equal(arr, df.values)

def test_fetch_ohlcv_failure(monkeypatch):
    monkeypatch.setattr(trading.yf, "download", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("download error")))
    with pytest.raises(ValueError):
        fetch_ohlcv("FAIL", "2025-01-01", "2025-01-02")

def test_rnn_policy_forward_shapes():
    batch_size, seq_len, feat_dim, action_dim = 3, 7, 5, 3
    net = RNNPolicyNetwork(input_size=feat_dim, hidden_size=16, num_layers=1, action_dim=action_dim)
    x = torch.randn(batch_size, seq_len, feat_dim)
    q, logits, v, hid = net(x)
    assert q.shape == (batch_size, action_dim)
    assert logits.shape == (batch_size, action_dim)
    assert v.shape == (batch_size, 1)
    assert isinstance(hid, tuple) and len(hid) == 2

def test_build_prompt_and_default_decide():
    agent = CrewAIDecisionAgent()
    # Test prompt formatting
    probs = [0.12, 0.34, 0.54]
    prompt = agent._build_prompt("TICK", probs)
    assert "For the stock 'TICK'" in prompt
    assert "SELL: 0.12" in prompt and "BUY: 0.54" in prompt
    # When LLM unavailable, decide should return HOLD
    agent.llm_available = False
    out = agent.decide("TICK", json.dumps({"action_probs": probs}))
    assert out["recommendation"] == "HOLD"

# -----------------------
# Integration Tests
# -----------------------

def test_env_full_episode():
    steps, window = 15, 3
    data = np.zeros((window + steps, 5), dtype=np.float32)
    data[:,3] = np.linspace(1, 1+steps, window + steps)
    env = TradingEnv(data, window_size=window, transaction_cost=0.0, slippage=0.0)
    obs = env.reset()
    total_reward, count = 0.0, 0
    done = False
    while not done:
        obs, r, done, info = env.step(2)
        total_reward += r
        count += 1
    assert count == steps
    assert info["net_worth"] >= 1.0

def test_crew_agent_decide_monkeypatched(monkeypatch):
    # Fake ChatCompletion.create
    fake_resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"recommendation":"SELL"}'))])
    class DummyChat:
        @staticmethod
        def create(model, messages):
            return fake_resp
    monkeypatch.setattr(trading.openai, "api_key", "dummy")
    monkeypatch.setattr(trading.openai.ChatCompletion, "create", DummyChat.create)
    agent = CrewAIDecisionAgent()
    agent.llm_available = True
    out = agent.decide("ABC", json.dumps({"action_probs":[0.1,0.2,0.7]}))
    assert out["stock"] == "ABC"
    assert out["recommendation"] == "SELL"

def test_inference_infer_and_benchmark(monkeypatch):
    # Prepare fake model
    def fake_predict(obs_batch, state, episode_start):
        return None, np.array([False])
    class FakePolicy:
        def get_distribution(self, obs, hidden, ep_start):
            class Dist:
                def __init__(self):
                    self.distribution = SimpleNamespace(probs=torch.tensor([[0.2,0.3,0.5]]))
            return Dist(), np.array([False])
    fake_model = SimpleNamespace(predict=fake_predict, policy=FakePolicy())
    monkeypatch.setattr(RecurrentPPO, "load", lambda path: fake_model)

    inf = Inference("dummy_path")
    window = np.zeros((10,5), dtype=np.float32)
    # Test infer
    result_json = inf.infer(window)
    result = json.loads(result_json)
    assert "action_probs" in result
    assert result["action_probs"] == [0.2, 0.3, 0.5]
    # Test benchmark
    latency = inf.benchmark(window, n_runs=10)
    assert isinstance(latency, float)
    assert latency >= 0.0
