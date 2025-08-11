import gym
from gym import spaces
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import time
import json

# RDNN.3 imports for recurrent PPO training and evaluation
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# RDNN.6 import for Crew AI (OpenAI)
import openai  # ensure OPENAI_API_KEY is set in environment

class TradingEnv(gym.Env):
    """
    Gym environment for trading based on OHLCV data.
    - Observation: window of last T bars (Open, High, Low, Close, Volume).
    - Action space: 0=SELL, 1=HOLD, 2=BUY.
    - Reward: Profit/Loss minus transaction cost and slippage.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, data, window_size=10, transaction_cost=0.001, slippage=0.001):
        super().__init__()
        self.data = np.array(data, dtype=np.float32)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        # Discrete action space: SELL, HOLD, BUY
        self.action_space = spaces.Discrete(3)
        # Observation: window_size x features
        obs_shape = (window_size, self.data.shape[1])
        self.observation_space = spaces.Box(-np.inf, np.inf, obs_shape, dtype=np.float32)
        self.reset()

    def reset(self):
        """Reset environment state and return initial observation."""
        self.current_step = self.window_size
        self.position = 0           # current position: -1=short, 0=flat, +1=long
        self.entry_price = 0.0      # price at which position was entered
        self.net_worth = 1.0        # starting net worth (normalized)
        return self._get_observation()

    def _get_observation(self):
        """Get the last `window_size` rows as observation."""
        return self.data[self.current_step - self.window_size : self.current_step]

    def step(self, action):
        """
        Execute one time step within the environment.
        Returns: observation, reward, done, info
        """
        done = False
        price = self.data[self.current_step, 3]  # Close price
        reward = 0.0

        # Enter short position
        if action == 0 and self.position != -1:
            reward -= self._cost(price)
            self.position = -1
            self.entry_price = price
        # Enter long position
        elif action == 2 and self.position != 1:
            reward -= self._cost(price)
            self.position = 1
            self.entry_price = price
        # HOLD does nothing

        # Calculate PnL and update net worth
        pnl = self.position * (price - self.entry_price)
        reward += pnl
        self.net_worth += pnl - self._cost(price)

        # Advance to next step
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True

        # Next observation or zero array if done
        obs = self._get_observation() if not done else np.zeros_like(self._get_observation())
        return obs, reward, done, {"net_worth": self.net_worth}

    def _cost(self, price):
        """Compute transaction cost + slippage for a trade at given price."""
        return price * (self.transaction_cost + self.slippage)

    def render(self, mode="human"):
        """Print the current step, position, and net worth."""
        print(f"Step: {self.current_step}, Position: {self.position}, Net worth: {self.net_worth:.3f}")


def fetch_ohlcv(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> np.ndarray:
    """
    Fetch OHLCV data for `ticker` between `start_date` and `end_date`.
    Returns a NumPy array of shape (n_rows, 5).
    """
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    df = df.dropna()[["Open", "High", "Low", "Close", "Volume"]]
    return df.values


class RNNPolicyNetwork(nn.Module):
    """
    LSTM-based encoder producing Q-values, policy logits, and value estimates.
    """
    def __init__(self, input_size, hidden_size, num_layers, action_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.q_head = nn.Linear(hidden_size, action_dim)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        lstm_out, hidden_out = self.lstm(x, hidden)
        last = lstm_out[:, -1, :]
        return (
            self.q_head(last),        # DQN Q-values
            self.policy_head(last),   # PPO policy logits
            self.value_head(last),    # value estimate
            hidden_out                # LSTM hidden state
        )


def train_recurrent_ppo(env, total_timesteps=10000, log_dir='./logs/'):
    """
    Train a recurrent PPO agent with tensorboard logging and evaluation callback.
    """
    model = RecurrentPPO(
        policy='MlpLstmPolicy',
        env=env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=256,
        tensorboard_log=log_dir,
    )
    eval_cb = EvalCallback(
        env,
        best_model_save_path='./best_model/rdnn3/',
        log_path=log_dir,
        eval_freq=100_000,
        n_eval_episodes=1
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name='rdnn3', callback=eval_cb)
    model.save('rdnn3_recurrentppo')
    return model


def train_recurrent_ppo_with_validation(train_env, eval_env, total_timesteps=50000, log_dir='./logs/rdnn4/'):
    """
    Train with validation and early stopping on no improvement.
    """
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=5,
        verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/rdnn4/',
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        callback_after_eval=stop_cb
    )
    model = RecurrentPPO(
        policy='MlpLstmPolicy',
        env=train_env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=256,
        tensorboard_log=log_dir,
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name='rdnn4', callback=eval_cb)
    model.save('rdnn4_recurrentppo')
    return model


class Inference:
    """
    Load a trained RecurrentPPO model and compute action probabilities.
    """
    def __init__(self, model_path):
        self.model = RecurrentPPO.load(model_path)
        self.hidden_state = None

    def infer(self, window: np.ndarray) -> str:
        """
        Return JSON string with action_probs for a single window.
        """
        episode_starts_np = np.array([self.hidden_state is None], dtype=bool)
        obs_batch = window[np.newaxis, ...]
        _, self.hidden_state = self.model.predict(
            obs_batch,
            state=self.hidden_state,
            episode_start=episode_starts_np
        )
        obs_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        ep_tensor = torch.tensor(episode_starts_np, dtype=torch.float32, device=obs_tensor.device)
        dist, new_states = self.model.policy.get_distribution(
            obs_tensor,
            self.hidden_state,
            ep_tensor
        )
        self.hidden_state = new_states
        probs = dist.distribution.probs.detach().cpu().numpy()[0].tolist()
        return json.dumps({"action_probs": probs})

    def benchmark(self, window: np.ndarray, n_runs: int = 100) -> float:
        """
        Measure average inference latency over `n_runs`.
        """
        obs_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        # warm-up
        _ = self.model.policy.get_distribution(
            obs_tensor,
            self.hidden_state or np.array([True], dtype=bool),
            torch.tensor([0.0], dtype=torch.float32, device=obs_tensor.device)
        )
        start = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = self.model.policy.get_distribution(
                    obs_tensor,
                    self.hidden_state,
                    torch.tensor([0.0], dtype=torch.float32, device=obs_tensor.device)
                )
        latency = (time.time() - start) / n_runs
        print(f"Average inference latency: {latency*1000:.3f} ms")
        return latency


# ---------------------------------------------------------------------------
# RDNN.6 Crew AI DecisionAgent #680
# RDNN.6.1 Map action_probs to BUY/SELL/HOLD via ChatGPT
# ---------------------------------------------------------------------------
class CrewAIDecisionAgent:
    def __init__(self):
        # LLM availability check (requires openai.api_key)
        self.llm_available = bool(openai.api_key)
    
    def _build_prompt(self, ticker: str, probs: list) -> str:
        """
        Construct prompt for LLM to choose BUY/SELL/HOLD based on probabilities.
        """
        return (
            f"You are a trading decision assistant. "
            f"For the stock '{ticker}', you receive the following probabilities:\n"
            f"SELL: {probs[0]:.2f}, HOLD: {probs[1]:.2f}, BUY: {probs[2]:.2f}.\n\n"
            f"Based on these, decide whether to BUY, SELL, or HOLD. "
            f"Respond exactly with a JSON: {{\"recommendation\": \"BUY\"}} "
            f"or {{\"recommendation\": \"SELL\"}} or {{\"recommendation\": \"HOLD\"}}."
        )

    def decide(self, ticker: str, action_probs_json: str) -> dict:
        """
        Return a dict with the LLM's trading recommendation.
        Exceptions are no longer caught—any API error will propagate.
        """
        data = json.loads(action_probs_json)
        probs = data.get("action_probs", [])
        recommendation = "HOLD"

        if self.llm_available:
            prompt = self._build_prompt(ticker, probs)
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            cand = parsed.get("recommendation", "").upper()
            if cand in {"BUY", "SELL", "HOLD"}:
                recommendation = cand

        return {"stock": ticker, "recommendation": recommendation}


if __name__ == '__main__':
    # User inputs
    ticker = input('Enter stock ticker symbol (e.g., AAPL): ').strip().upper()
    start_date = input('Enter start date (YYYY-MM-DD): ').strip()
    end_date = input('Enter end date (YYYY-MM-DD): ').strip()

    # Fetch data
    try:
        data_array = fetch_ohlcv(ticker, start_date, end_date)
        print(f'Fetched {data_array.shape[0]} rows of OHLCV data for {ticker}.')
    except Exception as e:
        print(f'Error fetching data for {ticker}: {e}')
        exit(1)

    # Prepare environments
    window_size = 10
    split_idx = int(len(data_array) * 0.8)
    train_data = data_array[:split_idx]
    test_data  = data_array[split_idx - window_size:]
    train_env  = TradingEnv(train_data, window_size=window_size)
    eval_env   = TradingEnv(test_data,  window_size=window_size)

    # Train model with early stopping
    model = train_recurrent_ppo_with_validation(train_env, eval_env)

    # Inference on sample window
    inference = Inference('./best_model/rdnn4/best_model.zip')
    sample_window = test_data[:window_size]
    action_probs_json = inference.infer(sample_window)
    print(f'Action probabilities: {action_probs_json}')
    inference.benchmark(sample_window, n_runs=500)

    # Crew AI decision
    crew_agent = CrewAIDecisionAgent()
    decision = crew_agent.decide(ticker, action_probs_json)
    print(f'Crew AI recommendation: {json.dumps(decision)}')
