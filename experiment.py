import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from env_custom import TradingEnv
from algorithms import DQNAgent, A2CAgent

# --- Configuration ---
TRAIN_START = "2017-01-01"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2024-10-25"

TRAIN_TICKER = "AAPL"
TEST_TICKERS = ["AAPL", "MSFT", "GOOG", "TSLA"]

EPISODES_TRAIN = 20  # 시간 관계상 줄임 (실제로는 50~100 이상 권장)
EPISODES_TEST = 10
TRADING_UNIT = 10

def download_data(ticker, start, end):
    print(f"Downloading {ticker} ({start}~{end})...")
    df = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=False)
    if df.empty: raise ValueError(f"No data for {ticker}")
    return df

def run_experiment():
    # 1. Data Load
    train_df = download_data(TRAIN_TICKER, TRAIN_START, TRAIN_END)
    test_dfs = {t: download_data(t, TEST_START, TEST_END) for t in TEST_TICKERS}

    results = {"DQN": {}, "A2C": {}}

    # 2. Run for each Algorithm
    for algo_name in ["DQN", "A2C"]:
        print(f"\n=== [Algorithm: {algo_name}] Training Start ===")
        
        # Init Env & Agent
        env = TradingEnv(train_df, trading_unit=TRADING_UNIT)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        if algo_name == "DQN":
            agent = DQNAgent(state_dim, action_dim)
        else:
            agent = A2CAgent(state_dim, action_dim)

        # Training Loop
        for ep in range(EPISODES_TRAIN):
            state, _ = env.reset()
            done = False
            trajectory = [] # For A2C
            total_reward = 0
            
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if algo_name == "DQN":
                    agent.store_transition(state, action, reward, next_state, done)
                    agent.update()
                else: # A2C
                    trajectory.append((state, action, reward, next_state, done))

                state = next_state
                total_reward += reward
            
            if algo_name == "A2C":
                agent.update(trajectory)
            
            if (ep+1) % 5 == 0:
                print(f"Episode {ep+1}/{EPISODES_TRAIN} | Total Reward: {total_reward:.4f}")

        # Testing Loop
        print(f"=== [Algorithm: {algo_name}] Testing Start ===")
        for ticker, df_test in test_dfs.items():
            final_values = []
            for _ in range(EPISODES_TEST):
                test_env = TradingEnv(df_test, trading_unit=TRADING_UNIT)
                state, _ = test_env.reset()
                done = False
                while not done:
                    action = agent.select_action(state, training=False) # Deterministic or Greedy
                    state, _, terminated, truncated, info = test_env.step(action)
                    done = terminated or truncated
                final_values.append(info['portfolio_value'])
            
            results[algo_name][ticker] = final_values
            print(f"Ticker: {ticker} | Mean PV: {np.mean(final_values):.2f}")

    return results

def plot_results(results):
    data = []
    for algo, tickers_res in results.items():
        for ticker, values in tickers_res.items():
            data.append({
                "Algorithm": algo,
                "Ticker": ticker,
                "Mean": np.mean(values),
                "Std": np.std(values)
            })
    
    df_plot = pd.DataFrame(data)
    
    # Mean-Variance Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x="Std", y="Mean", hue="Algorithm", style="Ticker", s=100)
    plt.title("Mean-Variance Analysis")
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Return (Mean Final Portfolio Value)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    res = run_experiment()
    plot_results(res)