import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from numpy._core.multiarray import dtype

class TradingEnv(gym.Env):
    """
    state 요소
    1. 현재 잔고(balance)
    2. 현재 보유 주식 수(shares)
    3. 현재 포트폴리오 가치(total_value)
    """
    def __init__(
        self,
        df,
        min_initial_balance: float = 10_000.0, 
        max_initial_balance: float = 100_000.0,
        trading_unit: int = 10):

        super(TradingEnv, self).__init__()
        
        self.min_initial_balance = min_initial_balance
        self.max_initial_balance = max_initial_balance
        self.trading_unit = trading_unit

        # state/action 정의

        self.observation_space = spaces.Box( # balance(잔고), shares(보유 주식 수), portfolio_value(현재 포트폴리오 가치)  Open(시가), High(당일 최고가), Low(당일 최저가), Close(종가), Volume(거래량)
            low = np.array([0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype = np.float32),
            high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype = np.float32),
            shape = (8, ),
            dtype = np.float32
        )

        self.prices = df[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype = np.float32)
        self.n_steps = len(self.prices) # step 수

        #0=sell, 1=hold, 2=buy
        self.action_space = spaces.Discrete(3)

        self.current_step = None #현재 step 저장
        self.balance = None # 현재 잔고 저장
        self.shares = None # 현재 보유 주식 수 저장
        self.portfolio_value = None #현재 포트폴리오 가치 저장


    def get_price(self):
        """
        거래는 항상 종가로 하니까 종가 출력하는 함수
        """
        return float(self.prices[self.current_step][3])

    def get_observation(self):
        """
        현재 상태의 잔고, 보유 주식 수, 포트폴리오 가치, 시가, 당일 최고가, 당일 최저가, 종가, 거래량 한번에 출력
        """
        price_vec = self.prices[self.current_step]
        obs = np.array([self.balance, self.shares, self.portfolio_value]+list(price_vec),
                       dtype = np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = np.random.uniform(self.min_initial_balance, self.max_initial_balance)
        self.shares = 0.0

        price = self.get_price()
        self.portfolio_value = price * self.shares + self.balance
        obs = self.get_observation()
        info = {}
        return obs, info

    def step(self, action, count):
        assert self.action_space.contains(action) #action이 0,1,2중 하나인지 확인

        current_price = self.get_price()

        prev_value = self.portfolio_value #이번스텝 시작할때 포트폴리오 가치

        if action == 0: # sell
            sell_amount = min(self.shares, self.trading_unit) #N주 매도, 보유량이 적으면 전액 매도
            if sell_amount > 0:
                self.shares -= sell_amount
                self.balance += current_price*sell_amount
        elif action == 1: # hold
            pass
        elif action == 2: # buy
            #N주 매수, 잔고 부족이면 매수 X
            cost = current_price * self.trading_unit
            if self.balance >= cost:
                self.shares += self.trading_unit
                self.balance -= cost
            else:
                pass #잔고 부족이면 매수 X  
        else:
            raise Exception("no valid action")

        self.current_step += 1 #step 증가
        terminated = self.current_step >= (self.n_steps - 1)
        truncated = False #내가 임의로 조기 종료시킬때

        if not terminated:
            next_price = self.get_price()
        else:
            next_price = current_price

        self.portfolio_value = self.balance + self.shares * next_price #포트폴리오 업데이트

        reward = (self.portfolio_value - prev_value) / max(prev_value, 1e-8) ## r_t = (V_t - V_{t-1}) / V_{t-1}

        obs = self.get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "shares": self.shares,
            "action": action
        }
        return obs, reward, terminated, truncated, info