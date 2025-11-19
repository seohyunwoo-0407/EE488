import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Common Networks ---
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_net = MLP(state_dim, action_dim)
        self.target_net = MLP(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.train_step = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Q(s, a)
        q_values = self.q_net(states).gather(1, actions)

        # Target: r + gamma * max Q(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target Network Update & Epsilon Decay
        self.train_step += 1
        if self.train_step % 100 == 0: # Update target every 100 steps
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- A2C Agent ---
class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=7e-4, gamma=0.99):
        self.gamma = gamma
        
        # Actor: State -> Action Probabilities
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: State -> Value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def select_action(self, state, training=True):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)
        if training:
            action = dist.sample()
        else:
            action = probs.argmax()
        return action.item()

    def update(self, transitions):
        # A2C는 보통 한 에피소드나 n-step 마다 업데이트하지만, 
        # 여기서는 간단하게 에피소드 단위(MC) 혹은 배치 단위로 처리 가능.
        # 아래는 Transition 리스트(trajectory)를 받아 업데이트하는 방식입니다.
        
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Calculate Target (TD Target)
        with torch.no_grad():
            next_values = self.critic(next_states)
            td_targets = rewards + self.gamma * next_values * (1 - dones)
        
        # Critic Loss
        values = self.critic(states)
        advantage = td_targets - values
        critic_loss = advantage.pow(2).mean()

        # Actor Loss
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions.squeeze())
        
        # Advantage를 detach하여 actor update 시 critic에 영향 안 주게 함
        actor_loss = -(log_probs * advantage.detach().squeeze()).mean()

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()