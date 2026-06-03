import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """深度Q网络"""

    def __init__(self, state_dim=4, action_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN智能体"""

    def __init__(self, state_dim=4, action_dim=3, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=500,
                 buffer_size=10000, batch_size=64, target_update_freq=200):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # epsilon贪心参数
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.learn_steps = 0  # 训练步数计数器

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 策略网络和目标网络
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def get_epsilon(self):
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
              np.exp(-self.steps_done / self.epsilon_decay)
        return eps

    def select_action(self, state, training=True):
        if training and random.random() < self.get_epsilon():
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 当前Q值
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: policy_net选动作, target_net估值, 减少过估计
        with torch.no_grad():
            best_next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states_t).gather(1, best_next_actions).squeeze(1)
            target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 按训练步数更新目标网络
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.update_target()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "learn_steps": self.learn_steps
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        self.learn_steps = checkpoint.get("learn_steps", 0)
