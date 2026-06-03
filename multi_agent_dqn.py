import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class MultiHeadQNetwork(nn.Module):
    """多头Q网络: 共享特征提取 + 独立动作头"""

    def __init__(self, state_dim, num_heads, action_per_head, hidden_dim=256):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_per_head) for _ in range(num_heads)
        ])

    def forward(self, x):
        features = self.shared_encoder(x)
        return [head(features) for head in self.heads]


class SharedReplayBuffer:
    """共享经验回放缓冲区"""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, joint_state, actions, reward, next_joint_state, done):
        self.buffer.append((joint_state, actions, reward, next_joint_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class CentralizedMADQN:
    """集中式多智能体Double DQN

    特性:
    - 共享经验池
    - 多头Q网络 (共享编码器 + 独立动作头)
    - Double DQN: policy_net选动作, target_net估值, 减少过估计
    - 同步更新策略网络和目标网络
    """

    def __init__(self, joint_state_dim=20, num_servers=3, action_per_server=6,
                 hidden_dim=256, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=800,
                 buffer_size=50000, batch_size=128, target_update_freq=300,
                 soft_update_tau=0.005, tau_start=0.01, tau_end=0.001,
                 tau_decay_steps=50000):
        self.num_servers = num_servers
        self.action_per_server = action_per_server
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.soft_update_tau = soft_update_tau
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_decay_steps = tau_decay_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.learn_steps = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = MultiHeadQNetwork(
            joint_state_dim, num_servers, action_per_server, hidden_dim
        ).to(self.device)
        self.target_net = MultiHeadQNetwork(
            joint_state_dim, num_servers, action_per_server, hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.shared_buffer = SharedReplayBuffer(buffer_size)

    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps_done / self.epsilon_decay)

    def select_actions(self, joint_state, training=True):
        if training and random.random() < self.get_epsilon():
            return [random.randrange(self.action_per_server) for _ in range(self.num_servers)]
        with torch.no_grad():
            state_t = torch.FloatTensor(joint_state).unsqueeze(0).to(self.device)
            q_heads = self.policy_net(state_t)
            actions = [q.argmax(dim=1).item() for q in q_heads]
        return actions

    def store_transition(self, joint_state, actions, reward, next_joint_state, done):
        self.shared_buffer.push(joint_state, actions, reward, next_joint_state, done)

    def update(self):
        if len(self.shared_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.shared_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_heads = self.policy_net(states_t)

        # Double DQN: policy_net选择最优动作, target_net评估其Q值
        with torch.no_grad():
            next_q_policy_heads = self.policy_net(next_states_t)
            next_q_target_heads = self.target_net(next_states_t)

        total_loss = torch.tensor(0.0, device=self.device)
        for i in range(self.num_servers):
            q_values = q_heads[i].gather(1, actions_t[:, i].unsqueeze(1)).squeeze(1)
            # Double DQN: argmax来自policy, 值来自target
            best_next_actions = next_q_policy_heads[i].argmax(dim=1, keepdim=True)
            next_q_value = next_q_target_heads[i].gather(1, best_next_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q_value * (1 - dones_t)
            total_loss += nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 同步软更新目标网络 (每步都微小更新, 减少训练波动)
        self.learn_steps += 1
        self._sync_target_network()

        return total_loss.item() / self.num_servers

    def _get_adaptive_tau(self):
        """根据训练进度动态调整Polyak参数: 初期较快更新, 后期较慢更新"""
        progress = min(1.0, self.learn_steps / max(self.tau_decay_steps, 1))
        return self.tau_start + (self.tau_end - self.tau_start) * progress

    def _sync_target_network(self):
        """自适应软更新: tau随训练进度从tau_start衰减到tau_end"""
        tau = self._get_adaptive_tau()
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "learn_steps": self.learn_steps,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        self.learn_steps = checkpoint["learn_steps"]


class IndependentDQN:
    """独立Double DQN: 每个服务器独立网络和缓冲区, 同步目标网络更新"""

    def __init__(self, num_servers=3, local_state_dim=14, action_dim=6,
                 hidden_dim=128, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=800,
                 buffer_size=10000, batch_size=64, target_update_freq=200,
                 soft_update_tau=0.005):
        self.num_servers = num_servers
        self.soft_update_tau = soft_update_tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agents = []
        for _ in range(num_servers):
            agent = SingleAgentDQN(
                local_state_dim, action_dim, hidden_dim,
                lr, gamma, epsilon_start, epsilon_end, epsilon_decay,
                buffer_size, batch_size, target_update_freq,
                soft_update_tau, self.device
            )
            self.agents.append(agent)

        self.steps_done = 0

    def select_actions(self, local_states, training=True):
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.select_action(local_states[i], training))
        return actions

    def store_transitions(self, local_states, actions, rewards_per_server, next_local_states, done):
        for i, agent in enumerate(self.agents):
            agent.buffer.append((local_states[i], actions[i], rewards_per_server[i],
                                 next_local_states[i], done))

    def update(self):
        losses = []
        for agent in self.agents:
            loss = agent.update()
            losses.append(loss)
        return losses

    def sync_all_targets(self):
        """同步所有智能体的目标网络 (在同一时刻更新, 减少训练波动)"""
        for agent in self.agents:
            agent.sync_target()

    def save(self, path):
        data = {}
        for i, agent in enumerate(self.agents):
            data[f"agent_{i}_policy"] = agent.policy_net.state_dict()
            data[f"agent_{i}_target"] = agent.target_net.state_dict()
        torch.save(data, path)


class SingleAgentDQN:
    """单个独立Double DQN Agent (自适应Polyak tau)"""

    def __init__(self, state_dim, action_dim, hidden_dim,
                 lr, gamma, epsilon_start, epsilon_end, epsilon_decay,
                 buffer_size, batch_size, target_update_freq,
                 soft_update_tau, device,
                 tau_start=0.01, tau_end=0.001, tau_decay_steps=30000):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.soft_update_tau = soft_update_tau
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_decay_steps = tau_decay_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.learn_steps = 0
        self.device = device

        self.policy_net = self._build_net(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = self._build_net(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)

    def _build_net(self, state_dim, action_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps_done / self.epsilon_decay)

    def select_action(self, state, training=True):
        self.steps_done += 1
        if training and random.random() < self.get_epsilon():
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_t).argmax(dim=1).item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: policy_net选动作, target_net评估
        with torch.no_grad():
            best_next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q_value = self.target_net(next_states_t).gather(1, best_next_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q_value * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 软更新目标网络
        self.learn_steps += 1
        self.sync_target()

        return loss.item()

    def _get_adaptive_tau(self):
        """根据训练进度动态调整tau: 初期快速跟踪, 后期稳定"""
        progress = min(1.0, self.learn_steps / max(self.tau_decay_steps, 1))
        return self.tau_start + (self.tau_end - self.tau_start) * progress

    def sync_target(self):
        """自适应软更新目标网络"""
        tau = self._get_adaptive_tau()
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )
