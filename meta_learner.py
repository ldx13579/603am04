"""Reptile元学习模块: 快速适应新负载模式

算法:
1. 从预训练CentralizedMADQN初始化元模型 theta
2. 每次元迭代:
   a. 从TaskDistribution采样任务 T_i
   b. 克隆 theta -> theta_i
   c. 在 T_i 上运行K步内循环梯度更新
   d. 外循环更新: theta <- theta + meta_lr * (theta_i - theta)
3. 快速适应新模式:
   a. 克隆元模型
   b. 在新数据上微调N个episode (1小时=12步@5min间隔)
"""

import numpy as np
import copy
import torch
from collections import OrderedDict

from cluster_env import ClusterEnv
from multi_agent_dqn import CentralizedMADQN
from workload_trace import (
    AlibabaTraceGenerator,
    Double11TraceGenerator,
    BurstPatternGenerator,
)


class TaskDistribution:
    """元学习任务分布: 生成多样化的工作负载场景"""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.task_types = [
            "normal", "high_load", "double11",
            "burst", "weekend", "spike",
        ]

    def sample_task(self):
        """采样一个随机负载任务, 返回trace数组"""
        task_type = self.rng.choice(self.task_types)
        seed = int(self.rng.integers(0, 100000))

        if task_type == "normal":
            gen = AlibabaTraceGenerator(
                duration_hours=24, seed=seed,
                base_rate=self.rng.uniform(0.8, 1.5),
                peak_multiplier=self.rng.uniform(2.0, 4.0),
            )
            return gen.generate_trace()

        elif task_type == "high_load":
            gen = AlibabaTraceGenerator(
                duration_hours=24, seed=seed,
                base_rate=self.rng.uniform(1.5, 3.0),
                peak_multiplier=self.rng.uniform(3.0, 5.0),
                burst_prob=0.1,
            )
            return gen.generate_trace()

        elif task_type == "double11":
            gen = Double11TraceGenerator(
                duration_hours=24, seed=seed,
                flash_multiplier=self.rng.uniform(10.0, 20.0),
                sustain_multiplier=self.rng.uniform(4.0, 8.0),
                flash_start_hour=self.rng.uniform(0.0, 4.0),
            )
            return gen.generate_trace()

        elif task_type == "burst":
            gen = BurstPatternGenerator(
                duration_hours=24, seed=seed,
                burst_interval_steps=int(self.rng.integers(20, 50)),
                burst_duration_steps=int(self.rng.integers(3, 8)),
                burst_magnitude=self.rng.uniform(5.0, 12.0),
            )
            return gen.generate_trace()

        elif task_type == "weekend":
            gen = AlibabaTraceGenerator(
                duration_hours=24, seed=seed,
                base_rate=self.rng.uniform(0.5, 0.8),
                peak_multiplier=self.rng.uniform(1.5, 2.5),
            )
            return gen.generate_trace()

        else:  # spike
            gen = AlibabaTraceGenerator(
                duration_hours=24, seed=seed,
                base_rate=self.rng.uniform(0.8, 1.2),
                burst_prob=0.15,
                burst_multiplier=self.rng.uniform(8.0, 15.0),
            )
            return gen.generate_trace()

    def sample_batch(self, n):
        """采样n个任务"""
        return [self.sample_task() for _ in range(n)]


class ReptileMetaLearner:
    """Reptile元学习器: 基于预训练MADDQN的快速适应框架

    特性:
    - 外循环: 跨多种负载模式学习通用初始化
    - 内循环: 在单一模式上快速微调
    - 快速适应: 仅需1小时(12步)数据即可适配新模式
    """

    def __init__(self, base_agent, meta_lr=0.1,
                 inner_episodes=5, num_tasks_per_iter=4,
                 energy_weight=0.5):
        self.meta_lr = meta_lr
        self.inner_episodes = inner_episodes
        self.num_tasks_per_iter = num_tasks_per_iter
        self.energy_weight = energy_weight

        self.device = base_agent.device
        self.joint_state_dim = base_agent.policy_net.shared_encoder[0].in_features
        self.num_servers = base_agent.num_servers
        self.action_per_server = base_agent.action_per_server

        self.meta_weights = copy.deepcopy(base_agent.policy_net.state_dict())
        self.meta_target_weights = copy.deepcopy(base_agent.target_net.state_dict())

        self.adaptation_history = []

    def _create_agent(self):
        """创建新的MADDQN agent实例并加载元权重"""
        agent = CentralizedMADQN(
            joint_state_dim=self.joint_state_dim,
            num_servers=self.num_servers,
            action_per_server=self.action_per_server,
            epsilon_start=0.3,
            epsilon_end=0.05,
            epsilon_decay=200,
        )
        agent.policy_net.load_state_dict(copy.deepcopy(self.meta_weights))
        agent.target_net.load_state_dict(copy.deepcopy(self.meta_target_weights))
        return agent

    def _train_on_task(self, agent, trace, num_episodes):
        """在单个任务上训练agent (内循环)"""
        env = ClusterEnv(trace=trace, energy_weight=self.energy_weight, seed=None)
        episode_rewards = []

        for _ in range(num_episodes):
            obs = env.reset()
            joint_state = obs["joint_state"]
            total_reward = 0
            done = False

            while not done:
                actions = agent.select_actions(joint_state, training=True)
                obs, rewards, done, info = env.step(actions)
                next_state = obs["joint_state"]

                agent.store_transition(
                    joint_state, actions, rewards["total"], next_state, done
                )
                agent.update()
                agent.steps_done += 1

                joint_state = next_state
                total_reward += rewards["total"]

            episode_rewards.append(total_reward)

        return episode_rewards

    def _reptile_outer_step(self, adapted_weights_list):
        """Reptile外循环更新: theta += meta_lr/n * sum(theta_i - theta)"""
        n = len(adapted_weights_list)
        if n == 0:
            return

        for key in self.meta_weights:
            direction = torch.zeros_like(self.meta_weights[key])
            for adapted_w in adapted_weights_list:
                direction += adapted_w[key] - self.meta_weights[key]
            direction /= n
            self.meta_weights[key] = self.meta_weights[key] + self.meta_lr * direction

    def meta_train(self, task_distribution, num_iterations=100,
                   verbose=True, log_interval=10):
        """元训练: 跨多种负载模式学习通用初始化

        Args:
            task_distribution: TaskDistribution实例
            num_iterations: 外循环迭代次数
            verbose: 是否打印进度
            log_interval: 日志间隔

        Returns:
            meta_rewards: 每次迭代的平均内循环奖励
        """
        meta_rewards = []

        for iteration in range(num_iterations):
            tasks = task_distribution.sample_batch(self.num_tasks_per_iter)
            adapted_weights_list = []
            iter_rewards = []

            for trace in tasks:
                agent = self._create_agent()
                rewards = self._train_on_task(agent, trace, self.inner_episodes)
                adapted_weights_list.append(
                    copy.deepcopy(agent.policy_net.state_dict())
                )
                iter_rewards.append(np.mean(rewards))

            self._reptile_outer_step(adapted_weights_list)
            avg_reward = np.mean(iter_rewards)
            meta_rewards.append(avg_reward)

            if verbose and (iteration + 1) % log_interval == 0:
                recent_avg = np.mean(meta_rewards[-log_interval:])
                print(f"  [Meta] Iteration {iteration+1:4d} | "
                      f"Avg Reward: {recent_avg:.1f}")

        return meta_rewards

    def fast_adapt(self, new_trace, adaptation_episodes=5):
        """快速适应: 在新负载模式上微调元模型

        仅需1小时数据(12步@5min间隔)即可适配新模式。

        Args:
            new_trace: 新负载模式trace (可以短至12步)
            adaptation_episodes: 微调episode数

        Returns:
            adapted_agent: 适配后的CentralizedMADQN agent
        """
        agent = self._create_agent()
        rewards = self._train_on_task(agent, new_trace, adaptation_episodes)

        self.adaptation_history.append({
            "trace_length": len(new_trace),
            "episodes": adaptation_episodes,
            "rewards": rewards,
            "final_reward": rewards[-1] if rewards else 0,
        })

        return agent

    def save_meta_model(self, path):
        """保存元模型权重"""
        torch.save({
            "meta_weights": self.meta_weights,
            "meta_target_weights": self.meta_target_weights,
            "adaptation_history": self.adaptation_history,
        }, path)

    def load_meta_model(self, path):
        """加载元模型权重"""
        checkpoint = torch.load(path, weights_only=False)
        self.meta_weights = checkpoint["meta_weights"]
        self.meta_target_weights = checkpoint["meta_target_weights"]
        self.adaptation_history = checkpoint.get("adaptation_history", [])
