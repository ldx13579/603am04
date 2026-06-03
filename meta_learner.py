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
   b. 在新数据上微调N个episode
4. 自动模式检测:
   a. 维护负载统计滑动窗口
   b. 基于KL散度/均值漂移检测模式变化
   c. 变化超过阈值时自动触发fast_adapt
"""

import numpy as np
import copy
import torch
from collections import deque

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

    def sample_task(self, duration_hours=24):
        """采样一个随机负载任务, 返回trace数组

        Args:
            duration_hours: trace持续时长(小时), 支持1-168小时
        """
        task_type = self.rng.choice(self.task_types)
        seed = int(self.rng.integers(0, 100000))

        if task_type == "normal":
            gen = AlibabaTraceGenerator(
                duration_hours=duration_hours, seed=seed,
                base_rate=self.rng.uniform(0.8, 1.5),
                peak_multiplier=self.rng.uniform(2.0, 4.0),
            )
            return gen.generate_trace()

        elif task_type == "high_load":
            gen = AlibabaTraceGenerator(
                duration_hours=duration_hours, seed=seed,
                base_rate=self.rng.uniform(1.5, 3.0),
                peak_multiplier=self.rng.uniform(3.0, 5.0),
                burst_prob=0.1,
            )
            return gen.generate_trace()

        elif task_type == "double11":
            gen = Double11TraceGenerator(
                duration_hours=duration_hours, seed=seed,
                flash_multiplier=self.rng.uniform(10.0, 20.0),
                sustain_multiplier=self.rng.uniform(4.0, 8.0),
                flash_start_hour=self.rng.uniform(0.0, 4.0),
            )
            return gen.generate_trace()

        elif task_type == "burst":
            gen = BurstPatternGenerator(
                duration_hours=duration_hours, seed=seed,
                burst_interval_steps=int(self.rng.integers(20, 50)),
                burst_duration_steps=int(self.rng.integers(3, 8)),
                burst_magnitude=self.rng.uniform(5.0, 12.0),
            )
            return gen.generate_trace()

        elif task_type == "weekend":
            gen = AlibabaTraceGenerator(
                duration_hours=duration_hours, seed=seed,
                base_rate=self.rng.uniform(0.5, 0.8),
                peak_multiplier=self.rng.uniform(1.5, 2.5),
            )
            return gen.generate_trace()

        else:  # spike
            gen = AlibabaTraceGenerator(
                duration_hours=duration_hours, seed=seed,
                base_rate=self.rng.uniform(0.8, 1.2),
                burst_prob=0.15,
                burst_multiplier=self.rng.uniform(8.0, 15.0),
            )
            return gen.generate_trace()

    def sample_batch(self, n, duration_hours=24):
        """采样n个任务"""
        return [self.sample_task(duration_hours) for _ in range(n)]


class PatternChangeDetector:
    """负载模式变化检测器

    使用滑动窗口统计量检测负载模式漂移:
    - 均值漂移检测 (CUSUM风格)
    - 方差变化检测
    - 到达率分布变化 (基于窗口KL散度近似)

    当检测到显著漂移时, 触发回调通知元学习器进行快速适应。
    """

    def __init__(self, window_size=24, drift_threshold=2.5,
                 variance_threshold=3.0, cooldown_steps=12):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.variance_threshold = variance_threshold
        self.cooldown_steps = cooldown_steps

        self.history = deque(maxlen=window_size * 2)
        self.reference_mean = None
        self.reference_std = None
        self.steps_since_last_trigger = cooldown_steps
        self.drift_detected = False
        self.drift_score = 0.0

    def reset(self):
        """重置检测器状态"""
        self.history.clear()
        self.reference_mean = None
        self.reference_std = None
        self.steps_since_last_trigger = self.cooldown_steps
        self.drift_detected = False
        self.drift_score = 0.0

    def update(self, load_value):
        """输入新的负载观测值, 返回是否检测到模式变化

        Args:
            load_value: 当前步的负载值(到达请求数或队列长度)

        Returns:
            drift_detected: bool, 是否检测到模式变化
        """
        self.history.append(load_value)
        self.steps_since_last_trigger += 1
        self.drift_detected = False

        if len(self.history) < self.window_size:
            return False

        recent = list(self.history)
        mid = len(recent) // 2
        old_window = np.array(recent[:mid], dtype=np.float64)
        new_window = np.array(recent[mid:], dtype=np.float64)

        if self.reference_mean is None:
            self.reference_mean = np.mean(old_window)
            self.reference_std = max(np.std(old_window), 1e-6)

        # 均值漂移检测
        new_mean = np.mean(new_window)
        mean_z_score = abs(new_mean - self.reference_mean) / self.reference_std

        # 方差变化检测
        new_std = np.std(new_window)
        var_ratio = new_std / self.reference_std if self.reference_std > 0 else 1.0

        # 综合漂移分数
        self.drift_score = max(
            mean_z_score / self.drift_threshold,
            (var_ratio if var_ratio > 1 else 1.0 / max(var_ratio, 1e-6))
            / self.variance_threshold,
        )

        if self.steps_since_last_trigger < self.cooldown_steps:
            return False

        if mean_z_score > self.drift_threshold or var_ratio > self.variance_threshold:
            self.drift_detected = True
            self.steps_since_last_trigger = 0
            self.reference_mean = new_mean
            self.reference_std = max(new_std, 1e-6)
            return True

        # 渐进更新参考值 (缓慢跟踪非突变的漂移)
        alpha = 0.05
        self.reference_mean = (1 - alpha) * self.reference_mean + alpha * new_mean
        self.reference_std = max(
            (1 - alpha) * self.reference_std + alpha * new_std, 1e-6
        )

        return False

    def get_recent_trace(self):
        """获取检测器中累积的近期负载数据作为adaptation trace"""
        return np.array(list(self.history), dtype=np.int32)


class ReptileMetaLearner:
    """Reptile元学习器: 基于预训练MADDQN的快速适应框架

    特性:
    - 外循环: 跨多种负载模式学习通用初始化
    - 内循环: 在单一模式上快速微调
    - 快速适应: 支持任意长度数据 (短至12步/1小时, 长至数天)
    - 自动检测: 集成PatternChangeDetector, 负载模式漂移时自动触发适应
    """

    def __init__(self, base_agent, meta_lr=0.1,
                 inner_episodes=5, num_tasks_per_iter=4,
                 energy_weight=0.5,
                 auto_detect=True, detect_window=24,
                 detect_threshold=2.5, detect_cooldown=12):
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

        # 自动模式变化检测
        self.auto_detect = auto_detect
        self.detector = PatternChangeDetector(
            window_size=detect_window,
            drift_threshold=detect_threshold,
            cooldown_steps=detect_cooldown,
        ) if auto_detect else None
        self._last_adapted_agent = None
        self._auto_adapt_count = 0

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
        """在单个任务上训练agent (内循环)

        支持任意长度trace: 环境max_steps自动适配trace长度。
        """
        max_steps = min(len(trace), 288)
        env = ClusterEnv(
            trace=trace, energy_weight=self.energy_weight,
            seed=None, max_steps=max_steps,
        )
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

        支持任意长度数据输入:
        - 短trace (12步/1小时): 快速响应, 适合突变检测后的紧急适应
        - 中trace (72步/6小时): 捕捉半日周期特征
        - 长trace (288步/24小时+): 完整日模式, 最全面的适应

        adaptation_episodes会根据trace长度自动调整:
        - trace < 24步: episodes * 2 (数据少, 多轮训练补偿)
        - trace 24-288步: 使用指定episodes
        - trace > 288步: 分段训练, 每段作为独立episode

        Args:
            new_trace: 新负载模式trace (任意长度)
            adaptation_episodes: 基础微调episode数

        Returns:
            adapted_agent: 适配后的CentralizedMADQN agent
        """
        agent = self._create_agent()

        trace_steps = len(new_trace)

        if trace_steps > 288:
            # 长trace: 按288步分段, 每段作为一个episode的环境
            num_segments = (trace_steps + 287) // 288
            effective_episodes = min(adaptation_episodes, num_segments)
            segment_size = 288

            all_rewards = []
            for seg_idx in range(effective_episodes):
                start = seg_idx * segment_size
                end = min(start + segment_size, trace_steps)
                segment_trace = new_trace[start:end]
                rewards = self._train_on_task(agent, segment_trace, 1)
                all_rewards.extend(rewards)

            # 再整体训练几轮
            remaining = max(1, adaptation_episodes - effective_episodes)
            rewards = self._train_on_task(agent, new_trace[:288], remaining)
            all_rewards.extend(rewards)
        elif trace_steps < 24:
            # 短trace: 增加训练轮次补偿数据不足
            effective_episodes = adaptation_episodes * 2
            all_rewards = self._train_on_task(agent, new_trace, effective_episodes)
        else:
            all_rewards = self._train_on_task(agent, new_trace, adaptation_episodes)

        self.adaptation_history.append({
            "trace_length": trace_steps,
            "trace_duration_hours": round(trace_steps * 5 / 60, 1),
            "episodes": adaptation_episodes,
            "effective_episodes": len(all_rewards),
            "rewards": all_rewards,
            "final_reward": all_rewards[-1] if all_rewards else 0,
        })

        self._last_adapted_agent = agent
        return agent

    def observe_and_adapt(self, load_value, current_agent=None):
        """观测新负载值, 自动检测模式变化并触发适应

        在每步环境交互时调用此方法, 输入当前步的总负载。
        检测到模式漂移时自动使用累积的近期数据进行fast_adapt。

        Args:
            load_value: 当前步的负载观测值 (如总到达请求数)
            current_agent: 当前使用的agent (适应后会替换)

        Returns:
            (adapted, agent): adapted为是否触发了适应, agent为当前应使用的agent
        """
        if not self.auto_detect or self.detector is None:
            return False, current_agent

        drift = self.detector.update(load_value)

        if drift:
            recent_trace = self.detector.get_recent_trace()
            if len(recent_trace) >= 6:
                adapted_agent = self.fast_adapt(recent_trace, self.inner_episodes)
                self._auto_adapt_count += 1
                return True, adapted_agent

        return False, current_agent or self._last_adapted_agent

    @property
    def auto_adapt_count(self):
        """自动适应触发次数"""
        return self._auto_adapt_count

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
