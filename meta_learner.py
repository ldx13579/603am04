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

    多维度综合检测:
    - 总负载均值漂移 (CUSUM风格)
    - 总负载方差变化
    - 各服务器负载分布差异 (Gini系数变化 + 负载不均衡度漂移)
    - 服务器间负载相关性变化

    综合漂移分数融合以上维度, 超过阈值时触发适应。
    """

    def __init__(self, window_size=24, drift_threshold=2.5,
                 variance_threshold=3.0, cooldown_steps=12,
                 num_servers=3):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.variance_threshold = variance_threshold
        self.cooldown_steps = cooldown_steps
        self.num_servers = num_servers

        # 总负载历史
        self.history = deque(maxlen=window_size * 2)
        self.reference_mean = None
        self.reference_std = None

        # 各服务器负载分布历史
        self.server_histories = [
            deque(maxlen=window_size * 2) for _ in range(num_servers)
        ]
        self.reference_gini = None
        self.reference_imbalance = None

        self.steps_since_last_trigger = cooldown_steps
        self.drift_detected = False
        self.drift_score = 0.0
        self.distribution_drift_score = 0.0

    def reset(self):
        """重置检测器状态"""
        self.history.clear()
        for h in self.server_histories:
            h.clear()
        self.reference_mean = None
        self.reference_std = None
        self.reference_gini = None
        self.reference_imbalance = None
        self.steps_since_last_trigger = self.cooldown_steps
        self.drift_detected = False
        self.drift_score = 0.0
        self.distribution_drift_score = 0.0

    @staticmethod
    def _compute_gini(values):
        """计算Gini系数衡量负载分布不均匀度 (0=完全均匀, 1=完全集中)"""
        values = np.array(values, dtype=np.float64)
        if len(values) == 0 or np.sum(values) == 0:
            return 0.0
        sorted_v = np.sort(values)
        n = len(sorted_v)
        cumsum = np.cumsum(sorted_v)
        return (2.0 * np.sum((np.arange(1, n + 1) * sorted_v)) /
                (n * np.sum(sorted_v)) - (n + 1) / n)

    @staticmethod
    def _compute_imbalance(values):
        """计算负载不均衡度: max/mean比率"""
        values = np.array(values, dtype=np.float64)
        mean_v = np.mean(values)
        if mean_v < 1e-6:
            return 1.0
        return np.max(values) / mean_v

    def update(self, load_value, per_server_loads=None):
        """输入新的负载观测值, 返回是否检测到模式变化

        Args:
            load_value: 当前步的总负载值
            per_server_loads: 各服务器负载列表 [q0, q1, q2]
                若提供则启用分布差异检测

        Returns:
            drift_detected: bool, 是否检测到模式变化
        """
        self.history.append(load_value)
        if per_server_loads is not None:
            for i, load in enumerate(per_server_loads[:self.num_servers]):
                self.server_histories[i].append(load)

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

        # 服务器负载分布差异检测
        self.distribution_drift_score = 0.0
        if per_server_loads is not None and all(
            len(h) >= self.window_size for h in self.server_histories
        ):
            dist_score = self._compute_distribution_drift()
            self.distribution_drift_score = dist_score

        # 综合漂移分数 (融合总负载漂移 + 分布差异)
        load_drift = max(
            mean_z_score / self.drift_threshold,
            (var_ratio if var_ratio > 1 else 1.0 / max(var_ratio, 1e-6))
            / self.variance_threshold,
        )
        self.drift_score = max(load_drift, self.distribution_drift_score)

        if self.steps_since_last_trigger < self.cooldown_steps:
            return False

        # 触发条件: 总负载漂移 OR 分布差异超阈值
        triggered = (
            mean_z_score > self.drift_threshold
            or var_ratio > self.variance_threshold
            or self.distribution_drift_score > 1.0
        )

        if triggered:
            self.drift_detected = True
            self.steps_since_last_trigger = 0
            self.reference_mean = new_mean
            self.reference_std = max(new_std, 1e-6)
            self._update_distribution_reference()
            return True

        # 渐进更新参考值
        alpha = 0.05
        self.reference_mean = (1 - alpha) * self.reference_mean + alpha * new_mean
        self.reference_std = max(
            (1 - alpha) * self.reference_std + alpha * new_std, 1e-6
        )

        return False

    def _compute_distribution_drift(self):
        """计算服务器间负载分布的漂移分数

        综合三个维度:
        1. Gini系数变化: 负载集中度是否发生变化
        2. 不均衡度变化: 最大负载/平均负载比率是否异常
        3. 服务器排序变化: 负载排名是否翻转 (新模式可能改变瓶颈服务器)
        """
        mid = self.window_size
        scores = []

        # 各窗口内各服务器的平均负载
        old_server_means = []
        new_server_means = []
        for i in range(self.num_servers):
            h = list(self.server_histories[i])
            if len(h) < mid:
                return 0.0
            old_part = h[:mid]
            new_part = h[mid:]
            if not old_part or not new_part:
                return 0.0
            old_server_means.append(np.mean(old_part))
            new_server_means.append(np.mean(new_part))

        # 1. Gini系数变化
        old_gini = self._compute_gini(old_server_means)
        new_gini = self._compute_gini(new_server_means)

        if self.reference_gini is None:
            self.reference_gini = old_gini

        gini_change = abs(new_gini - self.reference_gini) / max(self.reference_gini + 0.1, 0.1)
        scores.append(gini_change)

        # 2. 不均衡度变化
        old_imbalance = self._compute_imbalance(old_server_means)
        new_imbalance = self._compute_imbalance(new_server_means)

        if self.reference_imbalance is None:
            self.reference_imbalance = old_imbalance

        imbalance_change = abs(new_imbalance - self.reference_imbalance) / max(self.reference_imbalance, 1.0)
        scores.append(imbalance_change)

        # 3. 负载排序变化 (Kendall tau-like)
        old_rank = np.argsort(old_server_means)
        new_rank = np.argsort(new_server_means)
        rank_change = np.sum(old_rank != new_rank) / self.num_servers
        scores.append(rank_change)

        # 4. 各服务器负载比例变化
        old_total = max(sum(old_server_means), 1e-6)
        new_total = max(sum(new_server_means), 1e-6)
        old_proportions = np.array(old_server_means) / old_total
        new_proportions = np.array(new_server_means) / new_total
        proportion_shift = np.sum(np.abs(new_proportions - old_proportions))
        scores.append(proportion_shift)

        # 加权综合 (Gini变化和比例偏移权重较高)
        weights = [0.3, 0.2, 0.2, 0.3]
        combined = sum(s * w for s, w in zip(scores, weights))
        return combined / 0.3  # 归一化使得>1.0表示显著变化

    def _update_distribution_reference(self):
        """触发后更新分布参考值"""
        mid = self.window_size
        new_means = []
        for i in range(self.num_servers):
            h = list(self.server_histories[i])
            if len(h) > mid:
                new_means.append(np.mean(h[mid:]))
            else:
                new_means.append(np.mean(h) if h else 0)
        self.reference_gini = self._compute_gini(new_means)
        self.reference_imbalance = self._compute_imbalance(new_means)

    def get_recent_trace(self):
        """获取检测器中累积的近期负载数据作为adaptation trace"""
        return np.array(list(self.history), dtype=np.int32)

    def get_extended_trace(self, extra_history=None):
        """获取扩充后的近期负载数据, 合并额外历史以保障适应效果

        Args:
            extra_history: 额外的历史负载数据 (deque or list)

        Returns:
            合并后的trace, 优先使用更多数据以捕捉完整模式
        """
        recent = list(self.history)
        if extra_history:
            extra = list(extra_history)
            combined = extra + recent
            return np.array(combined, dtype=np.int32)
        return np.array(recent, dtype=np.int32)


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
            num_servers=self.num_servers,
        ) if auto_detect else None
        self._last_adapted_agent = None
        self._auto_adapt_count = 0

        # 扩充历史数据缓冲: 保留更长的负载历史以备适应时使用
        self._load_history_buffer = deque(maxlen=detect_window * 4)

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

    def observe_and_adapt(self, load_value, current_agent=None,
                          per_server_loads=None):
        """观测新负载值, 自动检测模式变化并触发适应

        在每步环境交互时调用此方法。检测到模式漂移时,
        合并检测器窗口数据和扩充历史缓冲进行fast_adapt,
        保障适应过程使用更充分的历史数据。

        Args:
            load_value: 当前步的总负载观测值
            current_agent: 当前使用的agent
            per_server_loads: 各服务器负载列表 [q0, q1, q2],
                提供后启用分布差异检测

        Returns:
            (adapted, agent): adapted为是否触发了适应, agent为当前应使用的agent
        """
        if not self.auto_detect or self.detector is None:
            return False, current_agent

        # 记录到扩充历史缓冲 (比检测器窗口更长)
        self._load_history_buffer.append(load_value)

        drift = self.detector.update(load_value, per_server_loads)

        if drift:
            # 使用扩充历史数据: 合并检测器窗口 + 额外历史缓冲
            adapt_trace = self.detector.get_extended_trace(
                extra_history=self._load_history_buffer
            )
            # 至少需要6步数据才有意义
            if len(adapt_trace) >= 6:
                adapted_agent = self.fast_adapt(adapt_trace, self.inner_episodes)
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
