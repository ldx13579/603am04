import numpy as np
from collections import deque
from workload_trace import AlibabaTraceGenerator


SERVER_CONFIGS = [
    {
        "name": "高性能服务器",
        "p_idle": 80.0,
        "p_max": 300.0,
        "base_service_rate": 0.8,
        "freq_multipliers": [0.5, 1.0, 1.5],
        "failure_prob": 0.002,
        "recovery_steps": 6,
    },
    {
        "name": "标准服务器",
        "p_idle": 50.0,
        "p_max": 200.0,
        "base_service_rate": 0.5,
        "freq_multipliers": [0.5, 1.0, 1.5],
        "failure_prob": 0.003,
        "recovery_steps": 4,
    },
    {
        "name": "低功耗服务器",
        "p_idle": 30.0,
        "p_max": 120.0,
        "base_service_rate": 0.3,
        "freq_multipliers": [0.6, 1.0, 1.4],
        "failure_prob": 0.001,
        "recovery_steps": 3,
    },
]


class ServerState:
    """单台服务器内部状态"""

    def __init__(self, config):
        self.config = config
        self.queue_length = 0
        self.current_freq = 1
        self.prev_freq = 1
        self.load_history = deque(maxlen=10)
        self.is_failed = False
        self.failure_countdown = 0
        self.max_queue = 50
        self.recovery_phase = 0
        self.recovery_duration = 3
        self.reset()

    def reset(self):
        self.queue_length = 0
        self.current_freq = 1
        self.prev_freq = 1
        self.load_history = deque(maxlen=10)
        for _ in range(10):
            self.load_history.append(0.0)
        self.is_failed = False
        self.failure_countdown = 0
        self.recovery_phase = 0

    def get_service_rate_factor(self):
        """恢复初期的服务速率限制: 逐步从50%提升到100%"""
        if self.recovery_phase > 0:
            progress = 1.0 - self.recovery_phase / self.recovery_duration
            return 0.5 + 0.5 * progress
        return 1.0


class ClusterEnv:
    """异构三服务器集群环境

    特性: 通信延迟、服务器故障、负载均衡、外部trace驱动
    """

    def __init__(self, server_configs=None, trace=None,
                 comm_delay_steps=1, max_steps=288,
                 energy_weight=0.5, seed=None):
        if server_configs is None:
            server_configs = SERVER_CONFIGS
        self.server_configs = server_configs
        self.num_servers = len(server_configs)
        self.servers = [ServerState(cfg) for cfg in server_configs]
        self.trace = trace
        self.comm_delay_steps = comm_delay_steps
        self.max_steps = max_steps
        self.energy_weight = energy_weight
        self.dt = 5.0
        self.freq_levels = 3
        self.sla_threshold = 20
        self.max_queue = 50

        self.time_step = 0
        self.comm_buffer = deque()
        self.rng = np.random.default_rng(seed)

        self.total_response_time = 0.0
        self.total_requests_served = 0

        self.redirect_log = []

    @property
    def local_state_dim(self):
        # 4 local + 2 global + 2*(num_servers-1) peer states
        return 4 + 2 + 2 * (self.num_servers - 1)

    @property
    def joint_state_dim(self):
        return self.num_servers * 4 + 2

    @property
    def action_dim(self):
        return 3

    def reset(self):
        for server in self.servers:
            server.reset()
        self.time_step = 0
        self.comm_buffer = deque()
        self.total_response_time = 0.0
        self.total_requests_served = 0
        self.redirect_log = []
        return self._get_obs()

    def _get_local_state(self, idx):
        server = self.servers[idx]
        avg_load = np.mean(server.load_history)
        time_of_day = (self.time_step % self.max_steps) / self.max_steps
        total_queue = sum(s.queue_length for s in self.servers)
        num_active = sum(1 for s in self.servers if not s.is_failed)

        state = [
            server.queue_length / self.max_queue,
            avg_load,
            time_of_day,
            server.current_freq / (self.freq_levels - 1),
            total_queue / (self.max_queue * self.num_servers),
            num_active / self.num_servers,
        ]

        for j in range(self.num_servers):
            if j == idx:
                continue
            peer = self.servers[j]
            state.append(peer.queue_length / self.max_queue)
            state.append(peer.current_freq / (self.freq_levels - 1))

        return np.array(state, dtype=np.float32)

    def _get_joint_state(self):
        parts = []
        for server in self.servers:
            avg_load = np.mean(server.load_history)
            time_of_day = (self.time_step % self.max_steps) / self.max_steps
            parts.extend([
                server.queue_length / self.max_queue,
                avg_load,
                time_of_day,
                server.current_freq / (self.freq_levels - 1),
            ])
        total_queue = sum(s.queue_length for s in self.servers)
        num_active = sum(1 for s in self.servers if not s.is_failed)
        parts.append(total_queue / (self.max_queue * self.num_servers))
        parts.append(num_active / self.num_servers)
        return np.array(parts, dtype=np.float32)

    def _get_obs(self):
        return {
            "joint_state": self._get_joint_state(),
            "local_states": [self._get_local_state(i) for i in range(self.num_servers)],
        }

    def step(self, actions):
        """
        actions: [action_server0, action_server1, action_server2]
        """
        for i, server in enumerate(self.servers):
            server.prev_freq = server.current_freq
            if not server.is_failed:
                if actions[i] == 0:
                    server.current_freq = max(0, server.current_freq - 1)
                elif actions[i] == 2:
                    server.current_freq = min(self.freq_levels - 1, server.current_freq + 1)

        arrivals = self._get_arrivals()
        self._distribute_requests(arrivals)
        self._deliver_delayed_requests()
        self._simulate_failures()

        powers = []
        step_response_time = 0.0
        step_served = 0

        for i, server in enumerate(self.servers):
            cfg = self.server_configs[i]
            if server.is_failed:
                powers.append(cfg["p_idle"])
                server.load_history.append(0.0)
                continue

            rate_factor = server.get_service_rate_factor()
            service_rate = cfg["base_service_rate"] * cfg["freq_multipliers"][server.current_freq] * rate_factor

            if server.queue_length > 0:
                max_process = self.rng.poisson(service_rate * self.dt)
                if server.recovery_phase > 0:
                    capacity_limit = int(service_rate * self.dt * 0.8)
                    max_process = min(max_process, max(1, capacity_limit))
                processed = min(max_process, server.queue_length)
                server.queue_length -= processed
                avg_wait = server.queue_length / max(service_rate * self.dt, 1e-6)
                step_response_time += processed * (1.0 / max(service_rate, 1e-6) + avg_wait)
                step_served += processed
            else:
                processed = 0

            if server.recovery_phase > 0:
                server.recovery_phase -= 1

            capacity = service_rate * self.dt
            effective_demand = processed + server.queue_length * 0.3
            cpu_util = min(1.0, effective_demand / max(capacity, 1e-6))
            server.load_history.append(cpu_util)

            freq_factor = cfg["freq_multipliers"][server.current_freq]
            power = cfg["p_idle"] + (cfg["p_max"] - cfg["p_idle"]) * cpu_util * freq_factor
            powers.append(power)

        if step_served > 0:
            self.total_response_time += step_response_time
            self.total_requests_served += step_served

        rewards_per_server = []
        for i, server in enumerate(self.servers):
            cfg = self.server_configs[i]
            r = self.energy_weight * (1.0 - powers[i] / cfg["p_max"])

            if server.queue_length > self.sla_threshold:
                qos_penalty = -3.0 * (server.queue_length - self.sla_threshold) / self.max_queue
                r += (1.0 - self.energy_weight) * qos_penalty

            if server.queue_length >= self.max_queue:
                r -= 10.0
                server.queue_length = self.max_queue

            freq_diff = abs(server.current_freq - server.prev_freq)
            if freq_diff > 0:
                r -= 0.2 * freq_diff

            rewards_per_server.append(r)

        queue_lengths = [s.queue_length for s in self.servers]
        balance_penalty = -np.std(queue_lengths) / self.max_queue * 0.5

        # 协作奖励: 服务器频率决策互补时给予奖励
        # 高负载服务器升频 + 低负载服务器降频 = 协同工作
        cooperation_reward = 0.0
        active_servers = [(i, s) for i, s in enumerate(self.servers) if not s.is_failed]
        if len(active_servers) >= 2:
            avg_queue = np.mean([s.queue_length for _, s in active_servers])
            for i, server in active_servers:
                is_overloaded = server.queue_length > avg_queue
                freq_change = server.current_freq - server.prev_freq
                # 高负载升频或低负载降频都是协作行为
                if is_overloaded and freq_change > 0:
                    cooperation_reward += 0.3
                elif not is_overloaded and freq_change < 0:
                    cooperation_reward += 0.3
            # 全体频率配置覆盖多层级时(异构分工), 额外奖励
            freq_set = set(s.current_freq for _, s in active_servers)
            if len(freq_set) >= 2:
                cooperation_reward += 0.2

        total_reward = sum(rewards_per_server) + balance_penalty + cooperation_reward

        self.time_step += 1
        done = self.time_step >= self.max_steps

        avg_resp = (self.total_response_time / max(self.total_requests_served, 1))
        info = {
            "powers": powers,
            "total_power": sum(powers),
            "queues": queue_lengths,
            "avg_response_time": avg_resp,
            "failures": [s.is_failed for s in self.servers],
            "freqs": [s.current_freq for s in self.servers],
            "redirects_this_step": len([r for r in self.redirect_log if r["step"] == self.time_step - 1]),
        }

        return self._get_obs(), {"total": total_reward, "per_server": rewards_per_server}, done, info

    def _get_arrivals(self):
        if self.trace is not None and self.time_step < len(self.trace):
            return int(self.trace[self.time_step])
        hour = (self.time_step * self.dt / 60.0) % 24
        time_factor = 1.0 + 1.5 * np.exp(-((hour - 14) ** 2) / 18.0)
        return int(self.rng.poisson(self.dt * time_factor))

    def _distribute_requests(self, arrivals):
        """动态负载均衡: 根据系统总负载自适应调整队列加权指数"""
        active_indices = [i for i in range(self.num_servers) if not self.servers[i].is_failed]
        if not active_indices:
            self.servers[0].queue_length = min(self.servers[0].queue_length + arrivals, self.max_queue)
            return

        total_queue = sum(self.servers[i].queue_length for i in active_indices)
        total_capacity = sum(self.max_queue for _ in active_indices)
        system_load_ratio = total_queue / max(total_capacity, 1)

        # 高负载时增大指数(1.5→4.0), 增强均衡效果
        dynamic_exponent = 1.5 + 2.5 * system_load_ratio

        weights = []
        for i in active_indices:
            cfg = self.server_configs[i]
            server = self.servers[i]
            cap = cfg["base_service_rate"] * cfg["freq_multipliers"][server.current_freq]
            cap *= server.get_service_rate_factor()

            queue_ratio = server.queue_length / self.max_queue
            queue_weight = max(0.05, (1.0 - queue_ratio) ** dynamic_exponent)
            weights.append(cap * queue_weight)

        total_weight = sum(weights)
        allocation = [0] * self.num_servers
        for i, idx in enumerate(active_indices):
            allocation[idx] = int(arrivals * weights[i] / total_weight)

        remainder = arrivals - sum(allocation)
        if remainder > 0:
            best_idx = active_indices[int(np.argmax(weights))]
            allocation[best_idx] += remainder

        self.comm_buffer.append((self.time_step + self.comm_delay_steps, allocation))

    def _deliver_delayed_requests(self):
        """从通信延迟缓冲区投递请求, 故障时重定向到队列最短的可用服务器"""
        while self.comm_buffer and self.comm_buffer[0][0] <= self.time_step:
            _, allocation = self.comm_buffer.popleft()
            for i, count in enumerate(allocation):
                if count == 0:
                    continue
                if self.servers[i].is_failed:
                    active = [j for j in range(self.num_servers)
                              if not self.servers[j].is_failed and j != i]
                    if active:
                        shortest_idx = min(active, key=lambda j: self.servers[j].queue_length)
                        self.servers[shortest_idx].queue_length += count
                        self.redirect_log.append({
                            "step": self.time_step,
                            "from": i,
                            "to": shortest_idx,
                            "count": count,
                            "reason": "server_failure",
                            "target_queue_before": self.servers[shortest_idx].queue_length - count,
                        })
                    else:
                        self.servers[i].queue_length += count
                else:
                    self.servers[i].queue_length += count

    def _simulate_failures(self):
        for i, server in enumerate(self.servers):
            if server.is_failed:
                server.failure_countdown -= 1
                if server.failure_countdown <= 0:
                    server.is_failed = False
                    server.recovery_phase = server.recovery_duration
            else:
                if self.rng.random() < self.server_configs[i]["failure_prob"]:
                    server.is_failed = True
                    server.failure_countdown = self.server_configs[i]["recovery_steps"]
