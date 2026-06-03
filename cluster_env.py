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

        # 在线迁移相关
        self.sustained_overload_counter = 0
        self.incoming_migration_buffer = deque()  # (arrival_step, count, source_id)
        self.request_sources = {}  # {source_server_id: count} 按来源分类计数

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
        self.sustained_overload_counter = 0
        self.incoming_migration_buffer = deque()
        self.request_sources = {}

    def get_service_rate_factor(self):
        """恢复初期的服务速率限制: 逐步从50%提升到100%"""
        if self.recovery_phase > 0:
            progress = 1.0 - self.recovery_phase / self.recovery_duration
            return 0.5 + 0.5 * progress
        return 1.0

    def get_pending_migration_count(self):
        """返回正在迁移中尚未到达的请求总数"""
        return sum(count for _, count, _ in self.incoming_migration_buffer)

    def get_local_request_ratio(self):
        """本地原生请求占比 (非迁移来的)"""
        total = sum(self.request_sources.values())
        if total == 0:
            return 1.0
        local = self.request_sources.get(-1, 0)  # -1 表示本地请求
        return local / total


class ClusterEnv:
    """异构三服务器集群环境 (含在线迁移决策)

    动作空间 (每台服务器6个动作):
        0: 降频
        1: 维持频率
        2: 升频
        3: 触发迁移 (低强度)
        4: 触发迁移 (中强度)
        5: 触发迁移 (高强度)

    迁移比例根据源队列长度和目标剩余容量动态计算:
        base_fraction = {3: 0.2, 4: 0.4, 5: 0.6}
        actual_fraction = base_fraction * (src_queue / max_queue) * (target_remaining / max_queue)

    迁移延迟根据系统负载动态调整:
        delay = base_delay + int(system_load_ratio * 2)
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

        # 迁移参数
        self.migration_base_delay = 1
        self.migration_energy_per_request = 0.5  # 每请求迁移能耗(瓦特)
        self.overload_threshold = 0.6  # 队列占比超此值计为过载

    @property
    def local_state_dim(self):
        # 6 local + 2 global + 3*(num_servers-1) peer states
        return 6 + 2 + 3 * (self.num_servers - 1)

    @property
    def joint_state_dim(self):
        return self.num_servers * 6 + 2

    @property
    def action_dim(self):
        return 6

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
            server.sustained_overload_counter / 10.0,
            server.get_local_request_ratio(),
            # 全局特征
            total_queue / (self.max_queue * self.num_servers),
            num_active / self.num_servers,
        ]

        # 邻居信息 (每个邻居3个特征)
        for j in range(self.num_servers):
            if j == idx:
                continue
            peer = self.servers[j]
            state.append(peer.queue_length / self.max_queue)
            state.append(peer.current_freq / (self.freq_levels - 1))
            state.append(peer.sustained_overload_counter / 10.0)

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
                server.sustained_overload_counter / 10.0,
                server.get_local_request_ratio(),
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

    def _get_system_load_ratio(self):
        """当前系统整体负载比例"""
        total_queue = sum(s.queue_length for s in self.servers)
        total_capacity = self.max_queue * self.num_servers
        return total_queue / max(total_capacity, 1)

    def _compute_migration_delay(self):
        """根据系统负载动态调整迁移延迟步数"""
        load_ratio = self._get_system_load_ratio()
        return self.migration_base_delay + int(load_ratio * 2)

    def _compute_dynamic_migration_count(self, action, source_idx, target_idx):
        """根据源队列长度、目标剩余容量和服务器处理能力动态计算迁移数量

        base_fraction由动作决定基础迁移强度
        processing_weight = target_service_rate / src_service_rate (目标处理能力越强分配越多)
        实际迁移量 = base_fraction * src_queue * capacity_ratio * processing_weight
        """
        base_fractions = {3: 0.2, 4: 0.4, 5: 0.6}
        base_frac = base_fractions[action]

        src_queue = self.servers[source_idx].queue_length
        target_remaining = self.max_queue - self.servers[target_idx].queue_length
        target_capacity_ratio = target_remaining / self.max_queue

        # 计算处理能力权重: 目标服务速率 / 源服务速率
        src_cfg = self.server_configs[source_idx]
        tgt_cfg = self.server_configs[target_idx]
        src_server = self.servers[source_idx]
        tgt_server = self.servers[target_idx]

        src_rate = (src_cfg["base_service_rate"]
                    * src_cfg["freq_multipliers"][src_server.current_freq]
                    * src_server.get_service_rate_factor())
        tgt_rate = (tgt_cfg["base_service_rate"]
                    * tgt_cfg["freq_multipliers"][tgt_server.current_freq]
                    * tgt_server.get_service_rate_factor())
        # 处理能力权重: 目标越强则迁移越多, 钳位在[0.5, 2.0]避免极端值
        processing_weight = np.clip(tgt_rate / max(src_rate, 1e-6), 0.5, 2.0)

        # 动态比例: 源越满 × 目标越空 × 目标处理能力越强 → 迁移越多
        effective_frac = base_frac * (src_queue / self.max_queue) * target_capacity_ratio * processing_weight
        migrate_count = max(1, int(src_queue * effective_frac))
        migrate_count = min(migrate_count, src_queue, target_remaining)
        return migrate_count

    def _execute_migrations(self, actions):
        """处理迁移动作 (actions 3/4/5)"""
        migration_info = []
        delay = self._compute_migration_delay()

        for i, server in enumerate(self.servers):
            if server.is_failed or actions[i] < 3:
                continue
            if server.queue_length <= 0:
                continue

            # 找最空闲的活跃邻居
            peers = [j for j in range(self.num_servers)
                     if j != i and not self.servers[j].is_failed]
            if not peers:
                continue

            target_idx = min(peers, key=lambda j: self.servers[j].queue_length)

            # 如果目标已经比源还满, 不迁移
            if self.servers[target_idx].queue_length >= server.queue_length:
                continue

            migrate_count = self._compute_dynamic_migration_count(
                actions[i], i, target_idx
            )
            if migrate_count <= 0:
                continue

            # 源立即减少
            server.queue_length -= migrate_count
            # 更新源的请求来源计数
            local_in_source = server.request_sources.get(-1, server.queue_length)
            removed_local = min(migrate_count, local_in_source)
            server.request_sources[-1] = max(0, local_in_source - removed_local)

            # 目标延迟接收
            arrival_step = self.time_step + delay
            self.servers[target_idx].incoming_migration_buffer.append(
                (arrival_step, migrate_count, i)
            )

            migration_info.append({
                "step": self.time_step,
                "from": i,
                "to": target_idx,
                "count": migrate_count,
                "delay": delay,
                "reason": "agent_migration",
            })

        self.redirect_log.extend(migration_info)
        return migration_info

    def _deliver_migration_buffer(self):
        """投递已完成迁移传输的请求"""
        for server_idx, server in enumerate(self.servers):
            while (server.incoming_migration_buffer and
                   server.incoming_migration_buffer[0][0] <= self.time_step):
                _, count, source_id = server.incoming_migration_buffer.popleft()
                delivered = min(count, self.max_queue - server.queue_length)
                server.queue_length += delivered
                # 按来源记录
                server.request_sources[source_id] = \
                    server.request_sources.get(source_id, 0) + delivered

    def _update_overload_counters(self):
        """更新持续过载计数器"""
        for server in self.servers:
            if server.queue_length / self.max_queue > self.overload_threshold:
                server.sustained_overload_counter += 1
            else:
                server.sustained_overload_counter = max(
                    0, server.sustained_overload_counter - 1
                )

    def step(self, actions):
        """
        actions: [action_server0, action_server1, action_server2]
        每个action取值0-5
        """
        # 1. 频率调整 (仅动作0/1/2调频, 3/4/5不改频率)
        for i, server in enumerate(self.servers):
            server.prev_freq = server.current_freq
            if not server.is_failed and actions[i] <= 2:
                if actions[i] == 0:
                    server.current_freq = max(0, server.current_freq - 1)
                elif actions[i] == 2:
                    server.current_freq = min(self.freq_levels - 1, server.current_freq + 1)

        # 2. 执行迁移动作
        migration_info = self._execute_migrations(actions)

        # 3. 投递已完成的迁移请求
        self._deliver_migration_buffer()

        # 4. 请求到达与分发
        arrivals = self._get_arrivals()
        self._distribute_requests(arrivals)
        self._deliver_delayed_requests()

        # 5. 故障模拟
        self._simulate_failures()

        # 6. 更新过载计数器
        self._update_overload_counters()

        # 7. 服务处理
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

                # 按比例减少各来源计数
                total_src = sum(server.request_sources.values())
                if total_src > 0:
                    for src_id in list(server.request_sources.keys()):
                        ratio = server.request_sources[src_id] / total_src
                        remove = int(processed * ratio)
                        server.request_sources[src_id] = max(
                            0, server.request_sources[src_id] - remove
                        )

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

        # 8. 计算奖励
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

            # 迁移奖励/惩罚
            server_migrations = [m for m in migration_info if m["from"] == i]
            if server_migrations:
                total_migrated = sum(m["count"] for m in server_migrations)
                # 迁移能耗惩罚
                migration_energy_penalty = (
                    total_migrated * self.migration_energy_per_request / cfg["p_max"] * 0.5
                )
                r -= migration_energy_penalty

                # 迁移成功奖励: 迁移后队列降到SLA以下
                if server.queue_length < self.sla_threshold:
                    r += 0.5

                # 过早迁移惩罚: 过载计数不足时不该迁移
                if server.sustained_overload_counter < 2:
                    r -= 1.0

            # 基于请求来源的差异化奖励
            # 本地请求处理优先: 本地占比高时给予奖励(鼓励就地处理)
            # 迁入请求积压惩罚: 迁入占比高时额外惩罚(避免成为迁移黑洞)
            local_ratio = server.get_local_request_ratio()
            migrated_in_count = sum(
                v for k, v in server.request_sources.items() if k != -1
            )
            if server.queue_length > 0:
                # 本地请求占比高 → 奖励 (鼓励本地高效处理)
                r += 0.2 * local_ratio
                # 迁入请求过多时惩罚 (防止某服务器成为迁移汇聚点)
                migrated_ratio = migrated_in_count / max(server.queue_length, 1)
                if migrated_ratio > 0.5 and server.queue_length > self.sla_threshold:
                    r -= 0.5 * migrated_ratio

            rewards_per_server.append(r)

        # 负载均衡惩罚
        queue_lengths = [s.queue_length for s in self.servers]
        balance_penalty = -np.std(queue_lengths) / self.max_queue * 0.5

        # 协作奖励
        cooperation_reward = 0.0
        active_servers = [(i, s) for i, s in enumerate(self.servers) if not s.is_failed]
        if len(active_servers) >= 2:
            avg_queue = np.mean([s.queue_length for _, s in active_servers])
            for i, server in active_servers:
                is_overloaded = server.queue_length > avg_queue
                freq_change = server.current_freq - server.prev_freq
                if is_overloaded and freq_change > 0:
                    cooperation_reward += 0.3
                elif not is_overloaded and freq_change < 0:
                    cooperation_reward += 0.3
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
            "migrations_this_step": len(migration_info),
            "overload_counters": [s.sustained_overload_counter for s in self.servers],
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
                        self.servers[shortest_idx].request_sources[-1] = \
                            self.servers[shortest_idx].request_sources.get(-1, 0) + count
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
                    # 记录为本地请求
                    self.servers[i].request_sources[-1] = \
                        self.servers[i].request_sources.get(-1, 0) + count

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
