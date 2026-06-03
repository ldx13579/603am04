import numpy as np
from collections import deque


class ServerEnv:
    """单服务器仿真环境

    请求到达: 泊松分布(每5分钟)
    处理时间: 指数分布
    功耗模型: P = P_idle + (P_max - P_idle) * cpu_utilization
    """

    def __init__(self):
        self.dt = 5.0  # 时间步长(分钟), 每步代表5分钟
        self.arrival_rate = 1.0 / 5.0  # 每分钟到达率(泊松过程)
        self.base_service_rate = 0.5  # 基础服务率(每分钟)

        # 频率等级: 0=低频, 1=中频, 2=高频
        self.freq_levels = 3
        self.freq_multipliers = [0.5, 1.0, 1.5]  # 频率对处理速度的影响

        # 功耗参数(瓦特)
        self.p_idle = 50.0
        self.p_max = 200.0

        # 频率切换惩罚系数
        self.freq_switch_penalty = 0.2

        # 状态记录
        self.queue_length = 0
        self.current_freq = 1  # 初始中频
        self.prev_freq = 1  # 上一步频率(用于切换惩罚)
        self.time_step = 0
        self.load_history = deque(maxlen=10)  # 最近10分钟负载

        # SLA参数
        self.max_queue = 50
        self.sla_threshold = 20  # 队列超过此值开始惩罚
        self.max_steps = 288  # 每个episode 288步(5分钟间隔, 模拟24小时)

    def reset(self):
        self.queue_length = 0
        self.current_freq = 1
        self.prev_freq = 1
        self.time_step = 0
        self.load_history = deque(maxlen=10)
        for _ in range(10):
            self.load_history.append(0.0)
        return self._get_state()

    def _get_state(self):
        avg_load = np.mean(self.load_history)
        time_of_day = (self.time_step % self.max_steps) / self.max_steps  # 归一化到[0,1]
        state = np.array([
            self.queue_length / self.max_queue,  # 归一化队列长度
            avg_load,  # 平均负载(已归一化)
            time_of_day,  # 时间特征
            self.current_freq / (self.freq_levels - 1)  # 归一化频率等级
        ], dtype=np.float32)
        return state

    def step(self, action):
        """
        action: 0=降频, 1=维持, 2=升频
        """
        # 记录上一步频率
        self.prev_freq = self.current_freq

        # 执行动作: 调整频率
        if action == 0:
            self.current_freq = max(0, self.current_freq - 1)
        elif action == 2:
            self.current_freq = min(self.freq_levels - 1, self.current_freq + 1)

        # 模拟请求到达(泊松过程, 到达率随时间变化模拟日间高峰)
        hour = (self.time_step * self.dt / 60.0) % 24  # 当前小时
        time_factor = 1.0 + 1.5 * np.exp(-((hour - 14) ** 2) / 18.0)  # 14点附近高峰
        arrivals = np.random.poisson(self.arrival_rate * self.dt * time_factor)
        self.queue_length += arrivals

        # 模拟请求处理(指数分布服务时间)
        service_rate = self.base_service_rate * self.freq_multipliers[self.current_freq]
        if self.queue_length > 0:
            processed = np.random.poisson(service_rate * self.dt)
            processed = min(processed, self.queue_length)
            self.queue_length -= processed
        else:
            processed = 0

        # CPU利用率 = (处理请求数 + 队列积压压力) / 服务能力
        # 队列积压反映潜在负载，使频率调整更及时响应负载变化
        capacity = service_rate * self.dt
        effective_demand = processed + self.queue_length * 0.3
        cpu_util = min(1.0, effective_demand / max(capacity, 1e-6))

        # 记录负载
        self.load_history.append(cpu_util)

        # 计算功耗(线性模型)
        freq_factor = self.freq_multipliers[self.current_freq]
        power = self.p_idle + (self.p_max - self.p_idle) * cpu_util * freq_factor

        # 计算奖励: 节能奖励 + SLA惩罚 + 频率切换惩罚
        reward = 1.0 - power / self.p_max

        # SLA违规惩罚
        if self.queue_length > self.sla_threshold:
            penalty = -3.0 * (self.queue_length - self.sla_threshold) / self.max_queue
            reward += penalty

        # 队列溢出大惩罚
        if self.queue_length >= self.max_queue:
            reward -= 10.0
            self.queue_length = self.max_queue

        # 频率切换惩罚(按切换幅度动态调整)
        freq_diff = abs(self.current_freq - self.prev_freq)
        if freq_diff > 0:
            reward -= self.freq_switch_penalty * freq_diff

        self.time_step += 1
        done = self.time_step >= self.max_steps  # 288步 = 24小时

        return self._get_state(), reward, done, {
            "power": power,
            "queue": self.queue_length,
            "cpu_util": cpu_util,
            "freq": self.current_freq
        }
