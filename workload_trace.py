import numpy as np


class AlibabaTraceGenerator:
    """基于阿里巴巴2018集群trace特征的工作负载合成器

    特征: 日间双峰模式、Negative Binomial过分散、突发事件
    """

    def __init__(self, duration_hours=24, dt_minutes=5, base_rate=1.0,
                 peak_hour=14.0, peak_multiplier=3.0,
                 burst_prob=0.05, burst_multiplier=5.0, seed=None):
        self.duration_hours = duration_hours
        self.dt_minutes = dt_minutes
        self.base_rate = base_rate
        self.peak_hour = peak_hour
        self.peak_multiplier = peak_multiplier
        self.burst_prob = burst_prob
        self.burst_multiplier = burst_multiplier
        self.num_steps = int(duration_hours * 60 / dt_minutes)
        self.rng = np.random.default_rng(seed)

    def _diurnal_pattern(self, hour):
        """双峰日间模式: 上午10点小峰 + 下午14点主峰"""
        primary = np.exp(-((hour - self.peak_hour) ** 2) / 18.0)
        secondary = 0.5 * np.exp(-((hour - 10.0) ** 2) / 8.0)
        night_base = 0.3
        return night_base + self.peak_multiplier * (primary + secondary)

    def _generate_arrivals(self, mean_rate):
        """用Negative Binomial生成过分散到达(variance > mean)"""
        if mean_rate < 0.1:
            return 0
        n_param = 3.0
        p_param = n_param / (n_param + mean_rate)
        return int(self.rng.negative_binomial(n_param, p_param))

    def generate_trace(self):
        """生成单天trace, shape=(num_steps,)"""
        trace = np.zeros(self.num_steps, dtype=np.int32)
        for step in range(self.num_steps):
            hour = (step * self.dt_minutes / 60.0) % 24.0
            rate = self.base_rate * self._diurnal_pattern(hour) * self.dt_minutes

            arrivals = self._generate_arrivals(rate)

            if self.rng.random() < self.burst_prob:
                arrivals = int(arrivals * self.burst_multiplier)

            trace[step] = max(0, arrivals)
        return trace

    def generate_multi_day_trace(self, days=7):
        """生成多天trace, 周末负载降低30%"""
        traces = []
        for day in range(days):
            day_trace = self.generate_trace()
            is_weekend = day % 7 >= 5
            if is_weekend:
                day_trace = (day_trace * 0.7).astype(np.int32)
            traces.append(day_trace)
        return np.concatenate(traces)


class Double11TraceGenerator:
    """双十一/闪购场景工作负载生成器

    特征:
    - 预热阶段 (峰值前4小时): 负载缓慢上升到基础的3倍
    - 闪购脉冲 (00:00-00:05): 瞬时10-20倍基础负载
    - 持续高负载 (峰值后5min-2h): 维持5-8倍
    - 逐步回落 (峰值后2h-8h): 指数衰减到2倍
    - 余波震荡 (之后): 间歇性小脉冲
    """

    def __init__(self, duration_hours=24, dt_minutes=5, base_rate=1.0,
                 flash_start_hour=0.0, flash_multiplier=15.0,
                 sustain_multiplier=6.0, warmup_hours=4.0, seed=None):
        self.duration_hours = duration_hours
        self.dt_minutes = dt_minutes
        self.base_rate = base_rate
        self.flash_start_hour = flash_start_hour
        self.flash_multiplier = flash_multiplier
        self.sustain_multiplier = sustain_multiplier
        self.warmup_hours = warmup_hours
        self.num_steps = int(duration_hours * 60 / dt_minutes)
        self.rng = np.random.default_rng(seed)

    def _get_multiplier(self, hour):
        """根据时间计算负载倍率"""
        flash = self.flash_start_hour
        warmup_start = (flash - self.warmup_hours) % 24

        # 计算距离flash的小时差
        if warmup_start < flash:
            in_warmup = warmup_start <= hour < flash
        else:
            in_warmup = hour >= warmup_start or hour < flash

        time_since_flash = (hour - flash) % 24

        if in_warmup:
            hours_into_warmup = (hour - warmup_start) % 24
            progress = hours_into_warmup / self.warmup_hours
            return 1.0 + 2.0 * progress ** 2

        if time_since_flash < 1.0 / 12.0:  # 5分钟脉冲
            noise = self.rng.uniform(0.8, 1.2)
            return self.flash_multiplier * noise

        if time_since_flash < 2.0:  # 持续高负载2小时
            decay = 1.0 - 0.3 * (time_since_flash / 2.0)
            return self.sustain_multiplier * decay

        if time_since_flash < 8.0:  # 指数衰减
            t = time_since_flash - 2.0
            return 2.0 + (self.sustain_multiplier - 2.0) * np.exp(-t / 2.0)

        # 余波: 基础 + 间歇性小脉冲
        if self.rng.random() < 0.1:
            return 2.0 + self.rng.uniform(0, 2.0)
        return 1.0 + 0.5 * np.sin(time_since_flash * 0.5)

    def generate_trace(self):
        """生成双十一场景trace"""
        trace = np.zeros(self.num_steps, dtype=np.int32)
        for step in range(self.num_steps):
            hour = (step * self.dt_minutes / 60.0) % 24.0
            multiplier = self._get_multiplier(hour)
            mean_rate = self.base_rate * multiplier * self.dt_minutes

            n_param = 3.0
            p_param = n_param / (n_param + max(mean_rate, 0.1))
            arrivals = int(self.rng.negative_binomial(n_param, p_param))
            trace[step] = max(0, arrivals)
        return trace


class BurstPatternGenerator:
    """可配置突发注入工作负载生成器 (用于鲁棒性测试)

    在基础日间模式上叠加可控突发事件:
    - burst_interval_steps: 突发间隔步数
    - burst_duration_steps: 每次突发持续步数
    - burst_magnitude: 突发倍率
    """

    def __init__(self, duration_hours=24, dt_minutes=5, base_rate=1.0,
                 burst_interval_steps=30, burst_duration_steps=5,
                 burst_magnitude=8.0, seed=None):
        self.duration_hours = duration_hours
        self.dt_minutes = dt_minutes
        self.base_rate = base_rate
        self.burst_interval_steps = burst_interval_steps
        self.burst_duration_steps = burst_duration_steps
        self.burst_magnitude = burst_magnitude
        self.num_steps = int(duration_hours * 60 / dt_minutes)
        self.rng = np.random.default_rng(seed)

    def _diurnal_pattern(self, hour):
        primary = np.exp(-((hour - 14.0) ** 2) / 18.0)
        secondary = 0.5 * np.exp(-((hour - 10.0) ** 2) / 8.0)
        return 0.3 + 3.0 * (primary + secondary)

    def generate_trace(self):
        """生成含周期性突发的trace"""
        trace = np.zeros(self.num_steps, dtype=np.int32)

        burst_starts = list(range(
            self.rng.integers(0, self.burst_interval_steps),
            self.num_steps,
            self.burst_interval_steps
        ))

        for step in range(self.num_steps):
            hour = (step * self.dt_minutes / 60.0) % 24.0
            rate = self.base_rate * self._diurnal_pattern(hour) * self.dt_minutes

            in_burst = any(
                bs <= step < bs + self.burst_duration_steps
                for bs in burst_starts
            )
            if in_burst:
                rate *= self.burst_magnitude

            n_param = 3.0
            p_param = n_param / (n_param + max(rate, 0.1))
            arrivals = int(self.rng.negative_binomial(n_param, p_param))
            trace[step] = max(0, arrivals)
        return trace
