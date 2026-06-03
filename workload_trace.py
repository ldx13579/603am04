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
