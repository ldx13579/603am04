import numpy as np


class EarlyStopping:
    """自适应早停策略

    根据训练进度动态调整早停阈值:
    - 训练初期(前30%): 宽松阈值 (min_delta * 0.3), 允许探索
    - 训练中期(30%-70%): 标准阈值 (min_delta)
    - 训练后期(70%+): 严格阈值 (min_delta * 2.0), 快速收敛或停止

    Parameters
    ----------
    patience : int
        无改善时的最大容忍轮数 (也随阶段动态调整)
    min_delta : float
        基准最小改善量
    window : int
        平滑窗口大小
    max_episodes : int
        预期最大训练轮数 (用于计算训练进度)
    """

    def __init__(self, patience: int = 300, min_delta: float = 0.5,
                 window: int = 100, max_episodes: int = 3000):
        self.base_patience = patience
        self.base_min_delta = min_delta
        self.window = window
        self.max_episodes = max_episodes

        self.best_avg = -np.inf
        self.episodes_without_improvement = 0
        self.should_stop = False
        self.current_episode = 0

    def _get_progress_ratio(self):
        """当前训练进度 [0, 1]"""
        return min(1.0, self.current_episode / max(self.max_episodes, 1))

    def _get_adaptive_delta(self):
        """根据训练阶段动态调整min_delta"""
        progress = self._get_progress_ratio()
        if progress < 0.3:
            return self.base_min_delta * 0.3
        elif progress < 0.7:
            return self.base_min_delta
        else:
            return self.base_min_delta * 2.0

    def _get_adaptive_patience(self):
        """根据训练阶段动态调整patience"""
        progress = self._get_progress_ratio()
        if progress < 0.3:
            return int(self.base_patience * 1.5)
        elif progress < 0.7:
            return self.base_patience
        else:
            return int(self.base_patience * 0.6)

    def step(self, episode_rewards: list) -> bool:
        """每轮结束后调用, 返回是否应停止训练"""
        self.current_episode = len(episode_rewards)

        if self.current_episode < self.window:
            return False

        current_avg = np.mean(episode_rewards[-self.window:])
        adaptive_delta = self._get_adaptive_delta()
        adaptive_patience = self._get_adaptive_patience()

        if current_avg > self.best_avg + adaptive_delta:
            self.best_avg = current_avg
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1

        self.should_stop = self.episodes_without_improvement >= adaptive_patience
        return self.should_stop

    def status(self) -> str:
        progress = self._get_progress_ratio()
        phase = "初期" if progress < 0.3 else ("中期" if progress < 0.7 else "后期")
        return (f"EarlyStopping[{phase}]: best_avg={self.best_avg:.2f}, "
                f"no_improve={self.episodes_without_improvement}/{self._get_adaptive_patience()}, "
                f"delta={self._get_adaptive_delta():.3f}")
