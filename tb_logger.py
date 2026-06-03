from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """TensorBoard训练可视化记录器

    记录指标:
    - episode/reward: 每轮累计奖励
    - episode/loss: 每轮平均损失
    - episode/epsilon: 探索率
    - episode/total_power: 总功耗
    - episode/avg_queue: 平均队列长度
    - episode/max_queue: 最大队列长度
    - episode/migrations: 迁移次数
    - episode/overload_steps: 过载步数
    """

    def __init__(self, log_dir="runs/cluster_ddqn"):
        self.writer = SummaryWriter(log_dir)

    def log_episode(self, episode: int, metrics: dict):
        """记录每轮训练指标"""
        for key, value in metrics.items():
            if value is not None:
                self.writer.add_scalar(f"episode/{key}", value, episode)

    def log_step(self, global_step: int, metrics: dict):
        """记录单步训练指标 (用于细粒度跟踪)"""
        for key, value in metrics.items():
            if value is not None:
                self.writer.add_scalar(f"step/{key}", value, global_step)

    def log_heatmap(self, tag: str, figure, step: int):
        """将matplotlib figure作为图片记录到TensorBoard"""
        self.writer.add_figure(tag, figure, step)

    def log_histogram(self, tag: str, values, step: int):
        """记录参数分布直方图"""
        self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int):
        """记录文本信息"""
        self.writer.add_text(tag, text, step)

    def close(self):
        self.writer.close()
