import numpy as np


class RoundRobinScheduler:
    """轮询调度启发式基线

    请求分配: 严格轮询
    频率控制: 基于队列长度的阈值规则
    """

    def __init__(self, num_servers=3):
        self.num_servers = num_servers
        self.current_server = 0

    def select_actions(self, local_states, **kwargs):
        """基于队列阈值的频率控制"""
        actions = []
        for state in local_states:
            queue_ratio = state[0]
            if queue_ratio < 0.2:
                actions.append(0)
            elif queue_ratio < 0.5:
                actions.append(1)
            else:
                actions.append(2)
        return actions

    def distribute_requests(self, arrivals):
        """严格轮询分配"""
        distribution = [0] * self.num_servers
        for _ in range(arrivals):
            distribution[self.current_server] += 1
            self.current_server = (self.current_server + 1) % self.num_servers
        return distribution
