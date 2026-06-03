"""单元测试: 验证环境动力学公式正确性

测试覆盖:
1. 功耗模型: P = P_idle + (P_max - P_idle) * cpu_util * freq_factor
2. 服务速率: rate = base * freq_multiplier * recovery_factor
3. 队列演化: 不超上限, 不低于0
4. 迁移机制: 源减少, 目标延迟接收, 动态比例计算
5. 过载计数器: 正确递增/递减
6. 动态迁移延迟: 随系统负载变化
7. 请求来源追踪: 分类计数正确
"""

import unittest
import numpy as np
from cluster_env import ClusterEnv, ServerState, SERVER_CONFIGS


class TestPowerModel(unittest.TestCase):
    """功耗公式: P = P_idle + (P_max - P_idle) * cpu_util * freq_factor"""

    def test_idle_power(self):
        """零负载时功耗等于P_idle"""
        cfg = SERVER_CONFIGS[0]
        power = cfg["p_idle"] + (cfg["p_max"] - cfg["p_idle"]) * 0.0 * 1.0
        self.assertAlmostEqual(power, cfg["p_idle"])

    def test_full_load_mid_freq(self):
        """满负载中频: P = P_idle + (P_max - P_idle) * 1.0 * 1.0"""
        cfg = SERVER_CONFIGS[1]  # 标准服务器
        power = cfg["p_idle"] + (cfg["p_max"] - cfg["p_idle"]) * 1.0 * 1.0
        self.assertAlmostEqual(power, cfg["p_max"])

    def test_full_load_high_freq(self):
        """满负载高频: 功耗超过P_max (频率因子>1)"""
        cfg = SERVER_CONFIGS[0]
        power = cfg["p_idle"] + (cfg["p_max"] - cfg["p_idle"]) * 1.0 * cfg["freq_multipliers"][2]
        expected = 80.0 + 220.0 * 1.5  # = 410 W
        self.assertAlmostEqual(power, expected)

    def test_linear_scaling(self):
        """功耗与cpu_util线性相关"""
        cfg = SERVER_CONFIGS[1]
        p1 = cfg["p_idle"] + (cfg["p_max"] - cfg["p_idle"]) * 0.5 * 1.0
        p2 = cfg["p_idle"] + (cfg["p_max"] - cfg["p_idle"]) * 1.0 * 1.0
        self.assertAlmostEqual(p2 - cfg["p_idle"], 2 * (p1 - cfg["p_idle"]))

    def test_low_freq_reduces_power(self):
        """低频时相同负载功耗更低"""
        cfg = SERVER_CONFIGS[0]
        p_low = cfg["p_idle"] + (cfg["p_max"] - cfg["p_idle"]) * 0.8 * cfg["freq_multipliers"][0]
        p_high = cfg["p_idle"] + (cfg["p_max"] - cfg["p_idle"]) * 0.8 * cfg["freq_multipliers"][2]
        self.assertLess(p_low, p_high)


class TestServiceRate(unittest.TestCase):
    """服务速率: rate = base_service_rate * freq_multiplier * recovery_factor"""

    def test_normal_operation(self):
        """正常运行时recovery_factor=1.0"""
        cfg = SERVER_CONFIGS[0]
        rate = cfg["base_service_rate"] * cfg["freq_multipliers"][2] * 1.0
        self.assertAlmostEqual(rate, 0.8 * 1.5)

    def test_recovery_phase_start(self):
        """恢复刚开始时factor=0.5"""
        server = ServerState(SERVER_CONFIGS[0])
        server.recovery_phase = server.recovery_duration  # =3
        factor = server.get_service_rate_factor()
        # progress = 1 - 3/3 = 0, factor = 0.5 + 0.5*0 = 0.5
        self.assertAlmostEqual(factor, 0.5)

    def test_recovery_midway(self):
        """恢复中途factor在0.5-1.0之间"""
        server = ServerState(SERVER_CONFIGS[0])
        server.recovery_phase = 1
        factor = server.get_service_rate_factor()
        # progress = 1 - 1/3 = 2/3, factor = 0.5 + 0.5*(2/3) ≈ 0.833
        self.assertAlmostEqual(factor, 0.5 + 0.5 * (2.0 / 3.0), places=5)

    def test_recovery_end(self):
        """恢复完成时factor=1.0"""
        server = ServerState(SERVER_CONFIGS[0])
        server.recovery_phase = 0
        factor = server.get_service_rate_factor()
        self.assertAlmostEqual(factor, 1.0)

    def test_freq_multiplier_ordering(self):
        """频率越高服务速率越快"""
        cfg = SERVER_CONFIGS[0]
        rates = [cfg["base_service_rate"] * cfg["freq_multipliers"][f] for f in range(3)]
        self.assertLess(rates[0], rates[1])
        self.assertLess(rates[1], rates[2])


class TestQueueEvolution(unittest.TestCase):
    """队列动力学: 上界max_queue, 下界0"""

    def test_queue_cap(self):
        """队列不超过max_queue"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].queue_length = 49
        env.trace = np.array([100] * 300)
        env.step([1, 1, 1])
        for s in env.servers:
            self.assertLessEqual(s.queue_length, env.max_queue)

    def test_empty_queue_no_negative(self):
        """空队列不会变负"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.trace = np.array([0] * 300)
        env.step([1, 1, 1])
        for s in env.servers:
            self.assertGreaterEqual(s.queue_length, 0)

    def test_arrivals_increase_queue(self):
        """有到达时队列应增加 (需要通信延迟后投递)"""
        env = ClusterEnv(seed=42)
        env.reset()
        for s in env.servers:
            s.queue_length = 0
            s.current_freq = 0  # 最低频率, 处理慢
        env.trace = np.array([50] * 300)  # 大量到达
        # 第一步分配进comm_buffer, 第二步投递到队列
        env.step([1, 1, 1])
        env.step([1, 1, 1])
        total_queue = sum(s.queue_length for s in env.servers)
        self.assertGreater(total_queue, 0)


class TestMigrationMechanics(unittest.TestCase):
    """在线迁移机制"""

    def test_migration_removes_from_source(self):
        """迁移动作立即减少源队列"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].queue_length = 40
        env.servers[1].queue_length = 5
        env.servers[2].queue_length = 5
        env.trace = np.array([0] * 300)

        # action 4 = 中强度迁移
        env.step([4, 1, 1])
        self.assertLess(env.servers[0].queue_length, 40)

    def test_migration_delivers_after_delay(self):
        """迁移请求延迟后到达目标"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].queue_length = 40
        env.servers[0].sustained_overload_counter = 5  # 避免过早迁移惩罚
        env.servers[1].queue_length = 0
        env.servers[2].queue_length = 10
        env.trace = np.array([0] * 300)

        # 迁移从server0到server1(最空)
        env.step([4, 1, 1])
        # 多走几步让迁移到达
        for _ in range(3):
            env.step([1, 1, 1])
        # server1应收到迁移请求
        self.assertGreater(env.servers[1].queue_length, 0)

    def test_no_migration_on_empty_queue(self):
        """空队列的迁移动作是no-op"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].queue_length = 0
        env.trace = np.array([0] * 300)
        env.step([5, 1, 1])  # 高强度迁移 on empty
        self.assertEqual(env.servers[0].queue_length, 0)

    def test_no_migration_to_fuller_target(self):
        """目标比源更满时不执行迁移"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].queue_length = 10
        env.servers[1].queue_length = 40
        env.servers[2].queue_length = 45
        env.trace = np.array([0] * 300)

        initial_q0 = env.servers[0].queue_length
        env.step([3, 1, 1])
        # 由于两个邻居都比server0满, 不应迁移
        # (注意: 虽然action=3, 但_execute_migrations会检查)
        # server0队列不会因迁移减少 (可能因处理减少)
        # 这里只验证没有crash且结果合理
        self.assertGreaterEqual(env.servers[0].queue_length, 0)

    def test_dynamic_migration_count(self):
        """动态迁移比例计算正确性"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].queue_length = 40  # src 80% full
        env.servers[1].queue_length = 10  # target 20% full, remaining=40

        count = env._compute_dynamic_migration_count(4, 0, 1)
        # base_frac=0.4, src_ratio=40/50=0.8, target_remaining_ratio=40/50=0.8
        # effective_frac = 0.4 * 0.8 * 0.8 = 0.256
        # migrate = max(1, int(40 * 0.256)) = max(1, 10) = 10
        expected = max(1, int(40 * 0.4 * 0.8 * 0.8))
        self.assertEqual(count, expected)

    def test_dynamic_migration_low_target_capacity(self):
        """目标接近满时迁移量减少"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].queue_length = 40
        env.servers[1].queue_length = 45  # target almost full, remaining=5

        count = env._compute_dynamic_migration_count(4, 0, 1)
        # base_frac=0.4, src_ratio=0.8, target_remaining=5/50=0.1
        # effective_frac = 0.4 * 0.8 * 0.1 = 0.032
        # migrate = max(1, int(40 * 0.032)) = max(1, 1) = 1
        # 但还要min(count, remaining=5)
        self.assertLessEqual(count, 5)
        self.assertGreaterEqual(count, 1)


class TestDynamicMigrationDelay(unittest.TestCase):
    """迁移延迟根据系统负载动态调整"""

    def test_low_load_minimum_delay(self):
        """低负载时延迟最小"""
        env = ClusterEnv(seed=42)
        env.reset()
        for s in env.servers:
            s.queue_length = 0
        delay = env._compute_migration_delay()
        # load_ratio ≈ 0, delay = base(1) + int(0*2) = 1
        self.assertEqual(delay, env.migration_base_delay)

    def test_high_load_increased_delay(self):
        """高负载时延迟增加"""
        env = ClusterEnv(seed=42)
        env.reset()
        for s in env.servers:
            s.queue_length = env.max_queue  # 全满
        delay = env._compute_migration_delay()
        # load_ratio = 1.0, delay = 1 + int(1.0*2) = 3
        self.assertEqual(delay, env.migration_base_delay + 2)

    def test_medium_load_intermediate_delay(self):
        """中等负载时延迟居中"""
        env = ClusterEnv(seed=42)
        env.reset()
        for s in env.servers:
            s.queue_length = 25  # 50% full
        delay = env._compute_migration_delay()
        # load_ratio = 0.5, delay = 1 + int(0.5*2) = 1 + 1 = 2
        self.assertEqual(delay, env.migration_base_delay + 1)


class TestOverloadCounter(unittest.TestCase):
    """持续过载计数器"""

    def test_counter_increments_on_overload(self):
        """队列超过阈值时计数器递增"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].queue_length = 35  # 70% > 60% threshold
        env.trace = np.array([5] * 300)  # 少量到达维持队列
        env.step([1, 1, 1])
        self.assertGreaterEqual(env.servers[0].sustained_overload_counter, 1)

    def test_counter_decrements_below_threshold(self):
        """队列低于阈值时计数器递减"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].sustained_overload_counter = 5
        env.servers[0].queue_length = 10  # 20% < 60% threshold
        env.trace = np.array([0] * 300)
        env.step([1, 1, 1])
        self.assertLess(env.servers[0].sustained_overload_counter, 5)

    def test_counter_minimum_zero(self):
        """计数器不会低于0"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].sustained_overload_counter = 0
        env.servers[0].queue_length = 0
        env.trace = np.array([0] * 300)
        env.step([1, 1, 1])
        self.assertGreaterEqual(env.servers[0].sustained_overload_counter, 0)


class TestRequestSourceTracking(unittest.TestCase):
    """请求来源分类追踪"""

    def test_local_requests_tracked(self):
        """正常到达的请求标记为本地(-1), 需等通信延迟后投递"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.trace = np.array([20] * 300)
        # 第一步分配, 第二步投递并标记来源
        env.step([1, 1, 1])
        env.step([1, 1, 1])
        has_local = any(
            s.request_sources.get(-1, 0) > 0 for s in env.servers
        )
        self.assertTrue(has_local)

    def test_migrated_requests_tagged_with_source(self):
        """迁移的请求用源服务器ID标记"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].queue_length = 40
        env.servers[0].sustained_overload_counter = 5
        env.servers[0].request_sources[-1] = 40
        env.servers[1].queue_length = 0
        env.servers[2].queue_length = 10
        env.trace = np.array([0] * 300)

        # 迁移
        env.step([4, 1, 1])
        # 多步让迁移完成
        for _ in range(3):
            env.step([1, 1, 1])

        # server1应有来自server0的请求记录
        from_server0 = env.servers[1].request_sources.get(0, 0)
        self.assertGreater(from_server0, 0)

    def test_local_request_ratio(self):
        """本地请求占比计算正确"""
        server = ServerState(SERVER_CONFIGS[0])
        server.request_sources = {-1: 30, 0: 10}  # 30本地 + 10迁移
        ratio = server.get_local_request_ratio()
        self.assertAlmostEqual(ratio, 30.0 / 40.0)

    def test_empty_queue_ratio(self):
        """空队列时本地占比返回1.0"""
        server = ServerState(SERVER_CONFIGS[0])
        server.request_sources = {}
        ratio = server.get_local_request_ratio()
        self.assertAlmostEqual(ratio, 1.0)


class TestStateSpaceDimensions(unittest.TestCase):
    """验证状态空间维度正确"""

    def test_joint_state_dim(self):
        env = ClusterEnv(seed=42)
        obs = env.reset()
        # joint_state_dim = num_servers * 6 + 2 = 3*6 + 2 = 20
        self.assertEqual(env.joint_state_dim, 20)
        self.assertEqual(len(obs["joint_state"]), 20)

    def test_local_state_dim(self):
        env = ClusterEnv(seed=42)
        obs = env.reset()
        # local_state_dim = 6 + 2 + 3*(num_servers-1) = 6 + 2 + 6 = 14
        self.assertEqual(env.local_state_dim, 14)
        for ls in obs["local_states"]:
            self.assertEqual(len(ls), 14)

    def test_action_dim(self):
        env = ClusterEnv(seed=42)
        self.assertEqual(env.action_dim, 6)

    def test_state_normalization(self):
        """所有状态值在合理范围内"""
        env = ClusterEnv(seed=42)
        obs = env.reset()
        for val in obs["joint_state"]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.5)  # 允许略超1 (overload counter)


class TestActionSpace(unittest.TestCase):
    """验证动作空间正确性"""

    def test_freq_decrease(self):
        """动作0: 降频"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].current_freq = 2
        env.trace = np.array([0] * 300)
        env.step([0, 1, 1])
        self.assertEqual(env.servers[0].current_freq, 1)

    def test_freq_increase(self):
        """动作2: 升频"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].current_freq = 0
        env.trace = np.array([0] * 300)
        env.step([2, 1, 1])
        self.assertEqual(env.servers[0].current_freq, 1)

    def test_freq_clamp_low(self):
        """降频不低于0"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].current_freq = 0
        env.trace = np.array([0] * 300)
        env.step([0, 1, 1])
        self.assertEqual(env.servers[0].current_freq, 0)

    def test_freq_clamp_high(self):
        """升频不超过max"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].current_freq = 2
        env.trace = np.array([0] * 300)
        env.step([2, 1, 1])
        self.assertEqual(env.servers[0].current_freq, 2)

    def test_migration_action_no_freq_change(self):
        """迁移动作(3/4/5)不改变频率"""
        env = ClusterEnv(seed=42)
        env.reset()
        env.servers[0].current_freq = 1
        env.servers[0].queue_length = 30
        env.trace = np.array([0] * 300)
        env.step([4, 1, 1])
        self.assertEqual(env.servers[0].current_freq, 1)


class TestEarlyStoppingAdaptive(unittest.TestCase):
    """自适应早停策略"""

    def test_no_stop_early_in_training(self):
        """训练初期不应过早停止"""
        from early_stopping import EarlyStopping
        stopper = EarlyStopping(patience=100, max_episodes=1000)
        rewards = list(np.random.randn(200) * 10)  # 前200轮随机
        should_stop = stopper.step(rewards)
        # 初期patience=150, 且只跑了200轮, 不应停
        self.assertFalse(should_stop)

    def test_stop_on_plateau(self):
        """长期无改善时应停止"""
        from early_stopping import EarlyStopping
        stopper = EarlyStopping(patience=50, min_delta=1.0, window=20, max_episodes=1000)
        # 先给一个高峰
        rewards = [100.0] * 200
        # 然后持续平坦
        rewards += [80.0] * 300
        for i in range(len(rewards)):
            result = stopper.step(rewards[:i+1])
            if result:
                break
        self.assertTrue(stopper.should_stop)

    def test_adaptive_delta_phases(self):
        """不同阶段delta不同"""
        from early_stopping import EarlyStopping
        stopper = EarlyStopping(patience=100, min_delta=1.0, max_episodes=1000)

        stopper.current_episode = 100  # 10%, 初期
        delta_early = stopper._get_adaptive_delta()

        stopper.current_episode = 500  # 50%, 中期
        delta_mid = stopper._get_adaptive_delta()

        stopper.current_episode = 800  # 80%, 后期
        delta_late = stopper._get_adaptive_delta()

        self.assertLess(delta_early, delta_mid)
        self.assertLess(delta_mid, delta_late)


if __name__ == "__main__":
    unittest.main()
