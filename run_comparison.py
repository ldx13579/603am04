"""
多智能体Double DQN异构集群节能调度 - 含在线迁移决策
==============================================
对比三种方法:
1. 集中式多智能体Double DQN (Centralized MADDQN) - 共享经验池 + 在线迁移
2. 独立Double DQN (Independent DDQN) - 各自缓冲区 + 在线迁移
3. 轮询启发式 (Round-Robin Heuristic)

新增特性:
- 动态迁移比例 (基于源队列和目标容量)
- 请求来源分类追踪
- 动态迁移延迟 (基于系统负载)
- 同步软更新目标网络
- 自适应早停 (初期宽松/后期严格)
- TensorBoard可视化
- 策略热力图

指标: 总能耗(kWh), 平均响应时间, 迁移次数
输出: 训练曲线, 对比柱状图, Pareto前沿, 集群动态图, 策略热力图
"""

import numpy as np
import matplotlib.pyplot as plt
from workload_trace import AlibabaTraceGenerator
from cluster_env import ClusterEnv
from multi_agent_dqn import CentralizedMADQN, IndependentDQN
from baselines import RoundRobinScheduler
from early_stopping import EarlyStopping
from tb_logger import TBLogger
from heatmap import generate_strategy_heatmap

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def train_centralized_madqn(trace, num_episodes=3000, energy_weight=0.5,
                            log_interval=200, verbose=True, warm_start_from=None,
                            use_tensorboard=True, early_stop_patience=300):
    env = ClusterEnv(trace=trace, energy_weight=energy_weight, seed=42)
    agent = CentralizedMADQN(
        joint_state_dim=env.joint_state_dim,
        num_servers=env.num_servers,
        action_per_server=env.action_dim,
    )

    if warm_start_from is not None:
        try:
            agent.policy_net.load_state_dict(warm_start_from.policy_net.state_dict())
            agent.target_net.load_state_dict(warm_start_from.target_net.state_dict())
            agent.steps_done = warm_start_from.steps_done
            agent.learn_steps = warm_start_from.learn_steps
        except RuntimeError:
            pass  # 维度不匹配时跳过warm start

    logger = TBLogger("runs/maddqn") if use_tensorboard else None
    stopper = EarlyStopping(
        patience=early_stop_patience, min_delta=0.5,
        window=100, max_episodes=num_episodes
    )

    episode_rewards = []
    episode_losses = []

    for episode in range(num_episodes):
        obs = env.reset()
        joint_state = obs["joint_state"]
        total_reward = 0
        ep_losses = []
        ep_power = []
        ep_queues = []
        ep_migrations = 0
        done = False

        while not done:
            actions = agent.select_actions(joint_state, training=True)
            obs, rewards, done, info = env.step(actions)
            next_joint_state = obs["joint_state"]

            agent.store_transition(joint_state, actions, rewards["total"],
                                   next_joint_state, done)
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)

            agent.steps_done += 1
            joint_state = next_joint_state
            total_reward += rewards["total"]
            ep_power.append(info["total_power"])
            ep_queues.append(sum(info["queues"]) / len(info["queues"]))
            ep_migrations += info["migrations_this_step"]

        episode_rewards.append(total_reward)
        avg_loss = np.mean(ep_losses) if ep_losses else 0
        episode_losses.append(avg_loss)

        if logger:
            logger.log_episode(episode, {
                "reward": total_reward,
                "loss": avg_loss if ep_losses else None,
                "epsilon": agent.get_epsilon(),
                "total_power": np.mean(ep_power),
                "avg_queue": np.mean(ep_queues),
                "max_queue": max(max(info["queues"]) for info in [info]),
                "migrations": ep_migrations,
            })

        if verbose and (episode + 1) % log_interval == 0:
            avg_r = np.mean(episode_rewards[-log_interval:])
            eps = agent.get_epsilon()
            print(f"  [MADDQN] Episode {episode+1:4d} | "
                  f"Reward: {avg_r:7.1f} | Eps: {eps:.3f} | "
                  f"Migrations: {ep_migrations}")

        if stopper.step(episode_rewards):
            if verbose:
                print(f"  [MADDQN] 早停触发 @ Episode {episode+1}: {stopper.status()}")
            break

    if logger:
        logger.close()

    return agent, episode_rewards, episode_losses


def train_independent_dqn(trace, num_episodes=3000, energy_weight=0.5,
                          log_interval=200, verbose=True, warm_start_from=None,
                          use_tensorboard=True, early_stop_patience=300):
    env = ClusterEnv(trace=trace, energy_weight=energy_weight, seed=42)
    agent = IndependentDQN(
        num_servers=env.num_servers,
        local_state_dim=env.local_state_dim,
        action_dim=env.action_dim,
    )

    if warm_start_from is not None:
        try:
            for i, src_agent in enumerate(warm_start_from.agents):
                agent.agents[i].policy_net.load_state_dict(src_agent.policy_net.state_dict())
                agent.agents[i].target_net.load_state_dict(src_agent.target_net.state_dict())
                agent.agents[i].steps_done = src_agent.steps_done
                agent.agents[i].learn_steps = src_agent.learn_steps
        except RuntimeError:
            pass

    logger = TBLogger("runs/iddqn") if use_tensorboard else None
    stopper = EarlyStopping(
        patience=early_stop_patience, min_delta=0.5,
        window=100, max_episodes=num_episodes
    )

    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        local_states = obs["local_states"]
        total_reward = 0
        ep_losses = []
        ep_power = []
        ep_migrations = 0
        done = False

        while not done:
            actions = agent.select_actions(local_states, training=True)
            obs, rewards, done, info = env.step(actions)
            next_local_states = obs["local_states"]

            agent.store_transitions(local_states, actions, rewards["per_server"],
                                    next_local_states, done)
            losses = agent.update()
            valid_losses = [l for l in losses if l is not None]
            if valid_losses:
                ep_losses.append(np.mean(valid_losses))

            agent.steps_done += 1
            local_states = next_local_states
            total_reward += rewards["total"]
            ep_power.append(info["total_power"])
            ep_migrations += info["migrations_this_step"]

        episode_rewards.append(total_reward)

        if logger:
            logger.log_episode(episode, {
                "reward": total_reward,
                "loss": np.mean(ep_losses) if ep_losses else None,
                "epsilon": agent.agents[0].get_epsilon(),
                "total_power": np.mean(ep_power),
                "migrations": ep_migrations,
            })

        if verbose and (episode + 1) % log_interval == 0:
            avg_r = np.mean(episode_rewards[-log_interval:])
            print(f"  [IDDQN]  Episode {episode+1:4d} | "
                  f"Reward: {avg_r:7.1f} | Migrations: {ep_migrations}")

        if stopper.step(episode_rewards):
            if verbose:
                print(f"  [IDDQN] 早停触发 @ Episode {episode+1}: {stopper.status()}")
            break

    if logger:
        logger.close()

    return agent, episode_rewards


def evaluate_method(agent, trace, num_episodes=20, method_type="madqn"):
    """评估方法, 返回 (energy_list_kwh, response_time_list, migration_counts)"""
    energies = []
    response_times = []
    migration_counts = []

    for ep in range(num_episodes):
        env = ClusterEnv(trace=trace, seed=ep + 100)
        obs = env.reset()
        total_energy_wh = 0.0
        total_migrations = 0
        done = False

        while not done:
            if method_type == "madqn":
                actions = agent.select_actions(obs["joint_state"], training=False)
            elif method_type == "idqn":
                actions = agent.select_actions(obs["local_states"], training=False)
            elif method_type == "rr":
                actions = agent.select_actions(obs["local_states"])
            else:
                actions = [1, 1, 1]

            obs, rewards, done, info = env.step(actions)
            total_energy_wh += info["total_power"] * (5.0 / 60.0)
            total_migrations += info["migrations_this_step"]

        energies.append(total_energy_wh / 1000.0)
        response_times.append(info["avg_response_time"])
        migration_counts.append(total_migrations)

    return energies, response_times, migration_counts


def generate_pareto_points(trace, test_trace, base_madqn=None, base_idqn=None, alphas=None):
    """通过不同energy_weight生成Pareto点"""
    if alphas is None:
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    pareto_data = {"MADDQN": [], "IDDQN": []}

    for alpha in alphas:
        print(f"\n  Pareto alpha={alpha:.1f} 微调中...")

        madqn_agent, _, _ = train_centralized_madqn(
            trace, num_episodes=800, energy_weight=alpha, verbose=False,
            warm_start_from=base_madqn, use_tensorboard=False
        )
        energies, resp_times, _ = evaluate_method(
            madqn_agent, test_trace, num_episodes=10, method_type="madqn"
        )
        pareto_data["MADDQN"].append((np.mean(energies), np.mean(resp_times), alpha))

        idqn_agent, _ = train_independent_dqn(
            trace, num_episodes=800, energy_weight=alpha, verbose=False,
            warm_start_from=base_idqn, use_tensorboard=False
        )
        energies, resp_times, _ = evaluate_method(
            idqn_agent, test_trace, num_episodes=10, method_type="idqn"
        )
        pareto_data["IDDQN"].append((np.mean(energies), np.mean(resp_times), alpha))

    return pareto_data


def plot_training_curves(madqn_rewards, idqn_rewards):
    """训练曲线对比"""
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 100

    def smooth(data):
        result = []
        for i in range(len(data)):
            start = max(0, i - window)
            result.append(np.mean(data[start:i + 1]))
        return result

    ax.plot(smooth(madqn_rewards), label='集中式MADDQN', color='blue', linewidth=1.2)
    ax.plot(smooth(idqn_rewards), label='独立DDQN', color='orange', linewidth=1.2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('累计奖励')
    ax.set_title('Double DQN训练曲线对比 (含在线迁移, 滑动平均)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.close()
    print("  已保存: training_curves.png")


def plot_comparison_metrics(results):
    """柱状图对比能耗、响应时间和迁移次数"""
    methods = list(results.keys())
    energies = [np.mean(results[m][0]) for m in methods]
    resp_times = [np.mean(results[m][1]) for m in methods]
    migrations = [np.mean(results[m][2]) for m in methods]
    energy_stds = [np.std(results[m][0]) for m in methods]
    resp_stds = [np.std(results[m][1]) for m in methods]
    mig_stds = [np.std(results[m][2]) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(methods))
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    axes[0].bar(x, energies, yerr=energy_stds, color=colors, capsize=5, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel('总能耗 (kWh)')
    axes[0].set_title('总能耗对比')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, resp_times, yerr=resp_stds, color=colors, capsize=5, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel('平均响应时间 (min)')
    axes[1].set_title('平均响应时间对比')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(x, migrations, yerr=mig_stds, color=colors, capsize=5, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods)
    axes[2].set_ylabel('迁移次数')
    axes[2].set_title('平均迁移次数')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('comparison_metrics.png', dpi=150)
    plt.close()
    print("  已保存: comparison_metrics.png")


def plot_pareto_frontier(pareto_data, rr_point):
    """绘制Pareto前沿"""
    fig, ax = plt.subplots(figsize=(8, 6))

    for method, points in pareto_data.items():
        energies = [p[0] for p in points]
        resp_times = [p[1] for p in points]

        color = '#2196F3' if method == 'MADDQN' else '#FF9800'
        ax.scatter(resp_times, energies, color=color, s=80, zorder=5, label=method)

        sorted_pts = sorted(zip(resp_times, energies))
        ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                color=color, linestyle='--', alpha=0.6)

        for e, r, a in points:
            ax.annotate(f'α={a}', (r, e), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=color)

    ax.scatter(rr_point[1], rr_point[0], color='#4CAF50', s=120,
               marker='*', zorder=6, label='轮询基线')

    ax.set_xlabel('平均响应时间 (min)')
    ax.set_ylabel('总能耗 (kWh)')
    ax.set_title('Pareto前沿: 能耗 vs 响应时间 (Double DQN + 在线迁移)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pareto_frontier.png', dpi=150)
    plt.close()
    print("  已保存: pareto_frontier.png")


def plot_cluster_dynamics(agent, trace, method_type="madqn"):
    """绘制单episode集群动态时序图 (含迁移事件标注)"""
    env = ClusterEnv(trace=trace, seed=999)
    obs = env.reset()
    done = False

    steps_data = {"powers": [], "queues": [], "freqs": [],
                  "failures": [], "migrations": []}

    while not done:
        if method_type == "madqn":
            actions = agent.select_actions(obs["joint_state"], training=False)
        else:
            actions = agent.select_actions(obs["local_states"], training=False)

        obs, rewards, done, info = env.step(actions)
        steps_data["powers"].append(info["powers"])
        steps_data["queues"].append(info["queues"])
        steps_data["freqs"].append(info["freqs"])
        steps_data["failures"].append(info["failures"])
        steps_data["migrations"].append(info["migrations_this_step"])

    time_axis = np.arange(len(steps_data["powers"])) * 5 / 60.0

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    server_names = ['高性能', '标准', '低功耗']
    colors = ['#E53935', '#1E88E5', '#43A047']

    # 队列长度
    for i in range(3):
        queues = [s[i] for s in steps_data["queues"]]
        axes[0].plot(time_axis, queues, label=server_names[i], color=colors[i], linewidth=1)
    axes[0].axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='SLA阈值')
    axes[0].set_ylabel('队列长度')
    axes[0].set_title('集群动态 (集中式MADDQN + 在线迁移)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 功耗
    for i in range(3):
        powers = [s[i] for s in steps_data["powers"]]
        axes[1].plot(time_axis, powers, label=server_names[i], color=colors[i], linewidth=1)
    axes[1].set_ylabel('功耗 (W)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # 频率
    for i in range(3):
        freqs = [s[i] for s in steps_data["freqs"]]
        axes[2].plot(time_axis, freqs, label=server_names[i], color=colors[i],
                     linewidth=1, drawstyle='steps-post')
    axes[2].set_ylabel('频率等级')
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(['低频', '中频', '高频'])
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # 迁移事件
    axes[3].bar(time_axis, steps_data["migrations"], width=5/60.0*0.8,
                color='#9C27B0', alpha=0.7)
    axes[3].set_ylabel('迁移次数')
    axes[3].set_xlabel('时间 (小时)')
    axes[3].set_title('在线迁移触发时刻')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cluster_dynamics.png', dpi=150)
    plt.close()
    print("  已保存: cluster_dynamics.png")


def main():
    print("=" * 70)
    print("多智能体Double DQN异构集群节能调度 - 含在线迁移决策")
    print("=" * 70)
    print(f"集群配置: 3台异构服务器 (高性能/标准/低功耗)")
    print(f"环境特性: 通信延迟, 随机故障, 在线迁移(动态比例+动态延迟)")
    print(f"算法改进: Double DQN (减少过估计) + 软更新目标网络")
    print(f"对比算法: 集中式MADDQN vs 独立DDQN vs 轮询启发式")
    print(f"早停策略: 自适应阈值 (初期宽松/后期严格)")
    print("=" * 70)

    print("\n[1/7] 生成阿里巴巴风格工作负载trace...")
    trace_gen = AlibabaTraceGenerator(duration_hours=24, seed=42)
    train_trace = trace_gen.generate_trace()
    test_trace_gen = AlibabaTraceGenerator(duration_hours=24, seed=123)
    test_trace = test_trace_gen.generate_trace()
    print(f"  训练trace: {len(train_trace)}步, 总请求: {train_trace.sum()}")
    print(f"  测试trace: {len(test_trace)}步, 总请求: {test_trace.sum()}")

    print("\n[2/7] 训练集中式MADDQN (Double DQN + 在线迁移)...")
    madqn_agent, madqn_rewards, _ = train_centralized_madqn(
        train_trace, num_episodes=3000, use_tensorboard=True
    )

    print("\n[3/7] 训练独立DDQN (各自缓冲区 + 同步目标网络)...")
    idqn_agent, idqn_rewards = train_independent_dqn(
        train_trace, num_episodes=3000, use_tensorboard=True
    )

    print("\n[4/7] 评估三种方法...")
    rr_agent = RoundRobinScheduler(num_servers=3)

    results = {}
    madqn_e, madqn_r, madqn_m = evaluate_method(
        madqn_agent, test_trace, num_episodes=20, method_type="madqn"
    )
    results["集中式MADDQN"] = (madqn_e, madqn_r, madqn_m)
    print(f"  MADDQN - 能耗: {np.mean(madqn_e):.3f} kWh, "
          f"响应: {np.mean(madqn_r):.2f} min, 迁移: {np.mean(madqn_m):.0f}次")

    idqn_e, idqn_r, idqn_m = evaluate_method(
        idqn_agent, test_trace, num_episodes=20, method_type="idqn"
    )
    results["独立DDQN"] = (idqn_e, idqn_r, idqn_m)
    print(f"  IDDQN  - 能耗: {np.mean(idqn_e):.3f} kWh, "
          f"响应: {np.mean(idqn_r):.2f} min, 迁移: {np.mean(idqn_m):.0f}次")

    rr_e, rr_r, rr_m = evaluate_method(
        rr_agent, test_trace, num_episodes=20, method_type="rr"
    )
    results["轮询启发式"] = (rr_e, rr_r, rr_m)
    print(f"  轮询   - 能耗: {np.mean(rr_e):.3f} kWh, "
          f"响应: {np.mean(rr_r):.2f} min, 迁移: {np.mean(rr_m):.0f}次")

    print("\n[5/7] 生成Pareto前沿数据 (基于已训练模型微调)...")
    pareto_data = generate_pareto_points(train_trace, test_trace,
                                         base_madqn=madqn_agent, base_idqn=idqn_agent)
    rr_point = (np.mean(rr_e), np.mean(rr_r))

    print("\n[6/7] 绘制结果图表...")
    plot_training_curves(madqn_rewards, idqn_rewards)
    plot_comparison_metrics(results)
    plot_pareto_frontier(pareto_data, rr_point)
    plot_cluster_dynamics(madqn_agent, test_trace, method_type="madqn")

    print("\n[7/7] 生成策略热力图...")
    for s_idx in range(3):
        generate_strategy_heatmap(
            madqn_agent, server_idx=s_idx,
            num_servers=3, joint_state_dim=madqn_agent.policy_net.shared_encoder[0].in_features,
            output_path=f"strategy_heatmap_server{s_idx}.png"
        )

    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
    print("输出文件:")
    print("  - training_curves.png         : Double DQN训练曲线对比")
    print("  - comparison_metrics.png      : 能耗/响应时间/迁移次数柱状图")
    print("  - pareto_frontier.png         : Pareto前沿图")
    print("  - cluster_dynamics.png        : 集群动态时序图(含迁移标注)")
    print("  - strategy_heatmap_server*.png: 各服务器策略热力图")
    print("  - runs/                       : TensorBoard日志目录")
    print("\n使用 tensorboard --logdir=runs/ 查看训练曲线")


if __name__ == "__main__":
    main()
