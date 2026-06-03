"""
多智能体DQN异构集群节能调度 - 算法对比实验
==============================================
对比三种方法:
1. 集中式多智能体DQN (Centralized MADQN) - 共享经验池
2. 独立DQN (Independent DQN) - 各自缓冲区
3. 轮询启发式 (Round-Robin Heuristic)

指标: 总能耗(kWh), 平均响应时间
输出: 训练曲线, 对比柱状图, Pareto前沿, 集群动态图
"""

import numpy as np
import matplotlib.pyplot as plt
from workload_trace import AlibabaTraceGenerator
from cluster_env import ClusterEnv
from multi_agent_dqn import CentralizedMADQN, IndependentDQN
from baselines import RoundRobinScheduler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def train_centralized_madqn(trace, num_episodes=3000, energy_weight=0.5,
                            log_interval=200, verbose=True, warm_start_from=None):
    env = ClusterEnv(trace=trace, energy_weight=energy_weight, seed=42)
    agent = CentralizedMADQN(
        joint_state_dim=env.joint_state_dim,
        num_servers=env.num_servers,
        action_per_server=env.action_dim,
    )

    if warm_start_from is not None:
        agent.policy_net.load_state_dict(warm_start_from.policy_net.state_dict())
        agent.target_net.load_state_dict(warm_start_from.target_net.state_dict())
        agent.steps_done = warm_start_from.steps_done
        agent.learn_steps = warm_start_from.learn_steps

    episode_rewards = []
    episode_losses = []

    for episode in range(num_episodes):
        obs = env.reset()
        joint_state = obs["joint_state"]
        total_reward = 0
        ep_losses = []
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

        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(ep_losses) if ep_losses else 0)

        if verbose and (episode + 1) % log_interval == 0:
            avg_r = np.mean(episode_rewards[-log_interval:])
            eps = agent.get_epsilon()
            print(f"  [MADQN] Episode {episode+1:4d} | Reward: {avg_r:7.1f} | Eps: {eps:.3f}")

    return agent, episode_rewards, episode_losses


def train_independent_dqn(trace, num_episodes=3000, energy_weight=0.5,
                          log_interval=200, verbose=True, warm_start_from=None):
    env = ClusterEnv(trace=trace, energy_weight=energy_weight, seed=42)
    agent = IndependentDQN(
        num_servers=env.num_servers,
        local_state_dim=env.local_state_dim,
        action_dim=env.action_dim,
    )

    if warm_start_from is not None:
        for i, src_agent in enumerate(warm_start_from.agents):
            agent.agents[i].policy_net.load_state_dict(src_agent.policy_net.state_dict())
            agent.agents[i].target_net.load_state_dict(src_agent.target_net.state_dict())
            agent.agents[i].steps_done = src_agent.steps_done
            agent.agents[i].learn_steps = src_agent.learn_steps

    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        local_states = obs["local_states"]
        total_reward = 0
        done = False

        while not done:
            actions = agent.select_actions(local_states, training=True)
            obs, rewards, done, info = env.step(actions)
            next_local_states = obs["local_states"]

            agent.store_transitions(local_states, actions, rewards["per_server"],
                                    next_local_states, done)
            agent.update()

            agent.steps_done += 1
            local_states = next_local_states
            total_reward += rewards["total"]

        episode_rewards.append(total_reward)

        if verbose and (episode + 1) % log_interval == 0:
            avg_r = np.mean(episode_rewards[-log_interval:])
            print(f"  [IDQN]  Episode {episode+1:4d} | Reward: {avg_r:7.1f}")

    return agent, episode_rewards


def evaluate_method(agent, trace, num_episodes=20, method_type="madqn"):
    """评估方法, 返回 (energy_list_kwh, response_time_list)"""
    energies = []
    response_times = []

    for ep in range(num_episodes):
        env = ClusterEnv(trace=trace, seed=ep + 100)
        obs = env.reset()
        total_energy_wh = 0.0
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

        energies.append(total_energy_wh / 1000.0)
        response_times.append(info["avg_response_time"])

    return energies, response_times


def generate_pareto_points(trace, test_trace, base_madqn=None, base_idqn=None, alphas=None):
    """通过不同energy_weight生成Pareto点, 复用已训练模型微调"""
    if alphas is None:
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    pareto_data = {"MADQN": [], "IDQN": []}

    for alpha in alphas:
        print(f"\n  Pareto alpha={alpha:.1f} 微调中...")

        madqn_agent, _, _ = train_centralized_madqn(
            trace, num_episodes=800, energy_weight=alpha, verbose=False,
            warm_start_from=base_madqn
        )
        energies, resp_times = evaluate_method(madqn_agent, test_trace, num_episodes=10, method_type="madqn")
        pareto_data["MADQN"].append((np.mean(energies), np.mean(resp_times), alpha))

        idqn_agent, _ = train_independent_dqn(
            trace, num_episodes=800, energy_weight=alpha, verbose=False,
            warm_start_from=base_idqn
        )
        energies, resp_times = evaluate_method(idqn_agent, test_trace, num_episodes=10, method_type="idqn")
        pareto_data["IDQN"].append((np.mean(energies), np.mean(resp_times), alpha))

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

    ax.plot(smooth(madqn_rewards), label='集中式MADQN', color='blue', linewidth=1.2)
    ax.plot(smooth(idqn_rewards), label='独立DQN', color='orange', linewidth=1.2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('累计奖励')
    ax.set_title('训练曲线对比 (滑动平均)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.close()
    print("  已保存: training_curves.png")


def plot_comparison_metrics(results):
    """柱状图对比能耗和响应时间"""
    methods = list(results.keys())
    energies = [np.mean(results[m][0]) for m in methods]
    resp_times = [np.mean(results[m][1]) for m in methods]
    energy_stds = [np.std(results[m][0]) for m in methods]
    resp_stds = [np.std(results[m][1]) for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
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
        alphas = [p[2] for p in points]

        color = '#2196F3' if method == 'MADQN' else '#FF9800'
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
    ax.set_title('Pareto前沿: 能耗 vs 响应时间')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pareto_frontier.png', dpi=150)
    plt.close()
    print("  已保存: pareto_frontier.png")


def plot_cluster_dynamics(agent, trace, method_type="madqn"):
    """绘制单episode集群动态时序图"""
    env = ClusterEnv(trace=trace, seed=999)
    obs = env.reset()
    done = False

    steps_data = {"powers": [], "queues": [], "freqs": [], "failures": []}

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

    time_axis = np.arange(len(steps_data["powers"])) * 5 / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    server_names = ['高性能', '标准', '低功耗']
    colors = ['#E53935', '#1E88E5', '#43A047']

    for i in range(3):
        queues = [s[i] for s in steps_data["queues"]]
        axes[0].plot(time_axis, queues, label=server_names[i], color=colors[i], linewidth=1)
    axes[0].axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='SLA阈值')
    axes[0].set_ylabel('队列长度')
    axes[0].set_title('集群动态 (集中式MADQN)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    for i in range(3):
        powers = [s[i] for s in steps_data["powers"]]
        axes[1].plot(time_axis, powers, label=server_names[i], color=colors[i], linewidth=1)
    axes[1].set_ylabel('功耗 (W)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    for i in range(3):
        freqs = [s[i] for s in steps_data["freqs"]]
        axes[2].plot(time_axis, freqs, label=server_names[i], color=colors[i],
                     linewidth=1, drawstyle='steps-post')
    axes[2].set_ylabel('频率等级')
    axes[2].set_xlabel('时间 (小时)')
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(['低频', '中频', '高频'])
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cluster_dynamics.png', dpi=150)
    plt.close()
    print("  已保存: cluster_dynamics.png")


def main():
    print("=" * 70)
    print("多智能体DQN异构集群节能调度 - 算法对比实验")
    print("=" * 70)
    print(f"集群配置: 3台异构服务器 (高性能/标准/低功耗)")
    print(f"环境特性: 通信延迟(1步), 随机故障, 阿里巴巴trace负载")
    print(f"对比算法: 集中式MADQN vs 独立DQN vs 轮询启发式")
    print("=" * 70)

    print("\n[1/6] 生成阿里巴巴风格工作负载trace...")
    trace_gen = AlibabaTraceGenerator(duration_hours=24, seed=42)
    train_trace = trace_gen.generate_trace()
    test_trace_gen = AlibabaTraceGenerator(duration_hours=24, seed=123)
    test_trace = test_trace_gen.generate_trace()
    print(f"  训练trace: {len(train_trace)}步, 总请求: {train_trace.sum()}")
    print(f"  测试trace: {len(test_trace)}步, 总请求: {test_trace.sum()}")

    print("\n[2/6] 训练集中式MADQN (共享经验池)...")
    madqn_agent, madqn_rewards, _ = train_centralized_madqn(
        train_trace, num_episodes=3000
    )

    print("\n[3/6] 训练独立DQN (各自缓冲区)...")
    idqn_agent, idqn_rewards = train_independent_dqn(
        train_trace, num_episodes=3000
    )

    print("\n[4/6] 评估三种方法...")
    rr_agent = RoundRobinScheduler(num_servers=3)

    results = {}
    madqn_e, madqn_r = evaluate_method(madqn_agent, test_trace, num_episodes=20, method_type="madqn")
    results["集中式MADQN"] = (madqn_e, madqn_r)
    print(f"  MADQN  - 能耗: {np.mean(madqn_e):.3f} kWh, 响应: {np.mean(madqn_r):.2f} min")

    idqn_e, idqn_r = evaluate_method(idqn_agent, test_trace, num_episodes=20, method_type="idqn")
    results["独立DQN"] = (idqn_e, idqn_r)
    print(f"  IDQN   - 能耗: {np.mean(idqn_e):.3f} kWh, 响应: {np.mean(idqn_r):.2f} min")

    rr_e, rr_r = evaluate_method(rr_agent, test_trace, num_episodes=20, method_type="rr")
    results["轮询启发式"] = (rr_e, rr_r)
    print(f"  轮询   - 能耗: {np.mean(rr_e):.3f} kWh, 响应: {np.mean(rr_r):.2f} min")

    print("\n[5/6] 生成Pareto前沿数据 (基于已训练模型微调)...")
    pareto_data = generate_pareto_points(train_trace, test_trace,
                                         base_madqn=madqn_agent, base_idqn=idqn_agent)
    rr_point = (np.mean(rr_e), np.mean(rr_r))

    print("\n[6/6] 绘制结果图表...")
    plot_training_curves(madqn_rewards, idqn_rewards)
    plot_comparison_metrics(results)
    plot_pareto_frontier(pareto_data, rr_point)
    plot_cluster_dynamics(madqn_agent, test_trace, method_type="madqn")

    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
    print("输出文件:")
    print("  - training_curves.png     : 训练曲线对比")
    print("  - comparison_metrics.png  : 能耗/响应时间柱状图")
    print("  - pareto_frontier.png     : Pareto前沿图")
    print("  - cluster_dynamics.png    : 集群动态时序图")


if __name__ == "__main__":
    main()
