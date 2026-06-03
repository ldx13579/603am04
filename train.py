import numpy as np
import matplotlib.pyplot as plt
from env import ServerEnv
from dqn import DQNAgent


def train(num_episodes=5000, log_interval=100):
    env = ServerEnv()
    agent = DQNAgent(
        state_dim=4,
        action_dim=3,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=300,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=200
    )

    episode_rewards = []
    episode_powers = []
    episode_queues = []
    losses = []

    print("=" * 60)
    print("DQN云服务器动态节能调度 - 训练开始")
    print("=" * 60)
    print(f"训练轮数: {num_episodes}")
    print(f"状态空间: [队列长度, 10分钟平均负载, 时间, 当前频率等级]")
    print(f"动作空间: [降频, 维持, 升频]")
    print(f"奖励: -功耗(归一化) + SLA惩罚 + 频率切换惩罚")
    print("=" * 60)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        total_power = 0
        max_queue = 0
        step_count = 0

        done = False
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            agent.steps_done += 1

            state = next_state
            total_reward += reward
            total_power += info["power"]
            max_queue = max(max_queue, info["queue"])
            step_count += 1

        # 目标网络已在agent.update()中按训练步数自动更新

        episode_rewards.append(total_reward)
        episode_powers.append(total_power / step_count)
        episode_queues.append(max_queue)

        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_power = np.mean(episode_powers[-log_interval:])
            avg_queue = np.mean(episode_queues[-log_interval:])
            epsilon = agent.get_epsilon()
            print(f"Episode {episode + 1:5d} | "
                  f"Reward: {avg_reward:7.1f} | "
                  f"Power: {avg_power:6.1f}W | "
                  f"MaxQueue: {avg_queue:5.1f} | "
                  f"Epsilon: {epsilon:.3f}")

    agent.save("dqn_server_model.pth")
    print("\n模型已保存: dqn_server_model.pth")

    plot_results(episode_rewards, episode_powers, losses)
    return agent


def plot_results(rewards, powers, losses):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # 奖励曲线
    window = 100
    smoothed_rewards = []
    for i in range(len(rewards)):
        start = max(0, i - window)
        smoothed_rewards.append(np.mean(rewards[start:i + 1]))

    axes[0].plot(smoothed_rewards, color='blue', linewidth=0.8)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Training Reward Curve (smoothed)')
    axes[0].grid(True, alpha=0.3)

    # 功耗曲线
    smoothed_powers = []
    for i in range(len(powers)):
        start = max(0, i - window)
        smoothed_powers.append(np.mean(powers[start:i + 1]))

    axes[1].plot(smoothed_powers, color='red', linewidth=0.8)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Average Power (W)')
    axes[1].set_title('Average Power Consumption')
    axes[1].grid(True, alpha=0.3)

    # 损失曲线
    if losses:
        loss_window = 500
        smoothed_losses = []
        for i in range(len(losses)):
            start = max(0, i - loss_window)
            smoothed_losses.append(np.mean(losses[start:i + 1]))
        axes[2].plot(smoothed_losses, color='green', linewidth=0.8)
        axes[2].set_xlabel('Training Step')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('DQN Training Loss')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.close()
    print("训练曲线已保存: training_results.png")


def evaluate(agent, num_episodes=10):
    """评估训练好的模型"""
    env = ServerEnv()
    print("\n" + "=" * 60)
    print("评估阶段")
    print("=" * 60)

    total_rewards = []
    total_powers = []
    freq_counts = {0: 0, 1: 0, 2: 0}

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        ep_power = 0
        steps = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_power += info["power"]
            freq_counts[info["freq"]] += 1
            steps += 1

        total_rewards.append(ep_reward)
        total_powers.append(ep_power / steps)

    print(f"平均奖励: {np.mean(total_rewards):.2f} (+/- {np.std(total_rewards):.2f})")
    print(f"平均功耗: {np.mean(total_powers):.1f}W")

    total_freq = sum(freq_counts.values())
    print(f"频率分布 - 低频: {freq_counts[0]/total_freq*100:.1f}% | "
          f"中频: {freq_counts[1]/total_freq*100:.1f}% | "
          f"高频: {freq_counts[2]/total_freq*100:.1f}%")

    # 与基线对比(始终中频)
    baseline_powers = []
    for _ in range(num_episodes):
        state = env.reset()
        bp = 0
        steps = 0
        done = False
        while not done:
            state, _, done, info = env.step(1)  # 始终维持
            bp += info["power"]
            steps += 1
        baseline_powers.append(bp / steps)

    saving = (np.mean(baseline_powers) - np.mean(total_powers)) / np.mean(baseline_powers) * 100
    print(f"\n对比基线(固定中频): {np.mean(baseline_powers):.1f}W")
    print(f"节能效果: {saving:.1f}%")


if __name__ == "__main__":
    agent = train(num_episodes=5000)
    evaluate(agent)
