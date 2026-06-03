"""策略热力图生成器

可视化不同负载模式下的智能体决策:
- X轴: 队列长度 (0-50)
- Y轴: 时间 (0-23h)
- 颜色: 选择的动作 (降频/维持/升频/迁移25%/迁移50%/迁移75%)

支持多种负载模式对比:
- 正常日间模式
- 持续高负载模式
- 突发负载模式
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

ACTION_LABELS = ["降频", "维持", "升频", "迁移25%", "迁移50%", "迁移75%"]


def generate_strategy_heatmap(agent, server_idx=0, num_servers=3,
                              joint_state_dim=20, max_queue=50, max_steps=288,
                              output_path="strategy_heatmap.png"):
    """生成单服务器策略热力图

    Parameters
    ----------
    agent : CentralizedMADQN
        训练好的智能体
    server_idx : int
        可视化哪台服务器的动作头
    """
    queue_bins = np.arange(0, max_queue + 1, 2)  # 0,2,...,50 (26 bins)
    hour_bins = np.arange(0, 24, 1)  # 0..23 (24 bins)

    heatmap = np.zeros((len(hour_bins), len(queue_bins)), dtype=np.int32)

    for hi, hour in enumerate(hour_bins):
        for qi, queue_len in enumerate(queue_bins):
            state = np.zeros(joint_state_dim, dtype=np.float32)
            time_of_day = hour / 24.0

            # 每服务器6个特征
            for s in range(num_servers):
                base = s * 6
                if s == server_idx:
                    state[base + 0] = queue_len / max_queue
                    state[base + 1] = queue_len / max_queue * 0.8
                    state[base + 2] = time_of_day
                    state[base + 3] = 0.5  # 中频
                    state[base + 4] = 0.0  # 无持续过载
                    state[base + 5] = 0.8  # 本地请求占比
                else:
                    state[base + 0] = 0.3
                    state[base + 1] = 0.3
                    state[base + 2] = time_of_day
                    state[base + 3] = 0.5
                    state[base + 4] = 0.0
                    state[base + 5] = 0.9

            # 全局特征
            global_base = num_servers * 6
            state[global_base] = (queue_len + 0.3 * max_queue * (num_servers - 1)) / (max_queue * num_servers)
            state[global_base + 1] = 1.0  # 全部活跃

            actions = agent.select_actions(state, training=False)
            heatmap[hi, qi] = actions[server_idx]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap, aspect='auto', origin='lower',
                   cmap=plt.cm.get_cmap('tab10', 6),
                   vmin=-0.5, vmax=5.5)

    ax.set_xticks(np.arange(0, len(queue_bins), 5))
    ax.set_xticklabels(queue_bins[::5])
    ax.set_yticks(np.arange(len(hour_bins)))
    ax.set_yticklabels(hour_bins)
    ax.set_xlabel('队列长度')
    ax.set_ylabel('时间 (小时)')
    server_names = ['高性能', '标准', '低功耗']
    ax.set_title(f'{server_names[server_idx]}服务器 策略热力图 (含迁移决策)')

    cbar = plt.colorbar(im, ax=ax, ticks=list(range(6)))
    cbar.ax.set_yticklabels(ACTION_LABELS)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  已保存: {output_path}")
    return heatmap


def generate_load_pattern_heatmaps(agent, num_servers=3, joint_state_dim=20,
                                   max_queue=50, output_dir="."):
    """在不同负载模式下生成热力图对比

    三种模式:
    1. 正常模式: 邻居队列适中 (30%)
    2. 高负载模式: 邻居队列高 (70%), 持续过载计数器>0
    3. 突发模式: 本服务器突然高负载, 邻居空闲
    """
    patterns = [
        {
            "name": "正常负载",
            "suffix": "normal",
            "peer_queue_ratio": 0.3,
            "peer_overload": 0.0,
            "self_overload_offset": 0.0,
            "local_ratio": 0.9,
        },
        {
            "name": "系统高负载",
            "suffix": "high_load",
            "peer_queue_ratio": 0.7,
            "peer_overload": 3.0 / 10.0,
            "self_overload_offset": 2.0 / 10.0,
            "local_ratio": 0.7,
        },
        {
            "name": "突发负载",
            "suffix": "burst",
            "peer_queue_ratio": 0.1,
            "peer_overload": 0.0,
            "self_overload_offset": 4.0 / 10.0,
            "local_ratio": 0.5,
        },
    ]

    all_heatmaps = {}

    for pattern in patterns:
        queue_bins = np.arange(0, max_queue + 1, 2)
        hour_bins = np.arange(0, 24, 1)
        heatmap = np.zeros((len(hour_bins), len(queue_bins)), dtype=np.int32)

        for hi, hour in enumerate(hour_bins):
            for qi, queue_len in enumerate(queue_bins):
                state = np.zeros(joint_state_dim, dtype=np.float32)
                time_of_day = hour / 24.0

                for s in range(num_servers):
                    base = s * 6
                    if s == 0:  # 目标服务器
                        state[base + 0] = queue_len / max_queue
                        state[base + 1] = queue_len / max_queue * 0.8
                        state[base + 2] = time_of_day
                        state[base + 3] = 0.5
                        overload_val = pattern["self_overload_offset"]
                        if queue_len / max_queue > 0.6:
                            overload_val += (queue_len / max_queue - 0.6) * 5 / 10.0
                        state[base + 4] = min(1.0, overload_val)
                        state[base + 5] = pattern["local_ratio"]
                    else:
                        state[base + 0] = pattern["peer_queue_ratio"]
                        state[base + 1] = pattern["peer_queue_ratio"] * 0.8
                        state[base + 2] = time_of_day
                        state[base + 3] = 0.5
                        state[base + 4] = pattern["peer_overload"]
                        state[base + 5] = 0.9

                global_base = num_servers * 6
                total_q = queue_len / max_queue + pattern["peer_queue_ratio"] * (num_servers - 1)
                state[global_base] = total_q / num_servers
                state[global_base + 1] = 1.0

                actions = agent.select_actions(state, training=False)
                heatmap[hi, qi] = actions[0]

        all_heatmaps[pattern["name"]] = heatmap

    # 绘制三合一对比图
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    queue_bins = np.arange(0, max_queue + 1, 2)
    hour_bins = np.arange(0, 24, 1)

    for idx, pattern in enumerate(patterns):
        ax = axes[idx]
        hm = all_heatmaps[pattern["name"]]
        im = ax.imshow(hm, aspect='auto', origin='lower',
                       cmap=plt.cm.get_cmap('tab10', 6),
                       vmin=-0.5, vmax=5.5)
        ax.set_xticks(np.arange(0, len(queue_bins), 5))
        ax.set_xticklabels(queue_bins[::5])
        ax.set_yticks(np.arange(0, len(hour_bins), 4))
        ax.set_yticklabels(hour_bins[::4])
        ax.set_xlabel('队列长度')
        ax.set_ylabel('时间 (小时)')
        ax.set_title(f'策略热力图: {pattern["name"]}')

    cbar = plt.colorbar(im, ax=axes.tolist(), ticks=list(range(6)),
                        fraction=0.02, pad=0.04)
    cbar.ax.set_yticklabels(ACTION_LABELS)

    plt.suptitle('不同负载模式下的迁移决策策略对比', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/strategy_heatmap_patterns.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  已保存: {output_dir}/strategy_heatmap_patterns.png")

    return all_heatmaps
