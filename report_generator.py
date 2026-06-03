"""最终实验报告生成器

包含:
1. 训练收敛速度分析
2. 实际节能百分比
3. 鲁棒性测试 (故障注入、突发负载、模式切换)
4. 元学习适应速度基准
"""

import numpy as np
import time
import base64
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cluster_env import ClusterEnv
from multi_agent_dqn import CentralizedMADQN, IndependentDQN
from baselines import RoundRobinScheduler
from workload_trace import (
    AlibabaTraceGenerator,
    Double11TraceGenerator,
    BurstPatternGenerator,
)
from meta_learner import ReptileMetaLearner, TaskDistribution
from run_comparison import train_centralized_madqn, train_independent_dqn, evaluate_method

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ReportGenerator:
    """综合实验报告生成器

    支持两种模式:
    1. 传入已训练的agent和rewards (在训练脚本中调用)
    2. 从.pth文件加载预训练模型 (避免重复训练)
    """

    def __init__(self, train_trace=None, test_trace=None,
                 madqn_agent=None, idqn_agent=None,
                 madqn_rewards=None, idqn_rewards=None,
                 madqn_model_path=None, idqn_model_path=None):
        self.train_trace = train_trace
        self.test_trace = test_trace
        self.madqn_agent = madqn_agent
        self.idqn_agent = idqn_agent
        self.madqn_rewards = madqn_rewards or []
        self.idqn_rewards = idqn_rewards or []
        self.results = {}

        if self.train_trace is None:
            trace_gen = AlibabaTraceGenerator(duration_hours=24, seed=42)
            self.train_trace = trace_gen.generate_trace()
        if self.test_trace is None:
            test_gen = AlibabaTraceGenerator(duration_hours=24, seed=123)
            self.test_trace = test_gen.generate_trace()

        if madqn_model_path and self.madqn_agent is None:
            self.madqn_agent = self._load_madqn(madqn_model_path)
        if idqn_model_path and self.idqn_agent is None:
            self.idqn_agent = self._load_idqn(idqn_model_path)

    def _load_madqn(self, path):
        """从文件加载预训练CentralizedMADQN"""
        from multi_agent_dqn import CentralizedMADQN
        env = ClusterEnv(trace=self.train_trace, seed=42)
        agent = CentralizedMADQN(
            joint_state_dim=env.joint_state_dim,
            num_servers=env.num_servers,
            action_per_server=env.action_dim,
        )
        agent.load(path)
        print(f"[Report] 已加载MADDQN模型: {path}")
        return agent

    def _load_idqn(self, path):
        """从文件加载预训练IndependentDQN"""
        import torch
        from multi_agent_dqn import IndependentDQN
        env = ClusterEnv(trace=self.train_trace, seed=42)
        agent = IndependentDQN(
            num_servers=env.num_servers,
            local_state_dim=env.local_state_dim,
            action_dim=env.action_dim,
        )
        data = torch.load(path, weights_only=True)
        for i, a in enumerate(agent.agents):
            a.policy_net.load_state_dict(data[f"agent_{i}_policy"])
            a.target_net.load_state_dict(data[f"agent_{i}_target"])
        print(f"[Report] 已加载IDDQN模型: {path}")
        return agent

    def analyze_convergence(self):
        """训练收敛速度分析"""
        def _convergence_info(rewards, name):
            if not rewards:
                return {"name": name, "converged": False}

            final_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            threshold_90 = final_reward * 0.9

            episode_90 = len(rewards)
            window = 50
            for i in range(window, len(rewards)):
                avg = np.mean(rewards[i-window:i])
                if avg >= threshold_90:
                    episode_90 = i
                    break

            reward_std = np.std(rewards[-100:]) if len(rewards) >= 100 else np.std(rewards)

            return {
                "name": name,
                "converged": True,
                "total_episodes": len(rewards),
                "episode_to_90pct": episode_90,
                "final_reward_mean": round(float(final_reward), 2),
                "final_reward_std": round(float(reward_std), 2),
                "convergence_rate": round(episode_90 / max(len(rewards), 1) * 100, 1),
            }

        madqn_info = _convergence_info(self.madqn_rewards, "Centralized MADDQN")
        idqn_info = _convergence_info(self.idqn_rewards, "Independent DDQN")

        self.results["convergence"] = {
            "madqn": madqn_info,
            "idqn": idqn_info,
        }
        return self.results["convergence"]

    def compute_energy_savings(self, num_episodes=20):
        """计算实际节能百分比"""
        if self.madqn_agent is None or self.test_trace is None:
            return {"error": "agents or test_trace not provided"}

        rr_agent = RoundRobinScheduler(num_servers=3)

        madqn_e, madqn_r, madqn_m = evaluate_method(
            self.madqn_agent, self.test_trace, num_episodes, "madqn"
        )

        rr_e, rr_r, rr_m = evaluate_method(
            rr_agent, self.test_trace, num_episodes, "rr"
        )

        # 固定频率基线 (全高频运行)
        fixed_energies = []
        for ep in range(num_episodes):
            env = ClusterEnv(trace=self.test_trace, seed=ep + 200)
            env.reset()
            total_e = 0.0
            done = False
            while not done:
                _, _, done, info = env.step([2, 2, 2])
                total_e += info["total_power"] * (5.0 / 60.0)
            fixed_energies.append(total_e / 1000.0)

        madqn_avg = np.mean(madqn_e)
        rr_avg = np.mean(rr_e)
        fixed_avg = np.mean(fixed_energies)

        self.results["energy_savings"] = {
            "madqn_kwh": round(madqn_avg, 4),
            "rr_kwh": round(rr_avg, 4),
            "fixed_high_freq_kwh": round(fixed_avg, 4),
            "saving_vs_rr_pct": round((rr_avg - madqn_avg) / rr_avg * 100, 2),
            "saving_vs_fixed_pct": round((fixed_avg - madqn_avg) / fixed_avg * 100, 2),
            "madqn_response_time": round(float(np.mean(madqn_r)), 4),
            "rr_response_time": round(float(np.mean(rr_r)), 4),
        }
        return self.results["energy_savings"]

    def robustness_test_failure_injection(self, num_trials=10):
        """鲁棒性测试: 随机故障注入"""
        if self.madqn_agent is None:
            return {}

        results = {"sla_violations": [], "energy_overhead": [], "recovery_steps": []}

        for trial in range(num_trials):
            env = ClusterEnv(trace=self.test_trace, seed=trial + 300)
            env.reset()

            for cfg in env.server_configs:
                cfg["failure_prob"] = 0.02

            obs = env.reset()
            total_energy = 0.0
            sla_violations = 0
            done = False

            while not done:
                actions = self.madqn_agent.select_actions(obs["joint_state"], training=False)
                obs, _, done, info = env.step(actions)
                total_energy += info["total_power"] * (5.0 / 60.0)
                sla_violations += sum(1 for q in info["queues"] if q > 20)

            results["sla_violations"].append(sla_violations)
            results["energy_overhead"].append(total_energy / 1000.0)

        self.results["robustness_failure"] = {
            "avg_sla_violations": round(float(np.mean(results["sla_violations"])), 1),
            "std_sla_violations": round(float(np.std(results["sla_violations"])), 1),
            "avg_energy_kwh": round(float(np.mean(results["energy_overhead"])), 4),
            "num_trials": num_trials,
        }
        return self.results["robustness_failure"]

    def robustness_test_burst_load(self, num_trials=10):
        """鲁棒性测试: 突发负载"""
        if self.madqn_agent is None:
            return {}

        results = {"sla_violations": [], "max_queue": [], "energy": []}

        for trial in range(num_trials):
            gen = BurstPatternGenerator(
                duration_hours=24, seed=trial + 400,
                burst_magnitude=10.0, burst_interval_steps=20,
            )
            burst_trace = gen.generate_trace()

            env = ClusterEnv(trace=burst_trace, seed=trial + 400)
            obs = env.reset()
            sla_violations = 0
            max_q = 0
            total_energy = 0.0
            done = False

            while not done:
                actions = self.madqn_agent.select_actions(obs["joint_state"], training=False)
                obs, _, done, info = env.step(actions)
                total_energy += info["total_power"] * (5.0 / 60.0)
                sla_violations += sum(1 for q in info["queues"] if q > 20)
                max_q = max(max_q, max(info["queues"]))

            results["sla_violations"].append(sla_violations)
            results["max_queue"].append(max_q)
            results["energy"].append(total_energy / 1000.0)

        self.results["robustness_burst"] = {
            "avg_sla_violations": round(float(np.mean(results["sla_violations"])), 1),
            "avg_max_queue": round(float(np.mean(results["max_queue"])), 1),
            "avg_energy_kwh": round(float(np.mean(results["energy"])), 4),
            "num_trials": num_trials,
        }
        return self.results["robustness_burst"]

    def robustness_test_pattern_change(self, num_trials=5):
        """鲁棒性测试: 中途模式切换 (正常->双十一)"""
        if self.madqn_agent is None:
            return {}

        results = {"reward_drop": [], "recovery_steps": []}

        for trial in range(num_trials):
            normal_gen = AlibabaTraceGenerator(duration_hours=24, seed=trial + 500)
            d11_gen = Double11TraceGenerator(duration_hours=24, seed=trial + 500)

            normal_trace = normal_gen.generate_trace()
            d11_trace = d11_gen.generate_trace()

            switch_step = len(normal_trace) // 2
            combined_trace = np.concatenate([
                normal_trace[:switch_step], d11_trace[switch_step:]
            ])

            env = ClusterEnv(trace=combined_trace, seed=trial + 500)
            obs = env.reset()

            rewards_before = []
            rewards_after = []
            step = 0
            done = False

            while not done:
                actions = self.madqn_agent.select_actions(obs["joint_state"], training=False)
                obs, rewards, done, info = env.step(actions)
                if step < switch_step:
                    rewards_before.append(rewards["total"])
                else:
                    rewards_after.append(rewards["total"])
                step += 1

            if rewards_before and rewards_after:
                avg_before = np.mean(rewards_before[-30:])
                avg_after_initial = np.mean(rewards_after[:30]) if len(rewards_after) >= 30 else np.mean(rewards_after)
                drop = (avg_before - avg_after_initial) / abs(avg_before) * 100 if avg_before != 0 else 0
                results["reward_drop"].append(drop)

                recovery = len(rewards_after)
                for i in range(30, len(rewards_after)):
                    window_avg = np.mean(rewards_after[i-30:i])
                    if window_avg >= avg_before * 0.8:
                        recovery = i
                        break
                results["recovery_steps"].append(recovery)

        self.results["robustness_pattern_change"] = {
            "avg_reward_drop_pct": round(float(np.mean(results["reward_drop"])), 1),
            "avg_recovery_steps": round(float(np.mean(results["recovery_steps"])), 0),
            "num_trials": num_trials,
        }
        return self.results["robustness_pattern_change"]

    def meta_learning_benchmark(self, meta_learner=None, num_trials=5):
        """元学习适应速度基准测试"""
        if self.madqn_agent is None:
            return {}

        d11_gen = Double11TraceGenerator(duration_hours=1, seed=888)
        one_hour_trace = d11_gen.generate_trace()

        # 从零微调 (无元学习)
        scratch_rewards = []
        for trial in range(num_trials):
            agent_scratch = CentralizedMADQN(
                joint_state_dim=20, num_servers=3, action_per_server=6,
                epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=100,
            )
            env = ClusterEnv(trace=one_hour_trace, energy_weight=0.5, seed=trial)
            obs = env.reset()
            total_r = 0
            for _ in range(5):
                obs = env.reset()
                state = obs["joint_state"]
                ep_r = 0
                done = False
                while not done:
                    actions = agent_scratch.select_actions(state, training=True)
                    obs, rewards, done, info = env.step(actions)
                    agent_scratch.store_transition(state, actions, rewards["total"], obs["joint_state"], done)
                    agent_scratch.update()
                    agent_scratch.steps_done += 1
                    state = obs["joint_state"]
                    ep_r += rewards["total"]
                total_r += ep_r
            scratch_rewards.append(total_r / 5)

        # 从元模型微调
        if meta_learner is None:
            meta_learner = ReptileMetaLearner(self.madqn_agent)

        meta_rewards = []
        for trial in range(num_trials):
            adapted = meta_learner.fast_adapt(one_hour_trace, adaptation_episodes=5)
            env = ClusterEnv(trace=one_hour_trace, energy_weight=0.5, seed=trial + 100)
            obs = env.reset()
            state = obs["joint_state"]
            ep_r = 0
            done = False
            while not done:
                actions = adapted.select_actions(state, training=False)
                obs, rewards, done, info = env.step(actions)
                state = obs["joint_state"]
                ep_r += rewards["total"]
            meta_rewards.append(ep_r)

        self.results["meta_learning"] = {
            "scratch_avg_reward": round(float(np.mean(scratch_rewards)), 2),
            "scratch_std": round(float(np.std(scratch_rewards)), 2),
            "meta_avg_reward": round(float(np.mean(meta_rewards)), 2),
            "meta_std": round(float(np.std(meta_rewards)), 2),
            "improvement_pct": round(
                (np.mean(meta_rewards) - np.mean(scratch_rewards))
                / max(abs(np.mean(scratch_rewards)), 1e-6) * 100, 1
            ),
            "adaptation_data": "1 hour (12 steps @ 5min)",
            "adaptation_episodes": 5,
        }
        return self.results["meta_learning"]

    def generate_console_report(self):
        """生成控制台文本报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("          FINAL EXPERIMENT REPORT")
        lines.append("          Multi-Agent DQN Energy-Efficient Scheduling")
        lines.append("=" * 70)

        # Convergence
        if "convergence" in self.results:
            lines.append("\n[1] TRAINING CONVERGENCE ANALYSIS")
            lines.append("-" * 40)
            for key in ["madqn", "idqn"]:
                info = self.results["convergence"][key]
                lines.append(f"  {info['name']}:")
                if info.get("converged"):
                    lines.append(f"    Total Episodes: {info['total_episodes']}")
                    lines.append(f"    Episode to 90% reward: {info['episode_to_90pct']}")
                    lines.append(f"    Final Reward: {info['final_reward_mean']} +/- {info['final_reward_std']}")
                    lines.append(f"    Convergence Rate: {info['convergence_rate']}%")

        # Energy Savings
        if "energy_savings" in self.results:
            lines.append("\n[2] ENERGY SAVING PERCENTAGE")
            lines.append("-" * 40)
            es = self.results["energy_savings"]
            lines.append(f"  MADDQN Energy: {es['madqn_kwh']} kWh/day")
            lines.append(f"  Round-Robin Energy: {es['rr_kwh']} kWh/day")
            lines.append(f"  Fixed High-Freq Energy: {es['fixed_high_freq_kwh']} kWh/day")
            lines.append(f"  ---")
            lines.append(f"  Saving vs Round-Robin: {es['saving_vs_rr_pct']}%")
            lines.append(f"  Saving vs Fixed High-Freq: {es['saving_vs_fixed_pct']}%")
            lines.append(f"  MADDQN Avg Response Time: {es['madqn_response_time']} min")
            lines.append(f"  Round-Robin Avg Response Time: {es['rr_response_time']} min")

        # Robustness - Failure
        if "robustness_failure" in self.results:
            lines.append("\n[3] ROBUSTNESS: FAILURE INJECTION")
            lines.append("-" * 40)
            rf = self.results["robustness_failure"]
            lines.append(f"  Failure Prob: 2% per step (10x normal)")
            lines.append(f"  Avg SLA Violations: {rf['avg_sla_violations']} +/- {rf['std_sla_violations']}")
            lines.append(f"  Avg Energy: {rf['avg_energy_kwh']} kWh")
            lines.append(f"  Trials: {rf['num_trials']}")

        # Robustness - Burst
        if "robustness_burst" in self.results:
            lines.append("\n[4] ROBUSTNESS: BURST LOAD (10x)")
            lines.append("-" * 40)
            rb = self.results["robustness_burst"]
            lines.append(f"  Burst Magnitude: 10x, Interval: 20 steps")
            lines.append(f"  Avg SLA Violations: {rb['avg_sla_violations']}")
            lines.append(f"  Avg Max Queue: {rb['avg_max_queue']}")
            lines.append(f"  Avg Energy: {rb['avg_energy_kwh']} kWh")

        # Robustness - Pattern Change
        if "robustness_pattern_change" in self.results:
            lines.append("\n[5] ROBUSTNESS: MID-EPISODE PATTERN CHANGE")
            lines.append("-" * 40)
            rp = self.results["robustness_pattern_change"]
            lines.append(f"  Pattern Switch: Normal -> Double-11 at midpoint")
            lines.append(f"  Avg Reward Drop: {rp['avg_reward_drop_pct']}%")
            lines.append(f"  Avg Recovery Steps: {rp['avg_recovery_steps']}")

        # Meta-Learning
        if "meta_learning" in self.results:
            lines.append("\n[6] META-LEARNING FAST ADAPTATION")
            lines.append("-" * 40)
            ml = self.results["meta_learning"]
            lines.append(f"  Adaptation Data: {ml['adaptation_data']}")
            lines.append(f"  Adaptation Episodes: {ml['adaptation_episodes']}")
            lines.append(f"  From Scratch Reward: {ml['scratch_avg_reward']} +/- {ml['scratch_std']}")
            lines.append(f"  Meta-Adapted Reward: {ml['meta_avg_reward']} +/- {ml['meta_std']}")
            lines.append(f"  Improvement: {ml['improvement_pct']}%")

        lines.append("\n" + "=" * 70)
        lines.append("Report generated at: " + time.strftime("%Y-%m-%d %H:%M:%S"))
        lines.append("=" * 70)

        report = "\n".join(lines)
        return report

    def generate_html_report(self, output_path="report.html"):
        """生成HTML格式报告"""
        console_report = self.generate_console_report()

        # 生成收敛曲线图
        fig, ax = plt.subplots(figsize=(10, 4))
        if self.madqn_rewards:
            window = 50
            smoothed = [np.mean(self.madqn_rewards[max(0,i-window):i+1])
                        for i in range(len(self.madqn_rewards))]
            ax.plot(smoothed, label='MADDQN', color='#2196F3', linewidth=1.5)
        if self.idqn_rewards:
            smoothed = [np.mean(self.idqn_rewards[max(0,i-window):i+1])
                        for i in range(len(self.idqn_rewards))]
            ax.plot(smoothed, label='IDDQN', color='#FF9800', linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        convergence_img = base64.b64encode(buf.getvalue()).decode()

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>DQN Energy Scheduling - Experiment Report</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #fafafa; }}
        h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 10px; }}
        h2 {{ color: #283593; margin-top: 30px; }}
        .metric {{ background: white; padding: 16px; border-radius: 8px; margin: 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metric .label {{ font-size: 13px; color: #666; }}
        .metric .value {{ font-size: 24px; font-weight: 600; color: #1565c0; }}
        .metric .value.green {{ color: #2e7d32; }}
        .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
        pre {{ background: #263238; color: #eeffff; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 12px; }}
        img {{ max-width: 100%; border-radius: 8px; margin: 16px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
        th, td {{ padding: 8px 12px; border: 1px solid #e0e0e0; text-align: left; }}
        th {{ background: #e8eaf6; }}
    </style>
</head>
<body>
    <h1>Multi-Agent DQN Energy-Efficient Scheduling - Experiment Report</h1>
    <p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>1. Training Convergence</h2>
    <img src="data:image/png;base64,{convergence_img}" alt="Convergence Curve">
"""
        if "convergence" in self.results:
            html += '<div class="grid">'
            for key in ["madqn", "idqn"]:
                info = self.results["convergence"][key]
                if info.get("converged"):
                    html += f"""
    <div class="metric">
        <div class="label">{info['name']}</div>
        <div class="value">{info['episode_to_90pct']}</div>
        <div class="label">episodes to 90% convergence</div>
    </div>"""
            html += '</div>'

        if "energy_savings" in self.results:
            es = self.results["energy_savings"]
            html += f"""
    <h2>2. Energy Savings</h2>
    <div class="grid">
        <div class="metric">
            <div class="label">vs Round-Robin</div>
            <div class="value green">{es['saving_vs_rr_pct']}%</div>
        </div>
        <div class="metric">
            <div class="label">vs Fixed High-Freq</div>
            <div class="value green">{es['saving_vs_fixed_pct']}%</div>
        </div>
        <div class="metric">
            <div class="label">MADDQN Daily Energy</div>
            <div class="value">{es['madqn_kwh']} kWh</div>
        </div>
    </div>"""

        if "meta_learning" in self.results:
            ml = self.results["meta_learning"]
            html += f"""
    <h2>3. Meta-Learning Fast Adaptation</h2>
    <table>
        <tr><th>Metric</th><th>From Scratch</th><th>Meta-Adapted</th></tr>
        <tr><td>Reward (5 episodes)</td><td>{ml['scratch_avg_reward']}</td><td>{ml['meta_avg_reward']}</td></tr>
        <tr><td>Std Dev</td><td>{ml['scratch_std']}</td><td>{ml['meta_std']}</td></tr>
        <tr><td>Improvement</td><td colspan="2" style="color:green; font-weight:bold;">{ml['improvement_pct']}%</td></tr>
    </table>
    <p>Adaptation data: {ml['adaptation_data']} | Episodes: {ml['adaptation_episodes']}</p>"""

        html += f"""
    <h2>4. Robustness Tests</h2>
    <table>
        <tr><th>Test</th><th>Key Metric</th><th>Result</th></tr>"""

        if "robustness_failure" in self.results:
            rf = self.results["robustness_failure"]
            html += f'<tr><td>Failure Injection (2%)</td><td>SLA Violations</td><td>{rf["avg_sla_violations"]}</td></tr>'

        if "robustness_burst" in self.results:
            rb = self.results["robustness_burst"]
            html += f'<tr><td>Burst Load (10x)</td><td>Max Queue</td><td>{rb["avg_max_queue"]}</td></tr>'

        if "robustness_pattern_change" in self.results:
            rp = self.results["robustness_pattern_change"]
            html += f'<tr><td>Pattern Change</td><td>Reward Drop</td><td>{rp["avg_reward_drop_pct"]}%</td></tr>'
            html += f'<tr><td></td><td>Recovery Steps</td><td>{rp["avg_recovery_steps"]}</td></tr>'

        html += f"""
    </table>

    <h2>5. Full Console Report</h2>
    <pre>{console_report}</pre>
</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return output_path

    def run_all(self, output_dir=".", verbose=True):
        """运行所有分析并生成报告"""
        if verbose:
            print("[Report] Starting comprehensive analysis...")

        if verbose:
            print("  [1/6] Convergence analysis...")
        self.analyze_convergence()

        if verbose:
            print("  [2/6] Energy savings computation...")
        self.compute_energy_savings(num_episodes=10)

        if verbose:
            print("  [3/6] Robustness: failure injection...")
        self.robustness_test_failure_injection(num_trials=5)

        if verbose:
            print("  [4/6] Robustness: burst load...")
        self.robustness_test_burst_load(num_trials=5)

        if verbose:
            print("  [5/6] Robustness: pattern change...")
        self.robustness_test_pattern_change(num_trials=3)

        if verbose:
            print("  [6/6] Meta-learning benchmark...")
        self.meta_learning_benchmark()

        console_report = self.generate_console_report()
        if verbose:
            print(console_report)

        html_path = f"{output_dir}/report.html"
        self.generate_html_report(html_path)
        if verbose:
            print(f"\n  HTML report saved to: {html_path}")

        return self.results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("--madqn-model", default=None,
                        help="Path to pre-trained MADDQN .pth file (skip training)")
    parser.add_argument("--idqn-model", default=None,
                        help="Path to pre-trained IDDQN .pth file (skip training)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Training episodes (if not loading models)")
    parser.add_argument("--output", default=".", help="Output directory")
    args = parser.parse_args()

    trace_gen = AlibabaTraceGenerator(duration_hours=24, seed=42)
    train_trace = trace_gen.generate_trace()
    test_gen = AlibabaTraceGenerator(duration_hours=24, seed=123)
    test_trace = test_gen.generate_trace()

    madqn_agent = None
    idqn_agent = None
    madqn_rewards = []
    idqn_rewards = []

    if args.madqn_model:
        print(f"Loading pre-trained MADDQN from: {args.madqn_model}")
    else:
        print(f"\nTraining MADDQN ({args.episodes} episodes)...")
        madqn_agent, madqn_rewards, _ = train_centralized_madqn(
            train_trace, num_episodes=args.episodes, use_tensorboard=False
        )

    if args.idqn_model:
        print(f"Loading pre-trained IDDQN from: {args.idqn_model}")
    else:
        print(f"\nTraining IDDQN ({args.episodes} episodes)...")
        idqn_agent, idqn_rewards = train_independent_dqn(
            train_trace, num_episodes=args.episodes, use_tensorboard=False
        )

    report = ReportGenerator(
        train_trace=train_trace,
        test_trace=test_trace,
        madqn_agent=madqn_agent,
        idqn_agent=idqn_agent,
        madqn_rewards=madqn_rewards,
        idqn_rewards=idqn_rewards,
        madqn_model_path=args.madqn_model,
        idqn_model_path=args.idqn_model,
    )
    report.run_all(output_dir=args.output)
