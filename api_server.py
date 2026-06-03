"""云控制器模拟插件 - REST API

通过REST API接收负载数据, 模拟Prometheus采集,
实时推送动作决策, 并记录决策日志。

启动: python run_api.py
访问: http://localhost:8000/
"""

import asyncio
import json
import time
from datetime import datetime
from collections import deque
from typing import List, Optional
from contextlib import asynccontextmanager
import threading

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from cluster_env import ClusterEnv
from multi_agent_dqn import CentralizedMADQN
from workload_trace import AlibabaTraceGenerator, Double11TraceGenerator
from meta_learner import ReptileMetaLearner, TaskDistribution


# ========================= Pydantic Models =========================

class LoadData(BaseModel):
    server_loads: List[float]
    timestamp: Optional[str] = None
    request_counts: Optional[List[int]] = None
    sequence_id: Optional[int] = None
    expected_queues: Optional[List[int]] = None

    @field_validator('server_loads')
    @classmethod
    def validate_loads(cls, v):
        if not v:
            raise ValueError("server_loads cannot be empty")
        for load in v:
            if not (0.0 <= load <= 1.0):
                raise ValueError(f"server_load must be in [0, 1], got {load}")
        return v

    @field_validator('request_counts')
    @classmethod
    def validate_request_counts(cls, v):
        if v is not None:
            for count in v:
                if count < 0:
                    raise ValueError(f"request_count must be >= 0, got {count}")
        return v

class ActionResponse(BaseModel):
    actions: List[int]
    action_names: List[str]
    confidence: List[float]

class MetricsResponse(BaseModel):
    timestamp: str
    powers: List[float]
    total_power: float
    queues: List[int]
    freqs: List[int]
    avg_response_time: float
    migrations_this_step: int
    energy_saving_pct: float

class DecisionLogEntry(BaseModel):
    timestamp: str
    step: int
    state_summary: dict
    actions: List[int]
    action_names: List[str]
    reward: float
    power: float

class SimulationConfig(BaseModel):
    speed: float = 1.0
    pattern: str = "normal"
    duration_hours: int = 24


# ========================= Action Names =========================

ACTION_NAMES = ["降频", "维持", "升频", "迁移-低", "迁移-中", "迁移-高"]


# ========================= Controller State =========================

class ControllerState:
    """控制器内存状态"""

    def __init__(self):
        self.env: Optional[ClusterEnv] = None
        self.agent: Optional[CentralizedMADQN] = None
        self.meta_learner: Optional[ReptileMetaLearner] = None

        self.decision_log: deque = deque(maxlen=1000)
        self.metrics_history: deque = deque(maxlen=360)
        self.websocket_clients: set = set()
        self._ws_lock = asyncio.Lock()

        self.is_running: bool = False
        self.simulation_task: Optional[asyncio.Task] = None
        self.current_step: int = 0
        self.total_energy_wh: float = 0.0
        self.baseline_energy_wh: float = 0.0

        self.current_trace: Optional[np.ndarray] = None
        self.current_obs: Optional[dict] = None

        # 状态同步
        self._sequence_counter: int = 0
        self._state_lock = threading.Lock()

        # 自动适应记录
        self.auto_adapt_log: deque = deque(maxlen=100)

    def initialize(self, model_path: Optional[str] = None):
        """初始化环境和agent"""
        trace_gen = AlibabaTraceGenerator(duration_hours=24, seed=42)
        self.current_trace = trace_gen.generate_trace()

        self.env = ClusterEnv(trace=self.current_trace, energy_weight=0.5, seed=42)
        self.current_obs = self.env.reset()

        self.agent = CentralizedMADQN(
            joint_state_dim=self.env.joint_state_dim,
            num_servers=self.env.num_servers,
            action_per_server=self.env.action_dim,
        )

        if model_path:
            try:
                self.agent.load(model_path)
                print(f"[Controller] 已加载模型: {model_path}")
            except Exception as e:
                print(f"[Controller] 模型加载失败 ({e}), 使用随机初始化")
                self._quick_train()
        else:
            self._quick_train()

        self.meta_learner = ReptileMetaLearner(
            self.agent, auto_detect=True,
            detect_window=24, detect_threshold=2.5, detect_cooldown=12,
        )
        self.current_step = 0
        self.total_energy_wh = 0.0
        self.baseline_energy_wh = 0.0
        self._sequence_counter = 0

    def _quick_train(self):
        """快速训练几个episode以获取初始策略"""
        print("[Controller] 快速训练初始策略 (200 episodes)...")
        env = ClusterEnv(trace=self.current_trace, energy_weight=0.5, seed=42)

        for ep in range(200):
            obs = env.reset()
            state = obs["joint_state"]
            done = False
            while not done:
                actions = self.agent.select_actions(state, training=True)
                obs, rewards, done, info = env.step(actions)
                self.agent.store_transition(
                    state, actions, rewards["total"], obs["joint_state"], done
                )
                self.agent.update()
                self.agent.steps_done += 1
                state = obs["joint_state"]

        print("[Controller] 初始训练完成")

    def validate_state_sync(self, data: 'LoadData') -> Optional[str]:
        """验证客户端提交的状态与服务端是否一致

        Returns:
            None if valid, error message string if inconsistent
        """
        if data.sequence_id is not None:
            expected_seq = self._sequence_counter
            if data.sequence_id != expected_seq:
                return (
                    f"Sequence mismatch: client={data.sequence_id}, "
                    f"server={expected_seq}. Client may have missed steps."
                )

        if data.expected_queues is not None and self.env is not None:
            actual_queues = [s.queue_length for s in self.env.servers]
            for i, (expected, actual) in enumerate(
                zip(data.expected_queues, actual_queues)
            ):
                if abs(expected - actual) > 5:
                    return (
                        f"Queue state mismatch on server {i}: "
                        f"client expects {expected}, actual is {actual}. "
                        f"State divergence detected."
                    )

        if len(data.server_loads) != self.env.num_servers:
            return (
                f"Server count mismatch: got {len(data.server_loads)} loads, "
                f"expected {self.env.num_servers}"
            )

        return None

    def step_environment(self, arrivals: Optional[int] = None):
        """推进环境一步, 返回指标"""
        if self.current_obs is None:
            return None

        with self._state_lock:
            joint_state = self.current_obs["joint_state"]
            actions = self.agent.select_actions(joint_state, training=False)

            self.current_obs, rewards, done, info = self.env.step(actions)

            step_energy = info["total_power"] * (5.0 / 60.0)
            self.total_energy_wh += step_energy

            baseline_power = sum(
                cfg["p_max"] * 0.7 for cfg in self.env.server_configs
            )
            self.baseline_energy_wh += baseline_power * (5.0 / 60.0)

            energy_saving_pct = 0.0
            if self.baseline_energy_wh > 0:
                energy_saving_pct = (
                    (self.baseline_energy_wh - self.total_energy_wh)
                    / self.baseline_energy_wh * 100.0
                )

            # 自动模式检测: 输入总到达负载
            total_load = sum(info["queues"])
            if self.meta_learner:
                adapted, new_agent = self.meta_learner.observe_and_adapt(
                    total_load, self.agent
                )
                if adapted and new_agent is not None:
                    self.agent = new_agent
                    self.auto_adapt_log.append({
                        "step": self.current_step,
                        "timestamp": datetime.now().isoformat(timespec='seconds'),
                        "drift_score": self.meta_learner.detector.drift_score,
                        "total_adapts": self.meta_learner.auto_adapt_count,
                    })

            now = datetime.now().isoformat(timespec='seconds')
            self._sequence_counter += 1

            metrics = {
                "timestamp": now,
                "step": self.current_step,
                "sequence_id": self._sequence_counter,
                "powers": [round(p, 2) for p in info["powers"]],
                "total_power": round(info["total_power"], 2),
                "queues": info["queues"],
                "freqs": info["freqs"],
                "avg_response_time": round(info["avg_response_time"], 4),
                "migrations_this_step": info["migrations_this_step"],
                "energy_saving_pct": round(energy_saving_pct, 2),
                "actions": actions,
                "action_names": [ACTION_NAMES[a] for a in actions],
                "reward": round(rewards["total"], 3),
            }

            self.metrics_history.append(metrics)
            self.decision_log.append(DecisionLogEntry(
                timestamp=now,
                step=self.current_step,
                state_summary={
                    "queues": info["queues"],
                    "freqs": info["freqs"],
                    "overload_counters": info["overload_counters"],
                },
                actions=actions,
                action_names=[ACTION_NAMES[a] for a in actions],
                reward=rewards["total"],
                power=info["total_power"],
            ))

            self.current_step += 1

            if done:
                self.current_obs = self.env.reset()
                self.current_step = 0
                self.total_energy_wh = 0.0
                self.baseline_energy_wh = 0.0

        return metrics

    async def add_websocket(self, websocket: WebSocket):
        """注册WebSocket客户端"""
        async with self._ws_lock:
            self.websocket_clients.add(websocket)

    async def remove_websocket(self, websocket: WebSocket):
        """移除WebSocket客户端并清理资源"""
        async with self._ws_lock:
            self.websocket_clients.discard(websocket)

    async def broadcast_metrics(self, metrics: dict):
        """WebSocket广播指标到所有连接的客户端"""
        async with self._ws_lock:
            if not self.websocket_clients:
                return
            message = json.dumps(metrics, ensure_ascii=False)
            stale = set()
            for ws in self.websocket_clients:
                try:
                    await ws.send_text(message)
                except Exception:
                    stale.add(ws)
            self.websocket_clients -= stale


# ========================= Application =========================

state = ControllerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.initialize()
    yield
    state.is_running = False
    if state.simulation_task:
        state.simulation_task.cancel()


app = FastAPI(
    title="Cloud DQN Controller",
    description="DQN云控制器模拟插件 - 实时负载调度与节能优化",
    version="1.0.0",
    lifespan=lifespan,
)


# ========================= REST Endpoints =========================

@app.post("/api/load", response_model=MetricsResponse)
async def receive_load(data: LoadData):
    """接收实时负载数据 (模拟Prometheus采集), 推进环境并返回指标

    支持状态同步校验:
    - sequence_id: 客户端期望的步序号, 不一致时返回400
    - expected_queues: 客户端预期的队列状态, 偏差过大时返回409
    """
    sync_error = state.validate_state_sync(data)
    if sync_error:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "state_sync_failed",
                "message": sync_error,
                "server_step": state.current_step,
                "server_sequence": state._sequence_counter,
                "server_queues": [s.queue_length for s in state.env.servers],
            },
        )

    if data.request_counts:
        for i, count in enumerate(data.request_counts[:len(state.env.servers)]):
            state.env.servers[i].queue_length = min(
                state.env.servers[i].queue_length + count,
                state.env.max_queue,
            )

    metrics = state.step_environment()
    if metrics is None:
        return MetricsResponse(
            timestamp=datetime.now().isoformat(),
            powers=[0, 0, 0], total_power=0, queues=[0, 0, 0],
            freqs=[1, 1, 1], avg_response_time=0,
            migrations_this_step=0, energy_saving_pct=0,
        )

    await state.broadcast_metrics(metrics)

    return MetricsResponse(
        timestamp=metrics["timestamp"],
        powers=metrics["powers"],
        total_power=metrics["total_power"],
        queues=metrics["queues"],
        freqs=metrics["freqs"],
        avg_response_time=metrics["avg_response_time"],
        migrations_this_step=metrics["migrations_this_step"],
        energy_saving_pct=metrics["energy_saving_pct"],
    )


@app.get("/api/action", response_model=ActionResponse)
async def get_action():
    """获取当前agent推荐动作"""
    if state.current_obs is None:
        return ActionResponse(actions=[1, 1, 1], action_names=["维持"]*3, confidence=[0]*3)

    import torch
    joint_state = state.current_obs["joint_state"]
    with torch.no_grad():
        state_t = torch.FloatTensor(joint_state).unsqueeze(0).to(state.agent.device)
        q_heads = state.agent.policy_net(state_t)
        actions = [q.argmax(dim=1).item() for q in q_heads]
        confidences = [q.max(dim=1).values.item() for q in q_heads]

    return ActionResponse(
        actions=actions,
        action_names=[ACTION_NAMES[a] for a in actions],
        confidence=[round(c, 3) for c in confidences],
    )


@app.get("/api/metrics")
async def get_metrics():
    """获取最近指标历史"""
    return list(state.metrics_history)


@app.get("/api/decision-log", response_model=List[DecisionLogEntry])
async def get_decision_log(limit: int = 100):
    """获取决策日志"""
    entries = list(state.decision_log)
    return entries[-limit:]


# ========================= Simulation Control =========================

@app.post("/api/simulation/start")
async def start_simulation(config: SimulationConfig = SimulationConfig()):
    """启动自动仿真"""
    if state.is_running:
        return {"status": "already_running"}

    if config.pattern == "double11":
        gen = Double11TraceGenerator(duration_hours=config.duration_hours, seed=77)
    else:
        gen = AlibabaTraceGenerator(duration_hours=config.duration_hours, seed=77)

    state.current_trace = gen.generate_trace()
    state.env = ClusterEnv(trace=state.current_trace, energy_weight=0.5, seed=None)
    state.current_obs = state.env.reset()
    state.current_step = 0
    state.total_energy_wh = 0.0
    state.baseline_energy_wh = 0.0
    state.is_running = True

    async def simulation_loop():
        interval = 0.5 / max(config.speed, 0.1)
        while state.is_running:
            metrics = state.step_environment()
            if metrics:
                await state.broadcast_metrics(metrics)
            await asyncio.sleep(interval)

    state.simulation_task = asyncio.create_task(simulation_loop())
    return {"status": "started", "pattern": config.pattern, "speed": config.speed}


@app.post("/api/simulation/stop")
async def stop_simulation():
    """停止自动仿真"""
    state.is_running = False
    if state.simulation_task:
        state.simulation_task.cancel()
        state.simulation_task = None
    return {"status": "stopped"}


# ========================= Meta-Learning =========================

@app.post("/api/meta/adapt")
async def trigger_adaptation(pattern: str = "double11", episodes: int = 5,
                             duration_hours: int = 1):
    """触发元学习快速适应

    Args:
        pattern: 负载模式 ("double11" 或 "normal")
        episodes: 微调episode数
        duration_hours: 适应数据时长(小时), 支持1-72小时
    """
    if state.meta_learner is None:
        return {"status": "error", "message": "meta_learner not initialized"}

    duration_hours = max(1, min(duration_hours, 72))

    if pattern == "double11":
        gen = Double11TraceGenerator(duration_hours=duration_hours, seed=None)
    else:
        gen = AlibabaTraceGenerator(duration_hours=duration_hours, seed=None)

    new_trace = gen.generate_trace()
    adapted_agent = state.meta_learner.fast_adapt(new_trace, adaptation_episodes=episodes)

    state.agent = adapted_agent
    return {
        "status": "adapted",
        "pattern": pattern,
        "episodes": episodes,
        "duration_hours": duration_hours,
        "trace_steps": len(new_trace),
    }


@app.get("/api/meta/status")
async def meta_status():
    """获取元学习器状态 (自动适应记录)"""
    if state.meta_learner is None:
        return {"status": "not_initialized"}

    return {
        "auto_detect_enabled": state.meta_learner.auto_detect,
        "auto_adapt_count": state.meta_learner.auto_adapt_count,
        "total_adaptations": len(state.meta_learner.adaptation_history),
        "recent_auto_adapts": list(state.auto_adapt_log)[-10:],
        "detector_drift_score": (
            state.meta_learner.detector.drift_score
            if state.meta_learner.detector else 0
        ),
    }


# ========================= WebSocket =========================

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """实时指标推送WebSocket (含连接清理)"""
    await websocket.accept()
    await state.add_websocket(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await state.remove_websocket(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


# ========================= Dashboard =========================

@app.get("/")
async def dashboard():
    """返回前端仪表板"""
    return FileResponse("static/dashboard.html")


import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
