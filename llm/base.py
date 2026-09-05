"""LLM 辅助 MARL 训练的接口定义。

这里只定义「接口」（抽象基类 + 数据上下文），不包含任何具体实现。
未来接入 LLM 时，实现这些接口并注入到 LLMAssistManager 即可，无需改动 runner / learner。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


class MetricsHistory:
    """训练指标历史收集器：{指标名: [历史值列表]}。

    learner 训练时写入，LLM 介入决策时读取（可拿最新值，也可拿完整历史）。
    """

    def __init__(self):
        self.history = defaultdict(list)

    def update(self, stats: dict):
        for k, v in stats.items():
            self.history[k].append(float(v))

    def latest(self) -> dict:
        """返回每个指标的最新值。"""
        return {k: v[-1] for k, v in self.history.items() if v}

    def snapshot(self) -> dict:
        """返回每个指标的完整历史（浅拷贝）。"""
        return {k: list(v) for k, v in self.history.items()}


@dataclass
class InterventionContext:
    """LLM 介入决策时可用的训练数据上下文。

    门控模块（InterventionGate）根据这些数据判断「是否让 LLM 介入」；
    字段按需扩展（例如加入 loss、熵、回报滑动平均等训练侧指标）。
    """
    t_env: int = 0                     # 当前环境步数
    episode: int = 0                   # 当前 episode 编号
    test_mode: bool = False            # 是否评估模式
    obs: Any = None                    # 当前观测 (n_agents, obs_size)
    state: Any = None                  # 全局状态 (state_size,)
    avail_actions: Any = None          # 可用动作掩码 (n_agents, n_actions)
    actions: Any = None                # 当前动作 (n_agents,)
    reward: float = 0.0
    terminated: bool = False
    info: dict = field(default_factory=dict)
    return_history: list = field(default_factory=list)   # 回报历史（reward 序列）
    recent_metrics: dict = field(default_factory=dict)   # 近期训练指标（最新值）
    metrics_history: dict = field(default_factory=dict)  # 训练指标历史（{指标名: [历史值]}）


class RewardShaper(ABC):
    """奖励塑形接口：在 env.step 之后、写入 replay buffer 之前，调整奖励信号。"""

    @abstractmethod
    def shape_reward(self, obs, actions, reward, terminated, info,
                     context: InterventionContext | None = None) -> float:
        """返回 reshape 后的奖励。"""


class ActionOverrider(ABC):
    """动作覆盖接口：在 mac 选出动作之后、env.step 之前，覆盖（部分）智能体的动作。"""

    @abstractmethod
    def override_actions(self, obs, proposed_actions, avail_actions,
                         context: InterventionContext | None = None):
        """返回覆盖后的动作，形状需与 proposed_actions 一致。"""


class SubgoalPlanner(ABC):
    """子目标规划接口：根据观测 / 全局状态，规划当前要达成的子目标。

    子目标的消费方式（拼接到观测 / 作为内在奖励 / 作为辅助训练目标）尚未确定，
    这里只定义「规划入口」，返回值形状由具体实现自行定义。
    """

    @abstractmethod
    def plan_subgoals(self, obs, state,
                      context: InterventionContext | None = None):
        """返回子目标表示。"""


class InterventionGate(ABC):
    """介入决策接口：根据训练数据判断「是否让 LLM 介入」某个模块。

    module 取值为 "reward_shaping" / "action_override" / "subgoal"。
    """

    @abstractmethod
    def should_intervene(self, module: str,
                         context: InterventionContext | None = None) -> bool:
        """返回 True 表示允许 LLM 介入该模块。"""
