"""LLM 辅助训练的总控：持有各 LLM 能力与介入决策，暴露给 runner 调用的 hook。

默认全部是 no-op（hook 原样透传），不影响现有训练。
后续接入 LLM 时，用 set_* 方法注入具体实现即可，无需再改 runner / learner。
"""
from __future__ import annotations


class LLMAssistManager:
    def __init__(self, cfg=None):
        self.cfg = dict(cfg or {})
        self.reward_shaper = None      # RewardShaper 实现
        self.action_overrider = None   # ActionOverrider 实现
        self.subgoal_planner = None    # SubgoalPlanner 实现
        self.gate = None               # InterventionGate 实现

    # ---- 注入具体实现 ----
    def set_reward_shaper(self, obj):
        self.reward_shaper = obj

    def set_action_overrider(self, obj):
        self.action_overrider = obj

    def set_subgoal_planner(self, obj):
        self.subgoal_planner = obj

    def set_gate(self, obj):
        self.gate = obj

    # ---- 内部开关 ----
    @property
    def enabled(self):
        return bool(self.cfg.get("enabled", False))

    def _active(self, key):
        return self.enabled and bool(self.cfg.get(key, False))

    def _gate_allows(self, module, context):
        # 未启用门控、或没有 gate 实现时，默认放行
        if self.gate is None or not self._active("intervene_gate"):
            return True
        return bool(self.gate.should_intervene(module, context))

    # ---- runner 调用的 hook（默认 no-op）----
    def shape_reward(self, obs, actions, reward, terminated, info, context=None):
        if not self._active("reward_shaping") or self.reward_shaper is None:
            return reward
        if not self._gate_allows("reward_shaping", context):
            return reward
        return self.reward_shaper.shape_reward(obs, actions, reward, terminated, info, context)

    def override_actions(self, obs, proposed_actions, avail_actions, context=None):
        if not self._active("action_override") or self.action_overrider is None:
            return proposed_actions
        if not self._gate_allows("action_override", context):
            return proposed_actions
        return self.action_overrider.override_actions(obs, proposed_actions, avail_actions, context)

    def plan_subgoals(self, obs, state, context=None):
        if not self._active("subgoal") or self.subgoal_planner is None:
            return None
        if not self._gate_allows("subgoal", context):
            return None
        return self.subgoal_planner.plan_subgoals(obs, state, context)
