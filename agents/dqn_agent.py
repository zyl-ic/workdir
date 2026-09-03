from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from networks.dueling_dqn import DuelingDQN


class DQNAgent:
    """
    DQN Agent — Double DQN + Dueling + PER + NoisyNet + Soft update。
    只支持离散动作空间。

    适配 SMAC 环境（envs.smac_env.SMACEnv）：观测是扁平向量，动作空间齐次。
    用法：

        env_info = env.get_env_info()
        agent = DQNAgent(
            obs_dim=env_info["obs_shape"],
            n_actions=env_info["n_actions"],
            hidden_dim=64,
            gamma=0.99,
        )
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int,
        gamma: float,
        use_noisy: bool = True,
        noisy_std: float = 0.5,
    ):
        # ---- 保存超参 ----
        self.hidden_dim = hidden_dim
        self.gamma      = gamma
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_noisy  = use_noisy
        self.noisy_std  = noisy_std

        # ---- 观测 / 动作维度（SMAC 齐次空间，直接用维度） ----
        self.input_dim = int(obs_dim)
        self.action_dim = int(n_actions)

        # ---- 网络 ----
        self.online_net = DuelingDQN(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.action_dim,
            use_noisy=use_noisy,
            noisy_std=noisy_std,
        ).to(self.device)

        self.target_net = DuelingDQN(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.action_dim,
            use_noisy=use_noisy,
            noisy_std=noisy_std,
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.best_record = 0.0

    # =================================================
    # clone
    # =================================================

    def clone(self) -> "DQNAgent":
        frozen = DQNAgent.__new__(DQNAgent)

        frozen.hidden_dim = self.hidden_dim
        frozen.gamma      = self.gamma
        frozen.use_noisy  = self.use_noisy
        frozen.noisy_std  = self.noisy_std
        frozen.device     = self.device
        frozen.input_dim  = self.input_dim
        frozen.action_dim = self.action_dim

        frozen.online_net = DuelingDQN(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim,
            output_dim=self.action_dim,
            use_noisy=self.use_noisy, noisy_std=self.noisy_std,
        ).to(self.device)
        frozen.online_net.load_state_dict(self.online_net.state_dict())

        frozen.target_net = DuelingDQN(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim,
            output_dim=self.action_dim,
            use_noisy=self.use_noisy, noisy_std=self.noisy_std,
        ).to(self.device)
        frozen.target_net.load_state_dict(self.target_net.state_dict())

        frozen.best_record = 0.0

        frozen.eval_mode()
        for p in frozen.online_net.parameters():
            p.requires_grad = False
        for p in frozen.target_net.parameters():
            p.requires_grad = False

        return frozen

    # =================================================
    # 动作选择
    # =================================================

    @torch.no_grad()
    def select_action(self, state, action_mask=None):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        q_values = self.online_net(state).squeeze(0)

        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            q_values[~mask] = -torch.inf

        return int(torch.argmax(q_values).item())

    # =================================================
    # 随机动作
    # =================================================

    def random_action(self, action_mask=None):
        if action_mask is not None:
            valid = np.where(action_mask == 1)[0]
        else:
            valid = np.arange(self.action_dim)
        if len(valid) == 0:
            return 0
        return int(np.random.choice(valid))

    # =================================================
    # 更新
    # =================================================

    def update(self, batch):
        states      = torch.as_tensor(batch["obs"],         dtype=torch.float32, device=self.device)
        actions     = torch.as_tensor(batch["actions"],     dtype=torch.long, device=self.device)
        rewards     = torch.as_tensor(batch["rewards"],     dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_obs"],    dtype=torch.float32, device=self.device)
        next_masks  = batch.get("next_action_masks")
        if next_masks is not None:
            next_masks = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)

        # ---- 当前 Q(s,a) ----
        q = self.online_net(states)
        current_q = q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ---- Double DQN target ----
        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            if next_masks is not None:
                next_q_online[~next_masks] = -torch.inf
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_net(next_states)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)

        return current_q, next_q, rewards

    # =================================================
    # 模式切换
    # =================================================

    def eval_mode(self):
        self.online_net.eval()

    def train_mode(self):
        self.online_net.train()

    # =================================================
    # 保存 / 加载
    # =================================================

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online":    self.online_net.state_dict(),
                "target":    self.target_net.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(checkpoint["online"])
        self.target_net.load_state_dict(checkpoint["target"])
        self.online_net.to(self.device)
        self.target_net.to(self.device)
