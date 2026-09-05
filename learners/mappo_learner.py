import os

import torch as th
from torch.optim import RMSprop

from components.episode_buffer import EpisodeBatch
from networks.mappo import CentralVCritic


class MAPPOLearner:
    """MAPPO（Multi-Agent PPO）。

    - actor：mac（agent_output_type = "pi_logits"，输出策略）
    - critic：CentralVCritic（中心化 value，输出标量 V(s)）
    - 训练：GAE 算 advantage → PPO clipped objective 更新 actor；MSE 更新 critic。
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger

        self.critic = CentralVCritic(scheme, args)

        self.actor_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())

        self.actor_optimiser = RMSprop(params=self.actor_params, lr=args.lr,
                                       alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr,
                                        alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]              # (bs, T, 1)
        actions = batch["actions"][:, :-1]             # 实际执行的动作（behavior）
        terminated = batch["terminated"][:, :-1].float()  # (bs, T, 1)
        mask = batch["filled"][:, :-1].float()         # (bs, T, 1)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]  # (bs, T, n_agents, n_actions)
        states = batch["state"]                         # (bs, T+1, state_size)

        # ---- critic 前向：V(s_t), t=0..T ----
        values = self.critic(states)  # (bs, T+1, 1)

        # ---- GAE raw advantage ----
        raw_advantages = self._compute_gae(rewards, values, terminated, mask).detach()  # (bs, T, 1)

        # value target = V + raw_advantage（critic 用 raw，即 return 估计）
        value_target = (values[:, :-1] + raw_advantages).detach()  # (bs, T, 1)

        # normalized advantage（actor 用，只对有效数据标准化）
        advantages = raw_advantages
        if self.args.get("advantage_norm", False):
            n = mask.sum()
            adv_mean = (raw_advantages * mask).sum() / n
            adv_std = (((raw_advantages - adv_mean) ** 2 * mask).sum() / n).sqrt()
            advantages = (raw_advantages - adv_mean) / (adv_std + 1e-8)

        # ---- on-policy：old log prob 是采样时记录的 ----
        old_log_probs = batch["log_prob"][:, :-1].squeeze(-1)  # (bs, T, n_agents)

        mask_agents = mask.repeat(1, 1, self.n_agents)  # (bs, T, n_agents)
        adv = advantages.repeat(1, 1, self.n_agents)    # (bs, T, n_agents)

        actor_loss = None
        entropy_mean = None
        critic_loss = None
        for _ in range(self.args.ppo_epochs):
            # ---- actor 更新（用实际执行的 action + normalized advantage）----
            new_log_probs, entropy = self._actor_log_probs(batch, actions, avail_actions, return_entropy=True)
            ratio = th.exp(new_log_probs - old_log_probs)  # (bs, T, n_agents)

            surr1 = ratio * adv
            surr2 = th.clamp(ratio, 1 - self.args.clip_eps, 1 + self.args.clip_eps) * adv
            policy_loss = -th.min(surr1, surr2)

            entropy_mean = (entropy * mask_agents).sum() / mask_agents.sum()

            actor_loss = (policy_loss * mask_agents).sum() / mask_agents.sum() \
                - self.args.entropy_coef * entropy_mean

            self.actor_optimiser.zero_grad()
            actor_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
            self.actor_optimiser.step()

            # ---- critic 更新（raw value target），每个 epoch 也更新 ----
            values_pred = self.critic(states)[:, :-1]  # (bs, T, 1)
            critic_loss = ((values_pred - value_target) ** 2 * mask).sum() / mask.sum()

            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()

        # ---- 日志 ----
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
            self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            self.logger.log_stat("actor_grad_norm", grad_norm, t_env)
            self.logger.log_stat("entropy_mean", entropy_mean.item(), t_env)
            self.logger.log_stat("advantage_mean", (raw_advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("ratio_mean", (ratio * mask_agents).sum().item() / mask_agents.sum().item(), t_env)
            self.log_stats_t = t_env

    def _actor_log_probs(self, batch, actions, avail_actions, return_entropy=False):
        """当前策略（纯 π）下，实际执行动作的 log 概率。"""
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t, test_mode=True)  # 纯策略，无 epsilon floor
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # (bs, T, n_agents, n_actions)

        # 屏蔽不可用动作并归一化（clamp + 全 0 行 fallback，避免死亡 agent 的 0/0）
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        row_sum = mac_out.sum(dim=-1, keepdim=True)
        mac_out = th.where(row_sum > 0, mac_out, th.ones_like(mac_out) / mac_out.size(-1))

        dist = th.distributions.Categorical(mac_out)
        log_probs = dist.log_prob(actions.squeeze(-1))  # (bs, T, n_agents)

        if return_entropy:
            return log_probs, dist.entropy()  # entropy: (bs, T, n_agents)
        return log_probs

    def _compute_gae(self, rewards, values, terminated, mask):
        """GAE：advantage = Σ (γλ)^k δ_{t+k}，δ_t = r_t + γ(1-d_t)V_{t+1} - V_t。"""
        bs = rewards.size(0)
        T = rewards.size(1)
        advantages = th.zeros(bs, T, 1, device=rewards.device)
        running = 0.0
        for t in reversed(range(T)):
            delta = rewards[:, t] + self.args.gamma * (1 - terminated[:, t]) * values[:, t + 1] - values[:, t]
            running = delta + self.args.gamma * self.args.gae_lambda * (1 - terminated[:, t]) * running
            running = running * mask[:, t]  # padding 时归零，不污染后续
            advantages[:, t] = running
        return advantages

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.actor_optimiser.state_dict(), "{}/actor_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage, weights_only=True))
        self.actor_optimiser.load_state_dict(
            th.load("{}/actor_opt.th".format(path), map_location=lambda storage, loc: storage, weights_only=True))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage, weights_only=True))
