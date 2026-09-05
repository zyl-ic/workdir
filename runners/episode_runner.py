from envs.smac_env import SMACEnv
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th

from llm.base import InterventionContext
from llm.manager import LLMAssistManager


class EpisodeRunner:
    def __init__(self, args, logger, mac, llm=None):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1
        self.mac = mac
        self.llm = llm if llm is not None else LLMAssistManager()

        self.env = SMACEnv(**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)

    def _llm_ctx(self, test_mode, **extra):
        """构造 LLM 介入决策所需的上下文（训练数据）。"""
        return InterventionContext(
            t_env=self.t_env,
            test_mode=test_mode,
            return_history=self.test_returns if test_mode else self.train_returns,
            recent_metrics=dict(self.test_stats if test_mode else self.train_stats),
            **extra,
        )

    def get_env_info(self):
        return self.env.get_env_info()


    def close_env(self):
        self.env.close()

    def reset(self):
        self.t = 0
        self.batch = self.new_batch()
        self.env.reset()

    def run(self, test_mode=False):
        self.reset()

        # ---- LLM hook：episode 开始时规划子目标（消费方式待定，暂存）----
        self.current_subgoals = self.llm.plan_subgoals(
            self.env.get_obs(), self.env.get_state(), self._llm_ctx(test_mode)
        )

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            state = self.env.get_state()
            avail_actions = self.env.get_avail_actions()
            obs = self.env.get_obs()

            pre_transition_data = {
                "state": [state],
                "avail_actions": [avail_actions],
                "obs": [obs]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            if self.args.learner == "mappo":
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                         test_mode=test_mode, return_probs=True)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            actions = actions.detach().cpu().numpy()
            policy_actions = actions.copy()  # policy 采样的动作（override 前）

            # ---- LLM hook：动作覆盖（默认原样返回）----
            actions[0] = self.llm.override_actions(
                obs, actions[0], avail_actions,
                self._llm_ctx(test_mode, obs=obs, avail_actions=avail_actions, actions=actions[0]),
            )

            # 实际执行动作的 log prob（override 后重算，保证与 behavior_action 对齐）
            log_probs = None
            if self.args.learner == "mappo":
                behavior = th.as_tensor(actions, device=probs.device)
                log_probs = self.mac.log_probs_of(probs, behavior).detach().cpu().numpy()

            reward, terminated, info = self.env.step(actions[0])

            # ---- LLM hook：奖励塑形（默认原样返回）----
            reward = self.llm.shape_reward(
                obs, actions[0], reward, terminated, info,
                self._llm_ctx(test_mode, obs=obs, avail_actions=avail_actions, actions=actions[0],
                              reward=reward, terminated=terminated, info=info),
            )

            episode_return += reward

            post_transition_data = {
                "actions": actions,                 # 实际执行的动作（behavior）
                "policy_actions": policy_actions,   # policy 采样的动作（override 前）
                "reward": [(reward,)],
                "terminated": [(terminated != info.get("episode_limit", False),)],
            }
            if self.args.learner == "mappo":
                post_transition_data["log_prob"] = log_probs

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        last_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": last_actions.detach().cpu().numpy()}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + info.get(k, 0) for k in set(cur_stats) | set(info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v/stats["n_episodes"], self.t_env)
        stats.clear()
