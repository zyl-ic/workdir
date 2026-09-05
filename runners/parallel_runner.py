from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th

from llm.base import InterventionContext
from llm.manager import LLMAssistManager

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
class ParallelRunner:

    def __init__(self, args, logger, mac, llm=None):
        self.args = args
        self.logger = logger
        self.mac = mac
        self.llm = llm if llm is not None else LLMAssistManager()
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs (each env uses a different seed)
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            env_args = dict(self.args.env_args)
            env_args["seed"] = env_args.get("seed", 0) + i
            self.ps.append(Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_args))),
            ))

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0
        self.t_env = 0
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def _llm_ctx(self, test_mode, **extra):
        return InterventionContext(
            t_env=self.t_env,
            test_mode=test_mode,
            return_history=self.test_returns if test_mode else self.train_returns,
            recent_metrics=dict(self.test_stats if test_mode else self.train_stats),
            **extra,
        )

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
        pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
        self.batch.update(pre_transition_data, ts=0)
        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        # ---- LLM hook: subgoal planning (per env) ----
        self.current_subgoals = [
            self.llm.plan_subgoals(
                self.batch["obs"][b, 0], self.batch["state"][b, 0],
                self._llm_ctx(test_mode),
            )
            for b in range(self.batch_size)
        ]

        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        final_env_infos = []

        while True:
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            if not envs_not_terminated:
                break

            # select actions (bs only affects action selection; forward still processes the whole batch)
            if self.args.learner == "mappo":
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                         bs=envs_not_terminated, test_mode=test_mode,
                                                         return_probs=True)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                  bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()
            policy_actions = cpu_actions.copy()  # policy-sampled actions (before override)

            # ---- LLM hook: action override (per env) ----
            for i, b_idx in enumerate(envs_not_terminated):
                obs = self.batch["obs"][b_idx, self.t]
                avail = self.batch["avail_actions"][b_idx, self.t]
                cpu_actions[i] = self.llm.override_actions(
                    obs, cpu_actions[i], avail,
                    self._llm_ctx(test_mode, obs=obs, avail_actions=avail, actions=cpu_actions[i]),
                )

            # log prob of executed actions (recomputed after override)
            cpu_log_probs = None
            if self.args.learner == "mappo":
                behavior = th.as_tensor(cpu_actions, device=probs.device)
                cpu_log_probs = self.mac.log_probs_of(probs, behavior).detach().to("cpu").numpy()

            # write executed actions (after override), policy actions, and log_prob (MAPPO on-policy)
            if self.args.learner == "mappo":
                self.batch.update({"actions": cpu_actions[:, None],
                                   "policy_actions": policy_actions[:, None],
                                   "log_prob": cpu_log_probs[:, None]},
                                  bs=envs_not_terminated, ts=self.t, mark_filled=False)
            else:
                self.batch.update({"actions": cpu_actions[:, None],
                                   "policy_actions": policy_actions[:, None]},
                                  bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # send actions to each un-terminated env
            for i, b_idx in enumerate(envs_not_terminated):
                self.parent_conns[b_idx].send(("step", cpu_actions[i]))

            # receive returns for the current step and next-step observations
            post_transition_data = {"reward": [], "terminated": []}
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            for b_idx in envs_not_terminated:
                data = self.parent_conns[b_idx].recv()

                # ---- LLM hook: reward shaping ----
                obs = self.batch["obs"][b_idx, self.t]
                avail = self.batch["avail_actions"][b_idx, self.t]
                act = self.batch["actions"][b_idx, self.t, :, 0]
                shaped_reward = self.llm.shape_reward(
                    obs, act, data["reward"], data["terminated"], data["info"],
                    self._llm_ctx(test_mode, obs=obs, avail_actions=avail, actions=act,
                                  reward=data["reward"], terminated=data["terminated"], info=data["info"]),
                )

                post_transition_data["reward"].append((shaped_reward,))
                episode_returns[b_idx] += shaped_reward
                episode_lengths[b_idx] += 1
                if not test_mode:
                    self.env_steps_this_run += 1

                env_terminated = False
                if data["terminated"]:
                    final_env_infos.append(data["info"])
                if data["terminated"] and not data["info"].get("episode_limit", False):
                    env_terminated = True
                terminated[b_idx] = data["terminated"]
                post_transition_data["terminated"].append((env_terminated,))

                pre_transition_data["state"].append(data["state"])
                pre_transition_data["avail_actions"].append(data["avail_actions"])
                pre_transition_data["obs"].append(data["obs"])

            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.t += 1
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))
        env_stats = []
        for parent_conn in self.parent_conns:
            env_stats.append(parent_conn.recv())

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
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
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            reward, terminated, env_info = env.step(actions)
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)"""

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
