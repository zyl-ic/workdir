import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch as th
from omegaconf import OmegaConf

from controllers.mac import BasicMAC
from learners.learner import QLearner
from learners.mappo_learner import MAPPOLearner
from runners.episode_runner import EpisodeRunner
from runners.parallel_runner import ParallelRunner
from envs.smac_env import SMACEnv
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.logger import Logger
from llm.base import MetricsHistory
from llm.manager import LLMAssistManager


def load_config(cli) -> OmegaConf:
    # 合并：default 共用 + 地图配置（maps/{map}.yaml）+ 算法配置（qmix.yaml / mappo.yaml）
    base = OmegaConf.load("config/default.yaml")
    map_name = cli.map if cli.map is not None else base.get("map", "3m")
    map_cfg = OmegaConf.load(f"config/maps/{map_name}.yaml")
    algo = OmegaConf.load(cli.config)
    args = OmegaConf.merge(base, map_cfg, algo)

    # 命令行覆盖（None 表示使用 yaml 里的值）
    if cli.seed is not None:
        args.env_args.seed = cli.seed
    if cli.max_steps is not None:
        args.t_max = cli.max_steps
    if cli.resume is not None:
        args.resume = cli.resume

    return args


def set_seed(seed):
    """设置 python / numpy / torch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def get_rng_state():
    state = {
        "torch": th.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    if th.cuda.is_available():
        state["torch_cuda"] = th.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    th.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None:
        th.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/qmix.yaml")
    parser.add_argument("--map", default=None, help="地图名（对应 config/maps/{map}.yaml，默认用 default.yaml 的 map）")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None,
                        help="是否从 checkpoint 续训（默认用 yaml 里的 resume）")
    cli = parser.parse_args()

    args = load_config(cli)
    logger = Logger(log_dir=f"logs/{args.learner}/{args.env_args.map_name}_s{args.env_args.seed}")

    # ---- 运行时信息：设备 ----
    args.device = "cuda" if th.cuda.is_available() else "cpu"

    # ---- 设随机种子（创建任何东西之前）----
    set_seed(args.env_args.seed)

    # ---- 先建一个临时环境读取维度信息（MAC/QMixer 构造需要 n_agents/state_shape）----
    env_args = OmegaConf.to_container(args.env_args, resolve=True)
    tmp_env = SMACEnv(**env_args)
    env_info = tmp_env.get_env_info()
    tmp_env.close()

    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.agent.n_actions = env_info["n_actions"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "policy_actions": {"vshape": (1,), "group": "agents", "dtype": th.long},  # policy 采样动作（override 前）
        "avail_actions": {"vshape": env_info["n_actions"], "group": "agents"},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "log_prob": {"vshape": (1,), "group": "agents"},  # on-policy（MAPPO）采样时记录
    }
    groups = {"agents": env_info["n_agents"]}

    # obs_last_action 需要把 actions 转成 one-hot 存到 actions_onehot
    preprocess = None
    if args.obs_last_action:
        preprocess = {
            "actions": ("actions_onehot", [OneHot(env_info["n_actions"])])
        }

    # 先建 buffer：preprocess 会把 actions_onehot 加进 buffer.scheme
    buffer = ReplayBuffer(
        scheme, groups,
        buffer_size=args.buffer_size,
        max_seq_length=env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu",
    )

    # MAC 用 buffer.scheme（含 actions_onehot），_get_input_shape 才能算对输入维度
    mac = BasicMAC(buffer.scheme, groups, args)

    # ---- LLM 辅助（只预留接口，默认 no-op）----
    llm = LLMAssistManager(OmegaConf.to_container(args.get("llm", {}), resolve=True) or {})

    # 共享的训练指标历史（learner 写，runner 读，供 LLM 介入决策使用）
    metrics_history = MetricsHistory()

    if args.runner == "parallel":
        runner = ParallelRunner(args, logger, mac, llm=llm, metrics_history=metrics_history)
        runner.setup(scheme, groups, preprocess=preprocess)
    else:
        if args.batch_size_run != 1:
            raise ValueError(
                f"EpisodeRunner 要求 batch_size_run=1，当前为 {args.batch_size_run}；"
                f"若要并行采样请用 runner: parallel"
            )
        runner = EpisodeRunner(args, logger, mac, llm=llm, metrics_history=metrics_history)
        runner.setup(scheme, groups, preprocess=preprocess)
    if args.learner == "mappo":
        learner = MAPPOLearner(mac, buffer.scheme, logger, args, metrics_history=metrics_history)
    else:
        learner = QLearner(mac, logger, args, metrics_history=metrics_history)

    if args.device != "cpu":
        learner.cuda()

    # ---- checkpoint ----
    ckpt_dir = Path(args.checkpoint_path) / f"{env_info['map_name']}_s{args.env_args.seed}"

    episode = 0
    if args.resume and (ckpt_dir / "agent.th").exists():
        learner.load_models(str(ckpt_dir))
        if (ckpt_dir / "buffer.th").exists():
            buffer.load(str(ckpt_dir / "buffer.th"))
        if (ckpt_dir / "rng.th").exists():
            set_rng_state(th.load(str(ckpt_dir / "rng.th"), weights_only=False))
        meta_path = ckpt_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            episode = meta.get("episode", 0)
            runner.t_env = meta.get("t_env", 0)
        logger.load()
        logger.info(f"Resumed from checkpoint: episode={episode}, t_env={runner.t_env}")

    logger.info("=" * 60)
    logger.info(f"config={cli.config}  map={env_info['map_name']}  "
                f"n_agents={env_info['n_agents']}  n_actions={env_info['n_actions']}")
    logger.info(f"obs_shape={env_info['obs_shape']}  state_shape={env_info['state_shape']}  "
                f"episode_limit={env_info['episode_limit']}")
    logger.info(f"obs_last_action={args.obs_last_action}  obs_agent_id={args.obs_agent_id}  "
                f"input_shape={args.agent.input_shape}")
    logger.info(f"device={args.device}  learner={args.learner}  t_max={args.t_max}  "
                f"runner={args.runner}  batch_size_run={args.batch_size_run}  "
                f"buffer_size={args.buffer_size}  batch_size={args.get('batch_size', None)}")
    logger.info("=" * 60)

    last_eval_t_env = runner.t_env
    next_log_episode = 0
    next_save_episode = args.save_interval
    try:
        while runner.t_env < args.t_max:
            episode_batch = runner.run(test_mode=False)
            episode_return = float(episode_batch["reward"].sum()) / args.batch_size_run

            # 存 buffer 前先搬到 CPU（两个算法都存，供后续 LLM 等使用）
            episode_batch.to("cpu")
            buffer.insert_episode_batch(episode_batch)

            if args.learner == "mappo":
                # on-policy：直接用刚采的 batch 训练，不从 buffer sample
                max_ep_t = int(episode_batch.max_t_filled().item())
                batch = episode_batch[:, :max_ep_t]
                batch.to(args.device)
                learner.train(batch, runner.t_env, episode)
            elif buffer.can_sample(args.batch_size):
                # off-policy（QMIX）：从 buffer 采样训练
                batch = buffer.sample(args.batch_size)
                max_ep_t = int(batch.max_t_filled().item())
                batch = batch[:, :max_ep_t]
                batch.to(args.device)
                learner.train(batch, runner.t_env, episode)

            if episode >= next_log_episode:
                logger.info(f"episode {episode:5d} | t_env {runner.t_env:8d} | return {episode_return:8.2f}")
                next_log_episode += 10

            episode += args.batch_size_run

            # ---- 周期性评估（贪婪策略，不消费训练预算）----
            if runner.t_env - last_eval_t_env >= args.evaluate_interval:
                n_eval_runs = max(1, args.test_nepisode // args.batch_size_run)
                logger.info(f"--- Evaluating {args.test_nepisode} episodes ---")
                for _ in range(n_eval_runs):
                    runner.run(test_mode=True)
                last_eval_t_env = runner.t_env
                logger.info("--- Evaluation done ---")

            if episode >= next_save_episode:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                learner.save_models(str(ckpt_dir))
                buffer.save(str(ckpt_dir / "buffer.th"))
                th.save(get_rng_state(), str(ckpt_dir / "rng.th"))
                (ckpt_dir / "meta.json").write_text(
                    json.dumps({"episode": episode, "t_env": runner.t_env})
                )
                logger.save()
                logger.info(f"Saved checkpoint at episode {episode}, t_env {runner.t_env}")
                next_save_episode += args.save_interval
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    finally:
        # 保存 checkpoint + 关闭环境（Ctrl+C 中断时也会执行，确保子进程/SC2 被关闭）
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        learner.save_models(str(ckpt_dir))
        buffer.save(str(ckpt_dir / "buffer.th"))
        th.save(get_rng_state(), str(ckpt_dir / "rng.th"))
        (ckpt_dir / "meta.json").write_text(
            json.dumps({"episode": episode, "t_env": runner.t_env})
        )
        runner.close_env()
        logger.save()
        logger.info("Training finished.")

if __name__ == "__main__":
    main()
