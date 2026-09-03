import argparse
import json
from pathlib import Path

import torch as th
from omegaconf import OmegaConf

from controllers.mac import BasicMAC
from learners.learner import QLearner
from runners.episode_runner import EpisodeRunner
from envs.smac_env import SMACEnv
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.logger import Logger


def load_config(cli) -> OmegaConf:
    args = OmegaConf.load(cli.config)

    # 命令行覆盖（None 表示使用 yaml 里的值）
    if cli.map is not None:
        args.env_args.map_name = cli.map
    if cli.seed is not None:
        args.env_args.seed = cli.seed
    if cli.max_steps is not None:
        args.t_max = cli.max_steps

    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--map", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    cli = parser.parse_args()

    args = load_config(cli)
    logger = Logger(log_dir=f"logs/{args.env_args.map_name}_s{args.env_args.seed}")

    # ---- 运行时信息：设备 ----
    args.device = "cuda" if th.cuda.is_available() else "cpu"

    # ---- 先建一个临时环境读取维度信息（MAC/QMixer 构造需要 n_agents/state_shape）----
    env_args = OmegaConf.to_container(args.env_args, resolve=True)
    tmp_env = SMACEnv(**env_args)
    env_info = tmp_env.get_env_info()
    tmp_env.close()

    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.agent.n_actions = env_info["n_actions"]

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": env_info["n_actions"], "group": "agents"},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
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
    runner = EpisodeRunner(args, logger, mac)
    runner.setup(scheme, groups, preprocess=preprocess)
    learner = QLearner(mac, logger, args)

    if args.device != "cpu":
        learner.cuda()

    # ---- checkpoint ----
    ckpt_dir = Path(args.checkpoint_path) / f"{env_info['map_name']}_s{args.env_args.seed}"

    episode = 0
    if args.resume and (ckpt_dir / "agent.th").exists():
        meta_path = ckpt_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            episode = meta.get("episode", 0)
            runner.t_env = meta.get("t_env", 0)
        learner.load_models(str(ckpt_dir))
        logger.info(f"Resumed from checkpoint: episode={episode}, t_env={runner.t_env}")

    logger.info("=" * 60)
    logger.info(f"config={cli.config}  map={env_info['map_name']}  "
                f"n_agents={env_info['n_agents']}  n_actions={env_info['n_actions']}")
    logger.info(f"obs_shape={env_info['obs_shape']}  state_shape={env_info['state_shape']}  "
                f"episode_limit={env_info['episode_limit']}")
    logger.info(f"obs_last_action={args.obs_last_action}  obs_agent_id={args.obs_agent_id}  "
                f"input_shape={args.agent.input_shape}")
    logger.info(f"device={args.device}  mixer={args.mixer}  t_max={args.t_max}  "
                f"buffer_size={args.buffer_size}  batch_size={args.batch_size}")
    logger.info("=" * 60)

    last_eval_t_env = runner.t_env
    while runner.t_env < args.t_max:
        episode_batch = runner.run(test_mode=False)
        episode_return = float(episode_batch["reward"].sum())

        # 存 buffer 前先搬到 CPU
        episode_batch.to("cpu")
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            batch = buffer.sample(args.batch_size)

            # 截断到实际填满的时间步，避免在 padding 上空跑
            max_ep_t = int(batch.max_t_filled().item())
            batch = batch[:, :max_ep_t]

            batch.to(args.device)
            learner.train(batch, runner.t_env, episode)

        if episode % 10 == 0:
            logger.info(f"episode {episode:5d} | t_env {runner.t_env:8d} | return {episode_return:8.2f}")

        episode += 1

        # ---- 周期性评估（贪婪策略，不消费训练预算）----
        if runner.t_env - last_eval_t_env >= args.evaluate_interval:
            logger.info(f"--- Evaluating {args.test_nepisode} episodes ---")
            for _ in range(args.test_nepisode):
                runner.run(test_mode=True)
            last_eval_t_env = runner.t_env
            logger.info("--- Evaluation done ---")

        if episode % args.save_interval == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            learner.save_models(str(ckpt_dir))
            (ckpt_dir / "meta.json").write_text(
                json.dumps({"episode": episode, "t_env": runner.t_env})
            )
            logger.info(f"Saved checkpoint at episode {episode}, t_env {runner.t_env}")

    # 训练结束再存一份
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    learner.save_models(str(ckpt_dir))
    (ckpt_dir / "meta.json").write_text(
        json.dumps({"episode": episode, "t_env": runner.t_env})
    )

    runner.close_env()
    logger.save()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
