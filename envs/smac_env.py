import numpy as np
from smac.env import StarCraft2Env


class SMACEnv:
    """
    A lightweight wrapper around SMAC's StarCraft2Env.

    The wrapper exposes a simple MARL interface:

        reset()
        step(actions)

        get_obs()
        get_state()
        get_avail_actions(

        get_env_info()

    The interface is designed for algorithms such as:
        - IQL
        - VDN
        - QMIX
        - MAPPO

    Default scenario:
        3m
    """

    def __init__(
        self,
        map_name="3m",
        seed=None,
        step_mul=8,
        difficulty="7",
        game_version=None,
        replay_dir=None,
    ):
        self.map_name = map_name
        self.seed = seed
        self.step_mul = step_mul
        self.difficulty = difficulty
        self.game_version = game_version
        self.replay_dir = replay_dir

        # ---------------------------------------------------------
        # Create the original SMAC environment
        # ---------------------------------------------------------
        self.env = StarCraft2Env(
            map_name=self.map_name,
            step_mul=self.step_mul,
            difficulty=self.difficulty,
            game_version=self.game_version,
            replay_dir=self.replay_dir,
            seed=self.seed,
        )

        # ---------------------------------------------------------
        # Get environment information
        # ---------------------------------------------------------
        self.n_agents = self.env.n_agents
        self.n_actions = self.env.n_actions
        self.episode_limit = self.env.episode_limit

        self.obs_size = self.env.get_obs_size()
        self.state_size = self.env.get_state_size()

        # Useful aliases for MARL code
        self.num_agents = self.n_agents
        self.action_dim = self.n_actions

        self.obs = None
        self.state = None
        self.avail_actions = None

        self.last_reward = 0.0
        self.terminated = False

    # =============================================================
    # Basic environment interface
    # =============================================================

    def reset(self):
        """
        Reset the StarCraft II environment.

        Returns
        -------
        obs : np.ndarray
            Shape:
                (n_agents, obs_size)

        state : np.ndarray
            Shape:
                (state_size,)

        avail_actions : np.ndarray
            Shape:
                (n_agents, n_actions)
        """

        self.env.reset()

        self.terminated = False
        self.last_reward = 0.0

        self.obs = self.get_obs()
        self.state = self.get_state()
        self.avail_actions = self.get_avail_actions()

        return self.obs.copy(), self.state.copy(), self.avail_actions.copy()

    def step(self, actions):
        """
        Execute one environment step.

        Parameters
        ----------
        actions : array-like
            Shape:
                (n_agents,)

            Each element is an integer action index.

        Returns
        -------
        obs : np.ndarray
            Shape:
                (n_agents, obs_size)

        state : np.ndarray
            Shape:
                (state_size,)

        reward : float
            Global cooperative reward.

        terminated : bool
            Whether the episode has ended.

        info : dict
            Additional environment information.
        """

        actions = np.asarray(actions, dtype=np.int64)

        if actions.shape != (self.n_agents,):
            raise ValueError(
                f"Expected actions shape "
                f"({self.n_agents},), got {actions.shape}"
            )

        # ---------------------------------------------------------
        # Check action validity
        # ---------------------------------------------------------
        avail_actions = self.get_avail_actions()

        for agent_id, action in enumerate(actions):
            if action < 0 or action >= self.n_actions:
                raise ValueError(
                    f"Invalid action {action} for agent {agent_id}. "
                    f"Valid range: [0, {self.n_actions - 1}]"
                )

            if avail_actions[agent_id, action] == 0:
                raise ValueError(
                    f"Agent {agent_id} selected unavailable action "
                    f"{action}."
                )

        # ---------------------------------------------------------
        # SMAC step
        # ---------------------------------------------------------
        reward, terminated, info = self.env.step(actions.tolist())

        self.last_reward = float(reward)
        self.terminated = bool(terminated)

        # ---------------------------------------------------------
        # Get next observation/state/action mask
        # ---------------------------------------------------------
        if not self.terminated:
            self.obs = self.get_obs()
            self.state = self.get_state()
            self.avail_actions = self.get_avail_actions()
        else:
            # SMAC may still allow queries after termination, but
            # keeping the latest valid information is convenient.
            self.obs = self.get_obs()
            self.state = self.get_state()
            self.avail_actions = self.get_avail_actions()

        info = {} if info is None else dict(info)

        # Useful standard information
        info["battle_won"] = self.get_battle_won()

        return (
            self.obs.copy(),
            self.state.copy(),
            float(reward),
            bool(terminated),
            info,
        )

    # =============================================================
    # Observation
    # =============================================================

    def get_obs(self):
        """
        Get observations for all agents.

        Returns
        -------
        np.ndarray
            Shape:
                (n_agents, obs_size)
        """

        obs = self.env.get_obs()

        obs = np.asarray(obs, dtype=np.float32)

        expected_shape = (self.n_agents, self.obs_size)

        if obs.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected observation shape: "
                f"expected {expected_shape}, got {obs.shape}"
            )

        return obs

    # =============================================================
    # Global state
    # =============================================================

    def get_state(self):
        """
        Get the global state.

        Returns
        -------
        np.ndarray
            Shape:
                (state_size,)
        """

        state = self.env.get_state()

        state = np.asarray(state, dtype=np.float32)

        if state.shape != (self.state_size,):
            raise RuntimeError(
                f"Unexpected state shape: "
                f"expected {(self.state_size,)}, got {state.shape}"
            )

        return state

    # =============================================================
    # Available actions
    # =============================================================

    def get_avail_actions(self):
        """
        Get available-action mask for every agent.

        Returns
        -------
        np.ndarray
            Shape:
                (n_agents, n_actions)

            Values:
                1 -> action is available
                0 -> action is unavailable
        """

        avail_actions = self.env.get_avail_actions()

        avail_actions = np.asarray(
            avail_actions,
            dtype=np.float32,
        )

        expected_shape = (self.n_agents, self.n_actions)

        if avail_actions.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected available-action shape: "
                f"expected {expected_shape}, "
                f"got {avail_actions.shape}"
            )

        return avail_actions

    # =============================================================
    # Environment information
    # =============================================================

    def get_env_info(self):
        """
        Return environment information.

        This dictionary is convenient for constructing networks,
        replay buffers, and training loops.
        """

        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "obs_shape": self.obs_size,
            "state_shape": self.state_size,
            "episode_limit": self.episode_limit,
            "map_name": self.map_name,
        }

    # =============================================================
    # Battle information
    # =============================================================

    def get_battle_won(self):
        """
        Return whether the latest episode was won.

        SMAC's battle_won() returns a dictionary such as:

            {"won": True}

        """

        try:
            result = self.env.get_stats()

            if isinstance(result, dict):
                if "won" in result:
                    return bool(result["won"])

        except Exception:
            pass

        return False

    # =============================================================
    # Episode statistics
    # =============================================================

    def get_stats(self):
        """
        Return SMAC episode statistics.
        """

        try:
            return self.env.get_stats()
        except Exception:
            return {}

    # =============================================================
    # Close
    # =============================================================

    def close(self):
        """
        Close the StarCraft II process.
        """

        self.env.close()

    # =============================================================
    # Context manager support
    # =============================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
