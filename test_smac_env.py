from envs.smac_env import SMACEnv


def main():

    env = SMACEnv(
        map_name="3m",
        seed=1,
    )

    print("=" * 60)
    print("Environment information")
    print("=" * 60)

    env_info = env.get_env_info()

    for key, value in env_info.items():
        print(f"{key:20s}: {value}")

    print("\nReset environment...")

    obs, state, avail_actions = env.reset()

    print("\nAfter reset:")
    print("obs shape          :", obs.shape)
    print("state shape        :", state.shape)
    print("avail_actions shape:", avail_actions.shape)

    print("\nTaking one random valid action...")

    actions = []

    for agent_id in range(env.n_agents):

        available = avail_actions[agent_id]

        valid_actions = [
            action
            for action in range(env.n_actions)
            if available[action] > 0
        ]

        action = valid_actions[0]

        actions.append(action)

    print("actions:", actions)

    next_obs, next_state, reward, terminated, info = env.step(actions)

    print("\nAfter step:")
    print("next_obs shape   :", next_obs.shape)
    print("next_state shape :", next_state.shape)
    print("reward           :", reward)
    print("terminated       :", terminated)
    print("info             :", info)

    env.close()

    print("\nSMACEnv test finished.")


if __name__ == "__main__":
    main()
