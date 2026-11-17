import gymnasium as gym
import numpy as np

from acrobot_rl.environment import TorqueNoiseRandomizer, make_acrobot


def test_fixed_noise_and_seed_are_reproducible():
    env_a = make_acrobot(torque_noise=0.2)
    env_b = make_acrobot(torque_noise=0.2)
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)
    for _ in range(10):
        step_a = env_a.step(1)
        step_b = env_b.step(1)
        np.testing.assert_allclose(step_a[0], step_b[0])
        assert step_a[1:] == step_b[1:]
    env_a.close()
    env_b.close()


def test_randomizer_changes_only_episode_noise():
    env = TorqueNoiseRandomizer(gym.make("Acrobot-v1"), 0.0, 0.3)
    values = []
    _, info = env.reset(seed=7)
    values.append(info["torque_noise_max"])
    assert env.unwrapped.torque_noise_max == info["torque_noise_max"]
    for _ in range(7):
        _, info = env.reset()
        values.append(info["torque_noise_max"])
    assert len(set(values)) > 1
    _, repeated = env.reset(seed=7)
    assert repeated["torque_noise_max"] == values[0]
    assert all(0.0 <= value <= 0.3 for value in values)
    env.close()
