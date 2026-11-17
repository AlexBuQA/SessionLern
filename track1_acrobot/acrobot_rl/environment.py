"""Noise-controlled Acrobot environments used by the experiment."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor


class TorqueNoiseRandomizer(gym.Wrapper):
    """Sample torque noise once per episode; all other dynamics stay unchanged."""

    def __init__(self, env: gym.Env, low: float, high: float):
        super().__init__(env)
        if not 0.0 <= low <= high:
            raise ValueError("Expected 0 <= low <= high for torque noise")
        self.low = float(low)
        self.high = float(high)
        self._noise_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Independent stream: changing noise must not change initial states.
            self._noise_rng = np.random.default_rng(seed + 1_000_003)
        noise = float(self._noise_rng.uniform(self.low, self.high))
        self.unwrapped.torque_noise_max = noise
        observation, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["torque_noise_max"] = noise
        return observation, info


def make_acrobot(
    *,
    torque_noise: float = 0.0,
    randomization_range: tuple[float, float] | None = None,
    render_mode: str | None = None,
    monitor: bool = False,
) -> gym.Env:
    env = gym.make("Acrobot-v1", render_mode=render_mode)
    env.unwrapped.torque_noise_max = float(torque_noise)
    if randomization_range is not None:
        env = TorqueNoiseRandomizer(env, *randomization_range)
    if monitor:
        env = Monitor(env)
    return env
