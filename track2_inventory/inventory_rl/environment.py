"""A finite-horizon, lost-sales inventory environment with one-day lead time."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class InventoryEnv(gym.Env):
    """Order one product; unmet demand is lost and excess stock has a cost.

    Observation: [inventory_position, previous_demand_capped_at_25].
    Inventory position is stock on hand after sales plus the incoming order.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        demand_mean: float = 6.0,
        horizon: int = 30,
        max_inventory: int = 30,
        max_order: int = 15,
        price: float = 12.0,
        purchase_cost: float = 5.0,
        holding_cost: float = 1.0,
        stockout_cost: float = 8.0,
    ):
        super().__init__()
        if demand_mean <= 0 or horizon <= 0:
            raise ValueError("demand_mean and horizon must be positive")
        self.demand_mean = float(demand_mean)
        self.horizon = int(horizon)
        self.max_inventory = int(max_inventory)
        self.max_order = int(max_order)
        self.price, self.purchase_cost = float(price), float(purchase_cost)
        self.holding_cost, self.stockout_cost = float(holding_cost), float(stockout_cost)
        self.action_space = spaces.Discrete(self.max_order + 1)
        self.observation_space = spaces.MultiDiscrete([self.max_inventory + 1, 26])
        self.day = 0
        self.on_hand = 0
        self.incoming = 0
        self.previous_demand = 0
        self.initial_inventory = 0

    def _observation(self) -> np.ndarray:
        inventory_position = self.on_hand + self.incoming
        return np.array([inventory_position, min(self.previous_demand, 25)], dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "demand_mean" in options:
            demand_mean = float(options["demand_mean"])
            if demand_mean <= 0:
                raise ValueError("demand_mean must be positive")
            self.demand_mean = demand_mean
        self.day = 0
        self.initial_inventory = int(options.get(
            "initial_inventory", self.np_random.integers(0, 11)
        ))
        if not 0 <= self.initial_inventory <= self.max_inventory:
            raise ValueError("initial_inventory is outside storage capacity")
        self.on_hand = self.initial_inventory
        self.incoming = 0
        self.previous_demand = 0
        return self._observation(), {
            "demand_mean": self.demand_mean,
            "initial_inventory": self.initial_inventory,
        }

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        requested_order = int(action)
        available = self.on_hand + self.incoming
        demand = int(self.np_random.poisson(self.demand_mean))
        sales = min(available, demand)
        lost_sales = demand - sales
        stock_after = available - sales
        actual_order = min(requested_order, self.max_inventory - stock_after)
        revenue = self.price * sales
        purchase = self.purchase_cost * actual_order
        holding = self.holding_cost * stock_after
        stockout = self.stockout_cost * lost_sales
        reward = revenue - purchase - holding - stockout

        self.on_hand = stock_after
        self.incoming = actual_order
        self.previous_demand = demand
        self.day += 1
        truncated = self.day >= self.horizon
        info = {
            "day": self.day,
            "available": available,
            "demand": demand,
            "sales": sales,
            "lost_sales": lost_sales,
            "requested_order": requested_order,
            "actual_order": actual_order,
            "inventory_after_sales": stock_after,
            "inventory_position": stock_after + actual_order,
            "revenue": revenue,
            "purchase_cost": purchase,
            "holding_cost": holding,
            "stockout_cost": stockout,
        }
        return self._observation(), float(reward), False, truncated, info


class DemandRandomizer(gym.Wrapper):
    """Change only the episode-level Poisson demand mean."""

    def __init__(self, env: InventoryEnv, low: float, high: float):
        super().__init__(env)
        if not 0 < low <= high:
            raise ValueError("Expected 0 < low <= high")
        self.low, self.high = float(low), float(high)
        self._demand_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._demand_rng = np.random.default_rng(seed + 2_000_033)
        demand_mean = float(self._demand_rng.uniform(self.low, self.high))
        options = dict(options or {})
        options["demand_mean"] = demand_mean
        observation, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["episode_demand_mean"] = demand_mean
        return observation, info
