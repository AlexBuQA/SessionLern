import numpy as np

from inventory_rl.environment import DemandRandomizer, InventoryEnv


def test_seed_reproduces_demands_and_initial_state():
    a, b = InventoryEnv(), InventoryEnv()
    obs_a, info_a = a.reset(seed=123)
    obs_b, info_b = b.reset(seed=123)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert info_a == info_b
    for _ in range(20):
        step_a = a.step(5)
        step_b = b.step(5)
        np.testing.assert_array_equal(step_a[0], step_b[0])
        assert step_a[1:] == step_b[1:]


def test_capacity_and_accounting_hold():
    env = InventoryEnv()
    env.reset(seed=5, options={"initial_inventory": 30})
    for _ in range(30):
        obs, reward, _, _, info = env.step(15)
        assert 0 <= obs[0] <= 30
        assert info["actual_order"] <= 15
        assert info["sales"] + info["lost_sales"] == info["demand"]
        assert info["inventory_after_sales"] + info["actual_order"] <= 30
        assert np.isfinite(reward)


def test_randomizer_changes_only_demand_mean():
    env = DemandRandomizer(InventoryEnv(), 3.0, 9.0)
    means = []
    _, info = env.reset(seed=10)
    means.append(info["demand_mean"])
    assert info["demand_mean"] == env.unwrapped.demand_mean
    for _ in range(7):
        _, info = env.reset()
        means.append(info["demand_mean"])
    assert len(set(means)) > 1
    _, repeated = env.reset(seed=10)
    assert repeated["demand_mean"] == means[0]
    assert all(3.0 <= mean <= 9.0 for mean in means)



def test_gymnasium_api_contract():
    from gymnasium.utils.env_checker import check_env

    check_env(InventoryEnv(), skip_render_check=True)
