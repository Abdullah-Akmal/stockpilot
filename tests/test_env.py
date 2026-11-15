import numpy as np
import pytest

from stockpilot.env import InventorySim, StockPilotEnv

ENV_CFG = {"max_lead_time": 14, "history_len": 14, "n_actions": 21, "action_max_mult": 3.0,
           "episode_len": 60, "block_len": 7}


def params(lead_time=3, shelf_life=None):
    return {"price": 4.0, "unit_cost": 2.0, "margin": 2.0, "holding_cost": 0.1, "order_cost": 5.0,
            "stockout_penalty": 3.0, "goodwill": 1.0, "lead_time": lead_time, "shelf_life": shelf_life,
            "mean_demand": 10.0, "std_demand": 5.0}


def make_env(lead_time=3, shelf_life=None, seed=0):
    rng = np.random.default_rng(seed)
    demand = rng.poisson(10, 200)
    dows = np.arange(200) % 7
    return StockPilotEnv(params(lead_time, shelf_life), ENV_CFG, demand, dows, seed=seed)


@pytest.mark.parametrize("lead_time", [0, 1, 3, 7])
@pytest.mark.parametrize("shelf_life", [None, 4])
def test_stock_never_negative(lead_time, shelf_life):
    for seed in range(5):
        env = make_env(lead_time, shelf_life, seed)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        done = False
        while not done:
            _, _, done, _, info = env.step(int(rng.integers(0, env.action_space.n)))
            s = info["step"]
            assert env.sim.on_hand >= 0 and s.on_hand_end >= 0
            assert all(q >= 0 for q, _ in env.sim.lots)
            assert all(x >= 0 for x in env.sim.pipeline)
            assert s.sold + s.lost == s.demand and s.sold >= 0 and s.lost >= 0


@pytest.mark.parametrize("lead_time", [0, 1, 2, 3, 7, 14])
def test_order_arrives_after_exactly_lead_time(lead_time):
    sim = InventorySim(lead_time, 0.1, 5.0, 3.0, 2.0, 4.0)
    sim.reset(on_hand=0)
    sim_days = []
    for day in range(lead_time + 3):
        sim_days.append(sim.on_hand)          # stock available at the start of `day`
        r = sim.step(50 if day == 0 else 0, 0)
        if day == lead_time:
            assert r.on_hand_end == 50
    if lead_time == 0:
        assert sim_days[0] == 0               # placed on day 0, sellable on day 0
        assert sim.on_hand == 50
    else:
        assert sim_days[:lead_time] == [0] * lead_time   # not there before day L
        assert sim_days[lead_time] == 50                  # there at the start of day L
        assert sim.on_hand == 50
        # sellable exactly on day L: demand on day L can be served
        sim2 = InventorySim(lead_time, 0.1, 5.0, 3.0, 2.0, 4.0)
        sim2.reset(on_hand=0)
        sold = [sim2.step(50 if d == 0 else 0, 1).sold for d in range(lead_time + 1)]
        assert sold == [0] * lead_time + [1]


def test_lead_time_zero_serves_same_day():
    sim = InventorySim(0, 0.1, 5.0, 3.0, 2.0, 4.0)
    sim.reset(on_hand=0)
    r = sim.step(10, 4)
    assert r.sold == 4 and r.lost == 0 and r.on_hand_end == 6


def test_cost_arithmetic_hand_computed():
    # L=2, h=0.1/unit-day, K=5/order, penalty=3/lost unit, unit cost 2, price 4 (margin 2, goodwill 1)
    sim = InventorySim(2, 0.1, 5.0, 3.0, 2.0, 4.0)
    sim.reset(on_hand=10)
    r0 = sim.step(8, 6)     # sell 6, end 4
    assert (r0.sold, r0.lost, r0.on_hand_end) == (6, 0, 4)
    assert r0.holding_cost == pytest.approx(0.4) and r0.order_cost == 5.0 and r0.lost_sales_cost == 0
    assert r0.profit == pytest.approx(2 * 6 - 0.4 - 5.0)
    r1 = sim.step(0, 7)     # 4 on hand, lose 3, order of 8 not here yet
    assert (r1.sold, r1.lost, r1.on_hand_end) == (4, 3, 0)
    assert r1.lost_sales_cost == pytest.approx(9.0) and r1.order_cost == 0
    assert r1.profit == pytest.approx(2 * 4 - 1 * 3)
    r2 = sim.step(0, 5)     # day 2: order from day 0 has arrived (8 units)
    assert (r2.sold, r2.lost, r2.on_hand_end) == (5, 0, 3)
    assert r2.holding_cost == pytest.approx(0.3)
    total = sum(r.total_cost for r in (r0, r1, r2))
    assert total == pytest.approx(0.4 + 5.0 + 9.0 + 0.3)
    # profit identity: margin * demand - total cost
    assert sum(r.profit for r in (r0, r1, r2)) == pytest.approx(2 * (6 + 7 + 5) - total)


def test_fifo_expiry_and_waste_cost():
    sim = InventorySim(0, 0.0, 0.0, 3.0, 2.0, 4.0, shelf_life=2)
    sim.reset(on_hand=0)
    r0 = sim.step(5, 0)
    assert r0.wasted == 0 and r0.on_hand_end == 5
    r1 = sim.step(3, 2)     # sells 2 from the OLD lot (FIFO); its remaining 3 reach age 2 -> expire
    assert r1.sold == 2 and r1.wasted == 3 and r1.on_hand_end == 3
    assert r1.waste_cost == pytest.approx(3 * 2.0)


@pytest.mark.parametrize("lead_time", [0, 3, 14])
def test_observation_shape_constant_across_lead_times(lead_time):
    env = make_env(lead_time)
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape == (2 + 14 + 1 + 14 + 7,)
    assert env.observation_space.contains(obs)


def test_action_range_scaled_to_lead_time_demand():
    env = make_env(3)
    assert env.action_to_qty(0) == 0
    assert env.action_to_qty(env.action_space.n - 1) == round(3.0 * 10.0 * 3)


def test_reward_roughly_unit_scale():
    env = make_env(3)
    env.reset(seed=0)
    rewards, done = [], False
    rng = np.random.default_rng(0)
    while not done:
        _, r, done, _, _ = env.step(int(rng.integers(0, 8)))
        rewards.append(r)
    assert np.percentile(np.abs(rewards), 90) < 3.0


def test_replay_uses_exact_demand_and_scale():
    p = params(2)
    demand = np.array([5, 0, 12, 7, 3])
    env = StockPilotEnv(p, ENV_CFG, replay={"demand": demand, "dows": np.arange(5), "history": [10] * 14,
                                            "on_hand": 100}, demand_scale=1.3)
    env.reset()
    seen, done = [], False
    while not done:
        _, _, done, _, info = env.step(0)
        seen.append(info["step"].demand)
    assert seen == list(np.round(demand * 1.3).astype(int))
