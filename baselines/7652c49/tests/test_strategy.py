import neuropoker.strategy as strategy


def test_select_discard_prefers_max_equity():
    config = strategy.StrategyConfig(discard_randomness=0.0)
    state = strategy.init_state(config)
    equities = [0.2, 0.6, 0.5]
    assert strategy.select_discard(equities, config, state) == 1


def test_pot_odds_zero_cost():
    assert strategy.pot_odds_to_call(0, 10) == 0.0


def test_decide_action_folds_when_behind():
    config = strategy.StrategyConfig(bluff_rate=0.0)
    state = strategy.init_state(config)
    action = strategy.decide_action(
        legal_actions=["fold", "call"],
        pot_odds=0.4,
        equity=0.2,
        config=config,
        state=state,
    )
    assert action == "fold"


def test_hebbian_update_increases_weights():
    weights = {"a": 0.0}
    features = {"a": 1.5}
    updated = strategy.hebbian_update(weights, features, lr=0.2)
    assert updated["a"] > weights["a"]


def test_mimetic_update_decreases_weights():
    weights = {"a": 1.0}
    features = {"a": 0.5}
    updated = strategy.mimetic_update(weights, features, lr=0.2)
    assert updated["a"] < weights["a"]


def test_activation_helpers_return_float():
    features = {"x": 0.2, "y": 0.4}
    config = strategy.StrategyConfig()
    state = strategy.init_state(config)
    assert isinstance(strategy.copula_activation(features), float)
    assert isinstance(strategy.graph_isomorphic_activation(features), float)
    assert isinstance(strategy.continuous_random_draw(features, state), float)
    assert isinstance(strategy.weighted_signature(features), float)
