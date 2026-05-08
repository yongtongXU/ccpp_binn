import pytest

from src.core.strategy import GBNNGreedyStrategy, RollingGBNNStrategy, create_strategy
from src.utils.config import load_config


def test_default_strategy_is_rolling_gbnn():
    cfg = load_config()
    strategy = create_strategy(cfg)
    assert isinstance(strategy, RollingGBNNStrategy)
    assert strategy.name == "rolling_gbnn"
    assert cfg["rolling_optimizer"]["use_global_strip_plan"] is False
    assert cfg["rolling_optimizer"]["use_priority_strip"] is False
    assert cfg["rolling_optimizer"]["use_strip_constraints"] is False


def test_greedy_strategy_entrypoint():
    cfg = load_config()
    cfg["method"]["name"] = "gbnn_greedy"
    strategy = create_strategy(cfg)
    assert isinstance(strategy, GBNNGreedyStrategy)
    assert strategy.name == "gbnn_greedy"


def test_unknown_strategy_reports_available_methods():
    cfg = load_config()
    cfg["method"]["name"] = "does_not_exist"
    with pytest.raises(ValueError, match="rolling_gbnn"):
        create_strategy(cfg)
