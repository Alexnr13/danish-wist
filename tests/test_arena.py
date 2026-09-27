import random

from danish_wist.bots import RandomBot, RuleBot
from learn.arena import duplicate, play, random_positions


def test_a_position_replays_identically():
    position = random_positions(1, random.Random(1))[0]
    first, second = play(position, [RuleBot()] * 4), play(position, [RuleBot()] * 4)
    assert first.history == second.history and first.scores == second.scores


def test_an_agent_has_no_advantage_over_itself():
    result = duplicate(RuleBot(), RuleBot(), random_positions(50, random.Random(2)))
    assert result.per_deal == [0.0] * 50


def test_rule_bot_beats_a_random_field_and_random_loses_to_a_rule_field():
    rng = random.Random(3)
    positions = random_positions(150, rng)
    better = duplicate(RuleBot(), RandomBot(rng), positions)
    worse = duplicate(RandomBot(rng), RuleBot(), positions)
    assert better.mean - better.ci95 > 0
    assert worse.mean + worse.ci95 < 0


def test_every_seat_is_given_a_role():
    result = duplicate(RuleBot(), RuleBot(), random_positions(20, random.Random(4)))
    assert sum(len(v) for v in result.by_role.values()) == 4 * 20


def test_agents_can_be_named_on_the_command_line(tmp_path):
    import pytest

    from learn.arena import make_agent
    from learn.search import SearchAgent

    rng = random.Random(5)
    assert isinstance(make_agent("rule", rng), RuleBot)
    with pytest.raises(ValueError):
        make_agent("nonsense", rng)

    torch = pytest.importorskip("torch")
    from learn.inference import NumpyAgent
    from learn.model import Net, NetConfig, export

    torch.manual_seed(5)
    export(Net(NetConfig(width=32, layers=1, heads=2)), str(tmp_path / "net.npz"))
    assert isinstance(make_agent(str(tmp_path / "net.npz"), rng), NumpyAgent)
    searcher = make_agent(f"search:{tmp_path / 'net.npz'}", rng, worlds=2)
    assert isinstance(searcher, SearchAgent) and isinstance(searcher.belief, NumpyAgent)


def test_workers_default_to_all_cores_but_two_and_fewer_on_a_gpu(monkeypatch):
    import os

    from learn import arena

    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    assert arena.default_workers(None) == arena.default_workers("cpu") == 22
    assert arena.default_workers("cuda") == arena.GPU_WORKERS < 22
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert arena.default_workers("cuda") == 2
