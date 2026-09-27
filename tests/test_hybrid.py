import random
from functools import partial

from danish_wist.bots import RandomBot, RuleBot
from learn.arena import duplicate, make_agent, play, random_positions
from learn.evaluate import evaluate_all, named_agents, run, screen
from learn.hybrid import AUCTION, CONTRACT, PLAY, Hybrid, part
from learn.runner import Runner


class Recorder:
    """Plays like RuleBot and keeps the phases it was asked about."""

    def __init__(self):
        self.phases = set()

    def choose_batch(self, views):
        self.phases |= {view.phase for view in views}
        return [RuleBot().choose(view) for view in views]


def test_a_hybrid_of_rulebots_is_rulebot():
    positions = random_positions(20, random.Random(1))
    hybrid = Hybrid(RuleBot(), RuleBot(), RuleBot())
    assert duplicate(hybrid, RuleBot(), positions).per_deal == [0.0] * 20
    assert run("play:rule", "rule", 10).per_deal == [0.0] * 10


def test_each_part_decides_only_its_own_phases():
    parts = Recorder(), Recorder(), Recorder()
    for position in random_positions(30, random.Random(2)):
        play(position, [Hybrid(*parts)] * 4)
    for recorder, name in zip(parts, (AUCTION, CONTRACT, PLAY), strict=True):
        assert recorder.phases and all(part(phase) == name for phase in recorder.phases)


def test_the_card_play_test_keeps_rulebot_s_contract():
    """With `play:`, only card play differs from the all-RuleBot baseline: the stake is fixed."""
    rng = random.Random(3)
    card_player = make_agent("play:random", rng)
    for position in random_positions(30, rng):
        baseline = play(position, [RuleBot()] * 4)
        for seat in range(4):
            agents = [RuleBot()] * 4
            agents[seat] = card_player
            deal = play(position, agents)
            assert (deal.declarer, deal.bid, deal.trumps, deal.alone) == (
                baseline.declarer,
                baseline.bid,
                baseline.trumps,
                baseline.alone,
            )


def test_hybrid_names_make_hybrids():
    rng = random.Random(4)
    hybrid = make_agent("hybrid:random,rule,rule", rng)
    assert isinstance(hybrid.parts[AUCTION], RandomBot)
    assert hybrid.parts[CONTRACT] is hybrid.parts[PLAY]  # one agent, made once
    import pytest

    for bad in ("hybrid:rule,rule", "play:rule,rule"):
        with pytest.raises(ValueError):
            make_agent(bad, rng)


def some_network(tmp_path) -> str:
    """A small random network, exported: a deterministic agent that plays unlike RuleBot."""
    import pytest

    torch = pytest.importorskip("torch")
    from learn.model import Net, NetConfig, export

    torch.manual_seed(6)
    path = str(tmp_path / "net.npz")
    export(Net(NetConfig(width=32, layers=1, heads=2)), path)
    return path


def test_several_candidates_share_the_baseline_and_match_separate_evaluations(tmp_path):
    network = some_network(tmp_path)
    names = [f"play:{network}", network, "rule"]
    positions = random_positions(6, random.Random(5))
    with Runner(partial(named_agents, names=[*names, "rule"], worlds=8, seed=0), workers=1) as r:
        together = evaluate_all(r, names, "rule", positions)
    assert list(together) == names and together["rule"].per_deal == [0.0] * 6
    for name in names:
        assert together[name].per_deal == run(name, "rule", 6, seed=5).per_deal
    assert together[network].per_deal != together[f"play:{network}"].per_deal


def test_screening_pools_the_seeds(tmp_path):
    network = some_network(tmp_path)
    results = screen([network, "rule"], "rule", 4, seeds=[1, 2])
    expected = run(network, "rule", 4, seed=1).per_deal + run(network, "rule", 4, seed=2).per_deal
    assert results[network].per_deal == expected
    assert results["rule"].per_deal == [0.0] * 8
