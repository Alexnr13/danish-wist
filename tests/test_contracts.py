import random

from danish_wist.bots import RuleBot
from learn.arena import play, random_positions
from learn.contracts import KINDS, study


def test_self_play_declares_one_contract_per_deal_that_is_not_redealt():
    positions = random_positions(24, random.Random(1))
    found = study(["rule"], positions, None)["rule"]
    redeals = sum(play(p, [RuleBot()] * 4).redeal for p in positions)
    assert found.seats == 4 * 24 and len(found.declared) == 24 - redeals
    assert all(kind in KINDS and 7 <= level <= 13 for kind, level, _, _ in found.declared)


def test_in_a_field_only_the_candidates_own_contracts_count():
    # RuleBot among RuleBots declares, in the four seats taken together, each
    # deal's contract exactly once.
    positions = random_positions(24, random.Random(2))
    alone = study(["rule"], positions, None)["rule"]
    among = study(["rule"], positions, "rule")["rule"]
    assert among.seats == alone.seats and sorted(among.declared) == sorted(alone.declared)


def test_the_tables_cover_every_contract():
    found = study(["rule"], random_positions(24, random.Random(3)), None)["rule"]
    assert "all" in found.table() and len(found.summary()) == 8
    assert found.summary()[0] == len(found.declared) / found.seats


def test_phase_changes_compare_each_network_with_the_first(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    from learn.contracts import phases
    from learn.model import Net, NetConfig, export

    torch.manual_seed(0)
    path = str(tmp_path / "net.npz")
    export(Net(NetConfig(width=32, layers=1, heads=2)), path)
    report = phases([path, path], 2, random.Random(0))
    assert report.count(" 100% ") >= 3  # the same network agrees with itself everywhere
