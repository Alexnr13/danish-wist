import math

import pytest

torch = pytest.importorskip("torch")

from learn.beliefs import GROUPS, measure, prior_log_odds, table  # noqa: E402
from learn.encoding import NOT_HIDDEN, OUT_OF_PLAY  # noqa: E402
from learn.model import Net, NetConfig, save  # noqa: E402


def test_the_prior_weighs_each_place_by_the_unseen_cards_it_holds():
    # At the auction: 13 unseen cards in each other hand, 3 in the cat, the rest seen.
    places = [0] * 13 + [1] * 13 + [2] * 13 + [OUT_OF_PLAY] * 3 + [NOT_HIDDEN] * 10
    targets = torch.tensor([places])
    odds = prior_log_odds(targets)[0].exp()
    assert odds.tolist() == pytest.approx([13 / 42, 13 / 42, 13 / 42, 3 / 42])
    per_card = -prior_log_odds(targets)[0][targets[0][targets[0] != NOT_HIDDEN]].mean()
    assert per_card.item() == pytest.approx(1.2766, abs=1e-3)  # REVIEW.md: "about 1.28"


def test_a_fresh_belief_head_guesses_evenly(tmp_path):
    torch.manual_seed(1)
    net = Net(NetConfig(width=32, layers=1, heads=2))
    torch.nn.init.zeros_(net.belief.weight)
    torch.nn.init.zeros_(net.belief.bias)
    save(net, str(tmp_path / "net.pt"))
    totals = measure(str(tmp_path / "net.pt"), 4, field="rule")
    assert set(totals) <= set(GROUPS) and "auction" in totals
    for decisions, cards, belief, prior in totals.values():
        assert decisions > 0 and belief / cards == pytest.approx(math.log(4), abs=1e-4)
        assert prior / cards < math.log(4)
    assert "all" in table(totals)
