import random

import pytest

torch = pytest.importorskip("torch")

from helpers import auction_won_by, deal_with  # noqa: E402

from danish_wist import Attachment, Bid, CallAce, Pass, Phase, Suit  # noqa: E402
from danish_wist.bots import RuleBot  # noqa: E402
from learn.encoding import ACTIONS, action_index  # noqa: E402
from learn.imitate import alternatives, loss_against, target_odds, teacher_samples  # noqa: E402
from learn.model import collate  # noqa: E402


def test_a_bid_keeps_the_other_kinds_of_contract_at_its_level_open():
    deal = deal_with({0: "AS KS QS JS AH"})
    view = deal.view(deal.to_act)
    while view.phase is not Phase.AUCTION:
        deal.apply(view.legal_actions[1])
        view = deal.view(deal.to_act)
    others = {ACTIONS[i] for i in alternatives(view, Bid(9))}
    assert others == {Bid(9, a) for a in Attachment}
    assert alternatives(view, Pass()) == ()


def test_contract_decisions_keep_every_other_choice_open_and_card_play_none():
    deal = deal_with({1: "KH QH AS KS QS JS", 3: "AH 5H"})
    auction_won_by(deal, 1, Bid(8))
    view = deal.view(1)
    assert {ACTIONS[i] for i in alternatives(view, CallAce(Suit.HEARTS))} == {
        a for a in view.legal_actions if a != CallAce(Suit.HEARTS)
    }
    while deal.phase is not Phase.PLAY:
        deal.apply(deal.legal_actions()[0])
    play = deal.view(deal.to_act)
    assert alternatives(play, play.legal_actions[0]) == ()


def test_exploration_spreads_its_share_over_the_alternatives():
    samples = teacher_samples(RuleBot(), 3, random.Random(1))
    odds = target_odds(samples, 0.1)
    assert torch.allclose(odds.sum(-1), torch.ones(len(samples)))
    for sample, row in zip(samples, odds, strict=True):
        expected = 0.9 if sample.alternatives else 1.0
        assert row[sample.target].item() == pytest.approx(expected)
        assert all(row[i] > 0 for i in sample.alternatives)


def test_without_exploration_the_loss_is_plain_cross_entropy():
    torch.manual_seed(2)
    samples = teacher_samples(RuleBot(), 2, random.Random(2))
    tokens, padding, legal = collate([s.observation for s in samples])
    logits = torch.randn(len(samples), len(ACTIONS)).masked_fill(~legal, float("-inf"))
    targets = torch.tensor([s.target for s in samples])
    expected = torch.nn.functional.cross_entropy(logits, targets)
    assert loss_against(logits, legal, target_odds(samples, 0.0)).item() == pytest.approx(
        expected.item(), rel=1e-5
    )
    assert action_index(ACTIONS[samples[0].target]) == samples[0].target
