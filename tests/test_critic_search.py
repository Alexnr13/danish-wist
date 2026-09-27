import random

import pytest

torch = pytest.importorskip("torch")

from danish_wist.bots import RuleBot  # noqa: E402
from danish_wist.game import Phase  # noqa: E402
from learn.arena import play, random_positions  # noqa: E402
from learn.critic_search import CriticSearch, critic_beside  # noqa: E402
from learn.model import Net, NetConfig  # noqa: E402
from learn.selfplay import as_critic  # noqa: E402

SMALL = NetConfig(width=32, layers=1, heads=2)


def searcher(**options) -> CriticSearch:
    torch.manual_seed(1)
    policy = Net(SMALL)
    return CriticSearch(policy, as_critic(policy), worlds=6, rng=random.Random(1), **options)


def test_the_critic_is_found_beside_its_policy():
    assert critic_beside("runs/x/checkpoints/policy-0100.pt") == "runs/x/checkpoints/critic-0100.pt"
    assert critic_beside("runs/x/policy.npz") == "runs/x/critic.pt"


def test_it_plays_the_card_whose_worlds_are_worth_most(monkeypatch):
    """With a stand-in critic that values each deal by the rank of the seat's last card."""
    search = searcher()

    def by_rank(deals, seats):
        led = [[c for s, c in d.trick if s == seat] for d, seat in zip(deals, seats, strict=True)]
        return torch.tensor([float(cards[-1].rank) if cards else 0.0 for cards in led])

    monkeypatch.setattr(search, "_values", by_rank)
    checked = 0
    for position in random_positions(6, random.Random(2)):
        deal = position.start()
        while not deal.is_over:
            view = deal.view(deal.to_act)
            if view.phase is Phase.PLAY and len(view.legal_actions) > 1 and not deal.trick:
                chosen = search.choose(view)  # leading: the card stays the trick's last
                assert chosen.card.rank == max(a.card.rank for a in view.legal_actions)
                checked += 1
            deal.apply(RuleBot().choose(view))
    assert checked > 5


def test_replies_bring_each_deal_back_to_the_seat_or_to_its_end():
    search = searcher(reply=True)
    for position in random_positions(4, random.Random(3)):
        deal = position.start()
        while not deal.is_over:
            view = deal.view(deal.to_act)
            if view.phase is Phase.PLAY and len(view.legal_actions) > 1:
                tried = search._tried(view, [deal])
                search._reply(tried, [view.seat] * len(tried))
                assert all(d.is_over or d.to_act == view.seat for d in tried)
            deal.apply(RuleBot().choose(view))


def test_a_searching_card_player_finishes_deals():
    search = searcher(reply=True)
    for position in random_positions(2, random.Random(4)):
        assert play(position, [search, RuleBot(), RuleBot(), RuleBot()]).is_over
