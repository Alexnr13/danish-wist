import random

from helpers import auction_won_by, deal_with

from danish_wist import Bid, CallAce, Card, Deal, DeclareFucdic, NameTrumps, Phase, Suit, TakeCat
from danish_wist.bots import RandomBot, RuleBot
from learn.arena import play, random_positions
from learn.search import SearchAgent, finish
from learn.worlds import sample_worlds


def test_every_world_shows_the_player_exactly_what_they_saw():
    rng = random.Random(1)
    checked = set()
    for game in range(40):
        deal = Deal.new(game % 4, rng)
        while not deal.is_over:
            view = deal.view(deal.to_act)
            if len(view.legal_actions) > 1 and rng.random() < 0.2:
                worlds = sample_worlds(view, 2, rng)
                assert len(worlds) == 2, view.phase
                assert all(world.view(view.seat) == view for world in worlds)
                checked.add(view.phase)
            deal.apply(rng.choice(view.legal_actions))
    assert {Phase.AUCTION, Phase.PLAY, Phase.DISCARD} <= checked


def test_worlds_differ_in_what_the_player_cannot_see():
    deal = Deal.new(0, random.Random(2))
    worlds = sample_worlds(deal.view(1), 5, random.Random(3))
    assert all(world.hands[1] == deal.hands[1] for world in worlds)
    assert len({tuple(map(str, world.hands[2])) for world in worlds}) > 1


def test_voids_and_the_hidden_fucdic_are_respected():
    deal = deal_with({1: "AS KS QS JS 10S AH KH QH AC KC QC JC 10C", 3: "AD 5D"})
    auction_won_by(deal, 1, Bid(8))
    deal.apply(CallAce(Suit.DIAMONDS))
    deal.apply(NameTrumps(Suit.SPADES))
    deal.apply(TakeCat(False))
    deal.apply(DeclareFucdic(Card.parse("10C")))
    rng = random.Random(4)
    while deal.to_act != 2 or deal.phase is not Phase.PLAY or len(deal.tricks) < 2:
        deal.apply(RuleBot().choose(deal.view(deal.to_act)))
    for world in sample_worlds(deal.view(2), 5, rng):
        assert not any(c.suit is Suit.DIAMONDS and c.rank > 0 for c in world.hands[1])
        assert world.fucdic is not None


def test_finish_plays_deals_to_the_end():
    deals = [p.start() for p in random_positions(3, random.Random(5))]
    finish(deals, RandomBot(random.Random(5)))
    assert all(d.is_over for d in deals)


def test_search_agent_plays_legal_moves():
    rng = random.Random(6)
    agent = SearchAgent(RuleBot(), worlds=2, rng=rng)
    for position in random_positions(2, rng):
        assert play(position, [agent, RuleBot(), agent, RuleBot()]).is_over


def _belief_placing(card: Card, place: int) -> list[list[float]]:
    """A belief that is sure where `card` is and has no idea about the rest."""
    from learn.encoding import REAL_CARDS

    belief = [[0.25] * 4 for _ in range(52)]
    belief[REAL_CARDS.index(card)] = [1.0 if p == place else 0.0 for p in range(4)]
    return belief


def test_a_confident_belief_decides_where_a_card_is_dealt():
    deal = Deal.new(0, random.Random(7))
    view = deal.view(1)
    card = next(c for c in deal.hands[3] if not c.is_joker)  # really with seat 3
    belief = _belief_placing(card, 0)  # sure it is with the next player: seat 2
    worlds = sample_worlds(view, 10, random.Random(8), belief=belief)
    assert len(worlds) == 10 and all(card in world.hands[2] for world in worlds)
    assert all(world.view(1) == view for world in worlds)


def test_a_belief_that_rules_everything_out_is_ignored_for_that_card():
    deal = Deal.new(0, random.Random(9))
    card = next(c for c in deal.hands[3] if not c.is_joker)
    belief = _belief_placing(card, 0)
    belief[[i for i, row in enumerate(belief) if row != [0.25] * 4][0]] = [0.0] * 4
    assert len(sample_worlds(deal.view(1), 5, random.Random(10), belief=belief)) == 5


def test_search_can_sample_worlds_from_a_networks_beliefs():
    import pytest

    torch = pytest.importorskip("torch")
    from learn.model import Net, NetAgent, NetConfig

    torch.manual_seed(12)
    net = NetAgent(Net(NetConfig(width=32, layers=1, heads=2)))
    rng = random.Random(12)
    agent = SearchAgent(RuleBot(), worlds=2, rng=rng, belief=net)
    position = random_positions(1, rng)[0]
    assert play(position, [agent, RuleBot(), RuleBot(), RuleBot()]).is_over
