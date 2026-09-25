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
