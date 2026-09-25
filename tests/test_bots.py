import random

from danish_wist import Match
from danish_wist.bots import RandomBot, RuleBot, allies
from danish_wist.game import Deal


def play_match(bots, deals: int, rng: random.Random) -> Match:
    match = Match(rng)
    for _ in range(deals):
        deal = match.new_deal()
        while not deal.is_over:
            deal.apply(bots[deal.to_act].choose(deal.view(deal.to_act)))  # raises if illegal
        match.record(deal)
    return match


def test_rule_bots_only_make_legal_moves():
    play_match([RuleBot()] * 4, 500, random.Random(1))


def test_rule_bots_beat_random_bots():
    rng = random.Random(2)
    match = play_match([RuleBot(), RandomBot(rng), RuleBot(), RandomBot(rng)], 300, rng)
    assert match.scores[0] + match.scores[2] > 0


def test_allies_follow_what_the_view_reveals():
    deal = Deal.new(0, random.Random(4))
    assert allies(deal.view(2)) == {2}  # nothing known yet
