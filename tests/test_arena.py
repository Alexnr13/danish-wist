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
