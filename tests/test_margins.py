import random

import pytest

from danish_wist.actions import encode
from danish_wist.bidding import Bid
from danish_wist.bots import RuleBot
from learn.arena import play, random_positions
from learn.margins import probe, table


@pytest.mark.parametrize("field", [None, "rule"])
def test_the_own_branch_is_the_deal_as_it_was_played(field):
    positions = random_positions(12, random.Random(1))
    rows = probe("rule", positions, field)
    assert rows
    for row in rows:
        assert row.scores["own"] == play(positions[row.deal], [RuleBot()] * 4).scores[row.seat]


@pytest.mark.parametrize("nth", [1, 2])
def test_each_branch_changes_only_the_chosen_bid(nth):
    rows = probe("rule", random_positions(24, random.Random(2)), "rule", nth=nth)
    bids = [row for row in rows if isinstance(row.own, Bid)]
    assert bids
    for row in bids:
        assert row.played["own"] == encode(row.own) and row.played["pass"] == "pass"
        for branch, step in (("up1", 1), ("up2", 2)):
            if row.applies[branch]:
                assert row.played[branch] == encode(Bid(row.own.level + step, row.own.attachment))
            else:
                assert row.played[branch] == encode(row.own)


def test_passing_seats_have_nothing_to_compare():
    rows = probe("rule", random_positions(24, random.Random(3)), "rule")
    passes = [row for row in rows if not isinstance(row.own, Bid)]
    assert passes and not any(row.applies[b] for row in passes for b in ("pass", "up1", "up2"))


def test_the_table_has_a_line_per_hand_strength():
    text = table(probe("rule", random_positions(24, random.Random(4)), "rule"))
    for label in ("aces+Jokers", "0", "4+", "all", "made"):
        assert label in text
