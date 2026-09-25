import json
import random

import pytest
from helpers import auction_won_by, deal_with

from danish_wist import Bid, Deal, Pass
from danish_wist.actions import decode, encode
from danish_wist.game import IllegalActionError
from danish_wist.record import replay, to_record


def random_deal(rng: random.Random) -> Deal:
    deal = Deal.new(dealer=rng.randrange(4), rng=rng)
    while not deal.is_over:
        deal.apply(rng.choice(deal.legal_actions()))
    return deal


def test_every_action_round_trips_through_text():
    rng = random.Random(7)
    for _ in range(200):
        deal = Deal.new(0, rng)
        while not deal.is_over:
            for action in deal.legal_actions():
                assert decode(encode(action)) == action
            deal.apply(rng.choice(deal.legal_actions()))


def test_replaying_a_record_reproduces_the_deal_exactly():
    rng = random.Random(11)
    for _ in range(300):
        deal = random_deal(rng)
        record = json.loads(json.dumps(to_record(deal, match="m1", deal_number=3)))
        replayed = replay(record)
        assert replayed.history == deal.history
        assert replayed.tricks == deal.tricks and replayed.scores == deal.scores
        assert to_record(replayed, match="m1", deal_number=3) == record


def test_record_of_an_unfinished_deal_replays_to_the_same_point():
    deal = deal_with({1: "AS"})
    auction_won_by(deal, 1, Bid(8))
    record = to_record(deal)
    assert record["outcome"] is None
    assert replay(record).view(1) == deal.view(1)


def test_tampered_record_is_rejected():
    deal = random_deal(random.Random(3))
    record = to_record(deal)

    wrong_seat = json.loads(json.dumps(record))
    wrong_seat["actions"][0][0] = (wrong_seat["actions"][0][0] + 1) % 4
    with pytest.raises(ValueError):
        replay(wrong_seat)

    wrong_outcome = json.loads(json.dumps(record))
    wrong_outcome["outcome"]["redeal"] = not wrong_outcome["outcome"]["redeal"]
    with pytest.raises(ValueError):
        replay(wrong_outcome)


def test_illegal_recorded_action_is_rejected():
    deal = deal_with({0: "AC", 1: "AS", 2: "AD", 3: "AH"})
    deal.apply(Pass())
    record = to_record(deal)
    record["actions"].append([2, "play AD"])
    with pytest.raises(IllegalActionError):
        replay(record)
