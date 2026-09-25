import itertools
import random
import time

import pytest

from danish_wist.bots import RandomBot, RuleBot
from learn.arena import play, random_positions
from learn.runner import play_many


class FirstChoice:
    def choose(self, view):
        return view.legal_actions[0]


class Recorder:
    """A batch agent that plays like RuleBot and notes every batch it is given."""

    def __init__(self):
        self.batches = []

    def choose_batch(self, views):
        self.batches.append(views)
        return [RuleBot().choose(view) for view in views]


class Broken:
    def choose(self, view):
        raise RuntimeError("broken agent")


def random_agents(worker):
    return [RandomBot(random.Random(worker))] * 4


def key(deal):
    return (deal.dealer, tuple(deal.initial_hands), tuple(deal.cat))


def by_position(deals):
    return {key(deal): deal.history for deal in deals}


def test_plays_every_position_like_one_at_a_time():
    positions = random_positions(40, random.Random(1))
    agents = [RuleBot(), FirstChoice(), RuleBot(), FirstChoice()]
    deals = list(play_many(positions, agents, games_in_flight=7, workers=1))
    assert len(deals) == 40 and all(deal.is_over for deal in deals)
    assert by_position(deals) == by_position(play(p, agents) for p in positions)


def test_decisions_are_batched_by_agent():
    recorder = Recorder()
    positions = random_positions(30, random.Random(2))
    deals = list(play_many(positions, [recorder] * 4, games_in_flight=10, workers=1))
    assert max(len(batch) for batch in recorder.batches) == 10
    assert sum(map(len, recorder.batches)) == sum(len(deal.history) for deal in deals)
    assert all(view.legal_actions for batch in recorder.batches for view in batch)


def test_endless_positions_are_read_lazily():
    endless = (random_positions(1, random.Random(n))[0] for n in itertools.count())
    deals = list(itertools.islice(play_many(endless, [RuleBot()] * 4, workers=1), 5))
    assert len(deals) == 5


def test_workers_play_the_same_deals():
    positions = random_positions(24, random.Random(3))
    deals = list(play_many(positions, [RuleBot()] * 4, games_in_flight=4, workers=2))
    assert by_position(deals) == by_position(play(p, [RuleBot()] * 4) for p in positions)


def test_workers_make_their_own_agents_and_stop_when_left():
    endless = (random_positions(1, random.Random(n))[0] for n in itertools.count())
    start = time.perf_counter()
    deals = play_many(endless, random_agents, games_in_flight=8, workers=2)
    assert len(list(itertools.islice(deals, 50))) == 50
    deals.close()
    assert time.perf_counter() - start < 30


def test_an_agent_error_reaches_the_caller():
    positions = random_positions(4, random.Random(4))
    with pytest.raises(RuntimeError, match="broken agent"):
        list(play_many(positions, [Broken()] * 4, workers=2))


def scores(deal):
    return deal.scores


def test_finish_runs_where_the_deal_was_played():
    positions = random_positions(12, random.Random(5))
    expected = sorted(play(p, [RuleBot()] * 4).scores for p in positions)
    for workers in (1, 2):
        results = play_many(positions, [RuleBot()] * 4, workers=workers, finish=scores)
        assert sorted(results) == expected
