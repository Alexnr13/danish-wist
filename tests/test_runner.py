import itertools
import os
import pickle
import random
import time

import pytest

from danish_wist.bots import RandomBot, RuleBot
from learn.arena import duplicate, play, random_positions
from learn.runner import Runner, chunk_size, play_many


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


# --- Runner: a pool that stays up, lineups per game, decisions in context ---


class Learner:
    """Plays at random from its own seed and records each decision by (game, seat)."""

    def __init__(self, worker):
        self.worker, self.rng, self.steps = worker, random.Random(worker), {}

    def seed(self, n):
        self.rng = random.Random(f"{n}/{self.worker}")

    def choose_decisions(self, decisions):
        actions = []
        for decision in decisions:
            assert decision.view == decision.deal.view(decision.seat)
            assert decision.deal.to_act == decision.seat
            action = self.rng.choice(decision.view.legal_actions)
            self.steps.setdefault((decision.game, decision.seat), []).append(action)
            actions.append(action)
        return actions

    def unfinished(self):
        return len(self.steps)

    def number_and_process(self):
        return self.worker, os.getpid()


def make_agents(worker):
    return {"learner": Learner(worker), "rule": RuleBot(), "first": FirstChoice()}


def broken_agents(worker):
    raise ValueError(f"no agents in worker {worker}")


def agents_with_a_broken_one(worker):
    return make_agents(worker) | {"broken": Broken()}


def learner_steps(game, deal, agents):
    steps = agents["learner"].steps
    mine = {seat: steps.pop((game, seat)) for seat in range(4) if (game, seat) in steps}
    return game, mine, deal.history, deal.scores


def scores_of(game, deal, agents):
    return game, deal.scores


LINEUPS = [
    ["learner"] * 4,
    ["learner", "rule", "learner", "first"],
    ["rule", "rule", "learner", "rule"],
    ["first", "rule", "first", "rule"],
]


def learner_games(count, seed):
    positions = random_positions(count, random.Random(seed))
    return [(position, LINEUPS[i % len(LINEUPS)]) for i, position in enumerate(positions)]


@pytest.mark.parametrize("workers", [1, 2])
def test_learner_records_its_own_decisions_by_game_and_seat(workers):
    games = learner_games(24, seed=6)
    with Runner(make_agents, workers=workers, games_in_flight=5) as runner:
        results = sorted(runner.play(games, finish=learner_steps))
        assert sum(runner.broadcast("learner", "unfinished")) == 0
    assert [game for game, *_ in results] == list(range(24))
    for game, mine, history, _ in results:
        lineup = games[game][1]
        assert set(mine) == {seat for seat in range(4) if lineup[seat] == "learner"}
        for seat, actions in mine.items():
            assert actions == [action for s, action in history if s == seat]


def test_the_same_seed_gives_the_same_games():
    games = learner_games(30, seed=7)
    with Runner(make_agents, workers=2, games_in_flight=4) as runner:

        def run(seed):
            runner.broadcast("learner", "seed", seed)
            results = runner.play(games, learner_steps)
            return sorted((game, history) for game, _, history, _ in results)

        first = run(1)
        assert run(1) == first
        assert run(2) != first


def test_workers_stay_up_and_broadcast_reaches_each_one():
    with Runner(make_agents, workers=2) as runner:
        before = runner.broadcast("learner", "number_and_process")
        assert [number for number, _ in before] == [0, 1]
        for _ in range(3):
            list(runner.play(learner_games(8, seed=8), finish=scores_of))
        assert runner.broadcast("learner", "number_and_process") == before


def test_duplicate_evaluation_through_the_runner_matches_the_arena():
    positions = random_positions(12, random.Random(9))
    field = ["first"] * 4
    seatings = [field] + [field[:seat] + ["rule"] + field[seat + 1 :] for seat in range(4)]
    games = [(position, seating) for position in positions for seating in seatings]
    with Runner(make_agents, workers=2) as runner:
        scores = dict(runner.play(games, finish=scores_of))
    per_deal = []
    for p in range(len(positions)):
        baseline = scores[p * 5]
        per_deal.append(sum(scores[p * 5 + 1 + s][s] - baseline[s] for s in range(4)) / 4)
    assert per_deal == duplicate(RuleBot(), FirstChoice(), positions).per_deal


def test_a_runner_recovers_from_a_failed_or_abandoned_play():
    games = learner_games(16, seed=10)
    endless = ((random_positions(1, random.Random(n))[0], ["rule"] * 4) for n in itertools.count())
    with Runner(make_agents, workers=2, games_in_flight=4) as runner:
        with pytest.raises(KeyError, match="nobody"):
            list(runner.play([(games[0][0], ["learner", "rule", "rule", "nobody"])], scores_of))
    with Runner(agents_with_a_broken_one, workers=2, games_in_flight=4) as runner:
        with pytest.raises(RuntimeError, match="broken agent"):
            list(runner.play([(p, ["broken"] * 4) for p, _ in games], scores_of))
        stopped = runner.play(endless, scores_of)
        assert len(list(itertools.islice(stopped, 10))) == 10
        stopped.close()
        with pytest.raises((pickle.PicklingError, AttributeError)):  # finish must pickle
            list(runner.play(games, finish=lambda game, deal, agents: game))
        assert sorted(game for game, _ in runner.play(games, scores_of)) == list(range(16))


def test_errors_making_agents_surface_when_the_runner_starts():
    with pytest.raises(ValueError, match="no agents in worker"):
        Runner(broken_agents, workers=2)


def test_one_play_at_a_time():
    for workers in (1, 2):
        with Runner(make_agents, workers=workers, games_in_flight=2) as runner:
            runner.play(learner_games(4, seed=11), scores_of)  # never started: no harm
            going = runner.play(learner_games(4, seed=11), scores_of)
            next(going)
            with pytest.raises(RuntimeError, match="still going"):
                runner.broadcast("learner", "seed", 1)
            with pytest.raises(RuntimeError, match="still going"):
                next(runner.play(learner_games(4, seed=11), scores_of))
            assert len(list(going)) == 3
            runner.broadcast("learner", "seed", 1)
    with pytest.raises(RuntimeError, match="closed"):
        runner.play(learner_games(1, seed=12))


def test_chunks_share_a_list_of_games_evenly_between_workers():
    for count in (1, 7, 100, 256, 1000, 1024, 5000):
        for workers in (2, 3, 6, 8, 10):
            for games_in_flight in (8, 32, 256):
                size = chunk_size(count, workers, games_in_flight)
                chunks = [min(size, count - start) for start in range(0, count, size)]
                loads = [sum(chunks[w::workers]) for w in range(workers)]
                assert size <= 4 * games_in_flight
                assert max(loads) <= count / workers + len(chunks[::workers])
