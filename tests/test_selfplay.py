import random
from dataclasses import asdict
from functools import partial

import pytest

torch = pytest.importorskip("torch")

from danish_wist import Deal  # noqa: E402
from danish_wist.bots import RuleBot  # noqa: E402
from learn.arena import duplicate, random_positions  # noqa: E402
from learn.encoding import Kind, card_id, encode, encode_oracle  # noqa: E402
from learn.model import Net, NetConfig  # noqa: E402
from learn.runner import Runner  # noqa: E402
from learn.selfplay import (  # noqa: E402
    LEARNER,
    RULE,
    TARGET,
    Settings,
    advantages,
    choose_lineups,
    collect,
    evaluate,
    make_agents,
    prepare,
    symexp,
    symlog,
    train,
)

SMALL = NetConfig(width=32, layers=1, heads=2)


def test_oracle_adds_every_hidden_card_after_the_players_own_tokens():
    deal = Deal.new(0, random.Random(1))
    oracle, own = encode_oracle(deal, 2), encode(deal.view(2))
    assert oracle[: len(own)] == own
    hidden = [t for t in oracle[len(own) :] if t[0] == Kind.HAND]
    others = [c for s in (0, 1, 3) for c in deal.hands[s]] + deal.cat
    assert sorted(t[1] for t in hidden) == sorted(card_id(c) for c in others)


def test_advantages_with_lambda_one_are_reward_minus_value():
    assert advantages([1.0, 2.0, 3.0], 10.0, lam=1.0) == [9.0, 8.0, 7.0]


def test_advantages_with_lambda_zero_are_one_step_errors():
    assert advantages([1.0, 2.0, 3.0], 10.0, lam=0.0) == [1.0, 1.0, 7.0]


def test_symlog_round_trip():
    x = torch.tensor([-12345.0, -1.0, 0.0, 3.0, 40000.0])
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-4)


def runner_for(net: Net):
    """A runner in this process with the learner loaded with `net`'s weights."""
    make = partial(make_agents, config=asdict(net.config), snapshots=2, one_thread=False)
    runner = Runner(make, workers=1, games_in_flight=16)
    runner.broadcast(LEARNER, "load", net.state_dict())
    return runner


def test_collect_records_learner_seats_and_their_scores():
    torch.manual_seed(0)
    lineups = [[LEARNER, RULE, LEARNER, RULE]] * 6 + [[LEARNER] * 4] * 6
    with runner_for(Net(SMALL)) as runner:
        found = collect(runner, lineups, random.Random(2))
        assert not runner._agents[LEARNER].steps  # every game's steps were collected
    assert found and all(t.steps for t in found)
    assert len(found) <= 2 * 6 + 4 * 6


def test_the_same_seed_collects_the_same_decisions():
    torch.manual_seed(1)
    net = Net(SMALL)
    runs = []
    for _ in range(2):
        with runner_for(net) as runner:
            found = collect(runner, [[LEARNER] * 4] * 4, random.Random(3))
            runs.append([[s.action for s in t.steps] for t in found])
    assert runs[0] == runs[1]


def test_lineups_are_mostly_self_play():
    lineups = choose_lineups(400, [RULE], share=0.25, rng=random.Random(3))
    mixed = sum(any(a != LEARNER for a in lineup) for lineup in lineups)
    assert 60 < mixed < 140
    assert all(LEARNER in lineup for lineup in lineups)


def test_prepare_normalises_advantages():
    torch.manual_seed(4)
    with runner_for(Net(SMALL)) as runner:
        found = collect(runner, [[LEARNER] * 4] * 8, random.Random(4))
    batch = prepare(found, Net(SMALL), Settings())
    assert abs(batch.advantages.mean().item()) < 1e-5
    assert len(batch.actions) == sum(len(t.steps) for t in found)


def test_runner_evaluation_matches_the_arena():
    positions = random_positions(6, random.Random(5))
    with runner_for(Net(SMALL)) as runner:
        by_runner = evaluate(runner, RULE, RULE, positions)
    assert by_runner.per_deal == duplicate(RuleBot(), RuleBot(), positions).per_deal == [0.0] * 6


def test_training_runs_and_changes_the_policy(tmp_path):
    torch.manual_seed(5)
    policy, critic = Net(SMALL), Net(SMALL)
    before = [p.clone() for p in policy.parameters()]
    settings = Settings(deals_per_iteration=8, snapshot_every=1, batch_size=64)
    history = train(policy, critic, 2, settings, random.Random(5), out=tmp_path)
    assert len(history) == 2 and all(e["policy_loss"] == e["policy_loss"] for e in history)
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters(), strict=True))
    assert (tmp_path / "policy.npz").exists() and (tmp_path / "log.jsonl").exists()


def test_a_run_can_be_resumed_from_its_checkpoints(tmp_path):
    from learn.model import load

    torch.manual_seed(6)
    settings = Settings(deals_per_iteration=8, snapshot_every=1, batch_size=64)
    train(Net(SMALL), Net(SMALL), 1, settings, random.Random(6), out=tmp_path, device="cpu")
    policy, critic = load(str(tmp_path / "policy.pt")), load(str(tmp_path / "critic.pt"))
    history = train(policy, critic, 1, settings, random.Random(7), out=tmp_path, device="cpu")
    assert (
        history[0]["iteration"] == 1 and len((tmp_path / "log.jsonl").read_text().splitlines()) == 2
    )


def test_belief_targets_cover_exactly_the_cards_a_seat_cannot_see():
    from learn.encoding import NOT_HIDDEN, OUT_OF_PLAY, REAL_CARDS, belief_targets

    deal = Deal.new(1, random.Random(8))
    targets = belief_targets(deal, 0)
    hidden = {c for s in (1, 2, 3) for c in deal.hands[s]} | set(deal.cat)
    for card, place in zip(REAL_CARDS[:52], targets, strict=True):
        if card in deal.hands[0]:
            assert place == NOT_HIDDEN
        elif card in deal.cat:
            assert place == OUT_OF_PLAY
        else:
            assert card in hidden and card in deal.hands[place + 1]


def test_belief_loss_is_trained_and_logged():
    torch.manual_seed(9)
    settings = Settings(deals_per_iteration=8, batch_size=64)
    history = train(Net(SMALL), Net(SMALL), 1, settings, random.Random(9))
    assert history[0]["belief_loss"] > 0


def test_exploiter_trains_in_one_seat_against_a_frozen_target():
    from learn.selfplay import exploit_lineups

    lineups = exploit_lineups(20, random.Random(10))
    assert all(lineup.count(LEARNER) == 1 and lineup.count(TARGET) == 3 for lineup in lineups)

    torch.manual_seed(10)
    settings = Settings(deals_per_iteration=8, batch_size=64, games_in_flight=16)
    history = train(
        Net(SMALL),
        Net(SMALL),
        1,
        settings,
        random.Random(10),
        eval_every=1,
        eval_deals=4,
        target=Net(SMALL),
    )
    assert "vs_target" in history[0] and "vs_rulebot" not in history[0]


def test_collecting_in_worker_processes():
    torch.manual_seed(11)
    net = Net(SMALL)
    make = partial(make_agents, config=asdict(net.config), snapshots=1, one_thread=True)
    with Runner(make, workers=2, games_in_flight=8) as runner:
        runner.broadcast(LEARNER, "load", net.state_dict())
        found = collect(runner, [[LEARNER, RULE, LEARNER, RULE]] * 6, random.Random(11))
    assert found and all(t.steps and t.steps[0].oracle.dtype.name == "int16" for t in found)
