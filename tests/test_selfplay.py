import random

import pytest

torch = pytest.importorskip("torch")

from danish_wist import Deal  # noqa: E402
from danish_wist.bots import RuleBot  # noqa: E402
from learn.encoding import Kind, card_id, encode, encode_oracle  # noqa: E402
from learn.model import Net, NetConfig  # noqa: E402
from learn.selfplay import (  # noqa: E402
    LEARNER,
    Settings,
    advantages,
    choose_lineups,
    collect,
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


def test_collect_records_learner_seats_and_their_scores():
    torch.manual_seed(0)
    rng = random.Random(2)
    lineups = [[LEARNER, RuleBot(), LEARNER, RuleBot()]] * 6 + [[LEARNER] * 4] * 6
    trajectories = collect(Net(SMALL), lineups, rng)
    assert trajectories and all(t.steps for t in trajectories)
    assert len(trajectories) <= 2 * 6 + 4 * 6
    assert all(t.reward == float(t.reward) for t in trajectories)


def test_lineups_are_mostly_self_play():
    lineups = choose_lineups(400, [RuleBot()], share=0.25, rng=random.Random(3))
    mixed = sum(any(a is not LEARNER for a in lineup) for lineup in lineups)
    assert 60 < mixed < 140
    assert all(any(a is LEARNER for a in lineup) for lineup in lineups)


def test_prepare_normalises_advantages():
    torch.manual_seed(4)
    trajectories = collect(Net(SMALL), [[LEARNER] * 4] * 8, random.Random(4))
    batch = prepare(trajectories, Net(SMALL), Settings())
    assert abs(batch.advantages.mean().item()) < 1e-5
    assert len(batch.actions) == sum(len(t.steps) for t in trajectories)


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
