import copy
import json
import random
from dataclasses import asdict
from functools import partial

import pytest

torch = pytest.importorskip("torch")

from danish_wist import Deal  # noqa: E402
from danish_wist.bots import RuleBot  # noqa: E402
from learn import selfplay  # noqa: E402
from learn.arena import duplicate, random_positions  # noqa: E402
from learn.encoding import Kind, card_id, encode, encode_oracle  # noqa: E402
from learn.evaluate import evaluate  # noqa: E402
from learn.model import Net, NetConfig, load  # noqa: E402
from learn.runner import Runner  # noqa: E402
from learn.selfplay import (  # noqa: E402
    LEARNER,
    RULE,
    TARGET,
    Settings,
    advantages,
    as_critic,
    choose_lineups,
    collect,
    expected_value,
    make_agents,
    prepare,
    symexp,
    symlog,
    train,
    two_hot,
    update,
    value_bins,
)

SMALL = NetConfig(width=32, layers=1, heads=2)
TINY = Settings(deals_per_iteration=8, snapshot_every=1, batch_size=64)


def critic() -> Net:
    return as_critic(Net(SMALL))


def test_oracle_adds_every_hidden_card_after_the_players_own_tokens():
    deal = Deal.new(0, random.Random(1))
    oracle, own = encode_oracle(deal, 2), encode(deal.view(2))
    assert oracle[: len(own)] == own
    assert encode_oracle(deal, 2, own) == oracle and len(own) < len(oracle)  # own is not changed
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


def test_two_hot_targets_keep_the_value():
    bins = value_bins()
    values = torch.tensor([-33280.0, -415.5, -1.0, 0.0, 0.3, 60.0, 1280.0, 99999.0])
    targets = two_hot(values, bins)
    assert torch.allclose(targets.sum(-1), torch.ones(len(values)))
    assert torch.allclose((targets * bins).sum(-1), values, rtol=1e-5, atol=1e-4)


def test_the_best_critic_predicts_the_mean_score_even_when_its_sign_is_uncertain():
    # A contract made for +200 or failed for -80, at even odds, is worth +60 on
    # average. A regression in symlog units would predict about +0.6.
    bins = value_bins()
    odds = two_hot(torch.tensor([200.0, -80.0]), bins).mean(0)  # what cross-entropy learns
    assert expected_value(odds.log(), bins).item() == pytest.approx(60.0, abs=1e-3)


def test_a_fresh_critic_predicts_zero():
    values = selfplay._critic_values(critic(), [encode_oracle(Deal.new(0, random.Random(2)), 1)])
    assert values.abs().max().item() < 0.1  # points


def test_a_critic_keeps_the_trunk_it_is_made_from():
    net = Net(SMALL)
    made = as_critic(net)
    assert torch.equal(made.embed[1].weight, net.embed[1].weight)
    assert made.value.out_features == selfplay.VALUE_BINS and net.value.out_features == 1


def runner_for(net: Net):
    """A runner in this process with the learner loaded with `net`'s weights."""
    make = partial(make_agents, config=asdict(net.config), snapshots=2, one_thread=False)
    runner = Runner(make, workers=1, games_in_flight=16)
    runner.broadcast(LEARNER, "load", net.state_dict())
    return runner


def some_trajectories(seed: int, deals: int = 8):
    torch.manual_seed(seed)
    with runner_for(Net(SMALL)) as runner:
        return collect(runner, [[LEARNER] * 4] * deals, random.Random(seed))


def test_collect_records_learner_seats_and_their_scores():
    torch.manual_seed(0)
    lineups = [[LEARNER, RULE, LEARNER, RULE]] * 6 + [[LEARNER] * 4] * 6
    with runner_for(Net(SMALL)) as runner:
        found = collect(runner, lineups, random.Random(2))
        assert not runner._agents[LEARNER].steps  # every game's steps were collected
    assert found and all(t.steps for t in found)
    assert len(found) <= 2 * 6 + 4 * 6


def test_the_declarer_s_trajectory_carries_the_contract():
    found = some_trajectories(19, deals=16)
    contracts = [t.contract for t in found if t.contract is not None]
    assert 0 < len(contracts) <= 16
    stats = selfplay.contract_stats(found)
    assert sum(stats["declared"].values()) == pytest.approx(1.0, abs=1e-3)
    assert 7 <= stats["level"] <= 13 and 0 <= stats["made"] <= 1


def test_exploring_moves_a_share_of_each_bid_to_the_other_kinds_at_its_level():
    from danish_wist.bidding import Bid
    from learn.encoding import ACTIONS

    torch.manual_seed(20)
    legal = torch.rand(6, len(ACTIONS)) < 0.5
    probs = torch.softmax(torch.randn(6, len(ACTIONS)).masked_fill(~legal, float("-inf")), -1)
    mixed = selfplay.explored(probs, legal, 0.2)
    bid = torch.tensor([isinstance(a, Bid) for a in ACTIONS])
    assert torch.allclose(mixed.sum(-1), torch.ones(6))
    assert torch.equal(mixed[:, ~bid], probs[:, ~bid]) and torch.all(mixed[~legal] == 0)
    assert torch.all(mixed[:, bid] >= 0.8 * probs[:, bid] - 1e-7)
    assert torch.equal(selfplay.explored(probs, legal, 0.0), probs)


def test_exploring_levels_moves_a_share_of_each_bid_to_the_same_kind_one_and_two_levels_up():
    from danish_wist.bidding import Attachment, Bid
    from learn.encoding import ACTIONS

    halves = [ACTIONS.index(Bid(level, Attachment.HALVES)) for level in (8, 9, 10, 12, 13)]
    probs = torch.zeros(3, len(ACTIONS))
    probs[0, halves[0]] = probs[1, halves[3]] = probs[2, halves[4]] = 1.0  # 8, 12 and 13 Halves
    legal = torch.ones(3, len(ACTIONS), dtype=torch.bool)
    mixed = selfplay.explored(probs, legal, 0.0, levels=0.2)
    assert mixed[0, halves[:3]].tolist() == pytest.approx([0.8, 0.1, 0.1])  # 8 -> 9 and 10
    assert mixed[1, halves[3:]].tolist() == pytest.approx([0.8, 0.2])  # 12 -> only 13 is left
    assert mixed[2, halves[4]] == 1.0  # nothing above 13
    assert torch.allclose(mixed.sum(-1), torch.ones(3))
    both = selfplay.explored(probs[:1], legal[:1], 0.3, levels=0.2)
    same_level = [ACTIONS.index(Bid(8, a)) for a in (None, Attachment.FLIP, Attachment.CLUBS)]
    assert both[0, halves[0]] == pytest.approx(0.5)
    assert both[0, same_level].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert both[0, halves[1:3]].tolist() == pytest.approx([0.1, 0.1])


def test_an_exploring_learner_records_the_chance_it_really_played_with():
    from learn.encoding import Observation

    torch.manual_seed(21)
    net = Net(SMALL)
    make = partial(make_agents, config=asdict(net.config), snapshots=1, one_thread=False)
    ratios, own = {}, {}
    for explore, levels in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5)):
        agents = partial(make, explore=explore, explore_levels=levels)
        with Runner(agents, workers=1, games_in_flight=16) as runner:
            runner.broadcast(LEARNER, "load", net.state_dict())
            steps = [
                s for t in collect(runner, [[LEARNER] * 4] * 8, random.Random(21)) for s in t.steps
            ]
        from learn.model import collate

        obs = [Observation(s.observation.tokens, s.observation.legal) for s in steps]
        with torch.no_grad():
            log_probs = torch.log_softmax(net.eval()(*collate(obs))[0], -1)
        now = log_probs.gather(1, torch.tensor([[s.action] for s in steps])).squeeze(-1)
        ratios[explore, levels] = (now - torch.tensor([s.log_prob for s in steps])).exp()
        own[explore, levels] = (now - torch.tensor([s.policy_log_prob for s in steps])).exp()
    plain = ratios[0.0, 0.0]
    assert torch.allclose(plain, torch.ones_like(plain), atol=1e-4)
    for explored_odds in (ratios[0.5, 0.0], ratios[0.0, 0.5]):
        assert (explored_odds - 1).abs().max() > 0.01  # some bids came from the explored odds
    for ratio in own.values():  # and the policy's own chance is recorded too
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4)


def test_explored_bids_learn_from_good_and_bad_results_alike():
    """PPO's first step on explored play is the policy gradient, weighted by own / played odds.

    The policy's favourite bid, played with 3/4 of its chance, and a bid it
    rarely makes, played with ten times its chance: each should be pushed up by
    a good result and down by a bad one, however far its own chance is from the
    one it was played with (decoupled PPO: clip against the policy, weigh by the odds).
    """
    own = torch.tensor([0.6, 0.02, 0.6, 0.02]).log()
    played = torch.tensor([0.45, 0.2, 0.45, 0.2]).log()
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    new = own.clone().requires_grad_()
    selfplay.ppo_objective(new, own, played, adv, 0.2).sum().backward()
    expected = (own - played).exp() * adv  # d(objective)/d(log chance) at the start
    assert torch.allclose(new.grad, expected)


def test_without_exploring_the_objective_is_ppo_s_clipped_one():
    old = torch.tensor([0.5, 0.5, 0.5, 0.5]).log()
    new = torch.tensor([0.7, 0.3, 0.7, 0.3]).log().requires_grad_()  # ratios 1.4, 0.6, 1.4, 0.6
    adv = torch.tensor([1.0, -1.0, -1.0, 1.0])
    objective = selfplay.ppo_objective(new, old, old, adv, 0.2)
    ratio = (new - old).exp()
    clipped = ratio.clamp(0.8, 1.2)
    assert torch.allclose(objective, torch.min(ratio * adv, clipped * adv))
    objective.sum().backward()
    assert new.grad[:2].tolist() == [0.0, 0.0]  # moved far enough in the advantage's direction
    assert new.grad[2:].abs().min() > 0  # but not stopped from moving back


def test_forced_moves_are_not_recorded():
    found = some_trajectories(12)
    assert all(len(step.observation.legal) > 1 for t in found for step in t.steps)


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
    found = some_trajectories(4)
    batch = prepare(found, critic(), Settings())
    assert abs(batch.advantages.mean().item()) < 1e-5
    assert len(batch.actions) == sum(len(t.steps) for t in found)


def test_the_stake_is_the_trick_value_and_triple_for_a_declarer_alone():
    from helpers import auction_won_by, deal_with

    from danish_wist import CallAce, NameTrumps, Suit, TakeCat
    from danish_wist.bidding import Bid

    def finished(trumps: Suit, cat: str) -> Deal:
        deal = deal_with({1: "KS QS JS", 3: "AH"} if not cat else {1: "KS QS JS"}, cat=cat)
        auction_won_by(deal, 1, Bid(8))
        deal.apply(CallAce(Suit.HEARTS))
        deal.apply(NameTrumps(trumps))
        deal.apply(TakeCat(False))
        rng = random.Random(0)
        while not deal.is_over:
            deal.apply(rng.choice(deal.legal_actions()))
        return deal

    alone = finished(Suit.SPADES, cat="AH")  # the called ace is in the cat
    assert [selfplay.stake(alone, seat) for seat in range(4)] == [20, 60, 20, 20]
    in_clubs = finished(Suit.CLUBS, cat="")  # a Plain contract in clubs scores as Clubs
    assert [selfplay.stake(in_clubs, seat) for seat in range(4)] == [40, 40, 40, 40]


def test_collected_trajectories_carry_their_stake():
    found = some_trajectories(23, deals=16)
    for t in found:
        fixed = [selfplay.stake_is_fixed(s.observation) for s in t.steps]
        assert fixed == sorted(fixed)  # the auction and the call come first
        assert t.stake > 0 or not any(fixed)


def test_stake_scaling_evens_out_advantages_once_the_stake_is_fixed():
    """Two copies of one seat's decisions, one at four times the other's stake."""
    found = some_trajectories(29, deals=16)
    steps = next(
        t.steps
        for t in found
        if 2 <= sum(selfplay.stake_is_fixed(s.observation) for s in t.steps) < len(t.steps)
    )
    fixed = torch.tensor([selfplay.stake_is_fixed(s.observation) for s in steps])
    pair = [
        selfplay.Trajectory(steps, 100.0, stake=20),
        selfplay.Trajectory(steps, 400.0, stake=80),
    ]
    scaled = prepare(pair, critic(), Settings(stake_scaling=True))  # a critic of zeros
    plain = prepare(pair, critic(), Settings())

    def halves(batch):
        return batch.advantages[: len(steps)], batch.advantages[len(steps) :]

    first, second = halves(scaled)
    assert torch.allclose(first[fixed], second[fixed], atol=1e-5)  # the same play, the same lesson
    assert not torch.allclose(first[~fixed], second[~fixed])  # bids still weigh the true stakes
    first, second = halves(plain)
    assert not torch.allclose(first[fixed], second[fixed])
    assert torch.equal(scaled.returns, plain.returns)  # the critic still learns points


def updated(batch, chunk: int) -> list[torch.Tensor]:
    """The policy's and critic's weights after one update of fixed networks on `batch`."""
    torch.manual_seed(13)
    policy, value = Net(SMALL), critic()
    optimisers = (
        torch.optim.SGD(policy.parameters(), 0.1),
        torch.optim.SGD(value.parameters(), 0.1),
    )
    settings = Settings(batch_size=64, chunk=chunk)
    update(policy, value, copy.deepcopy(policy), optimisers, batch, settings, random.Random(1))
    return [p.detach().clone() for p in [*policy.parameters(), *value.parameters()]]


def test_updating_in_chunks_is_the_same_as_in_one_pass():
    batch = prepare(some_trajectories(13), critic(), Settings())
    whole, chunked = updated(batch, 10_000), updated(batch, 7)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(whole, chunked, strict=True))


def test_a_step_with_a_non_finite_gradient_is_skipped():
    torch.manual_seed(14)
    batch = prepare(some_trajectories(14), critic(), Settings())
    batch.advantages[:] = float("nan")
    policy, value = Net(SMALL), critic()
    before = [p.detach().clone() for p in policy.parameters()]
    optimisers = (torch.optim.AdamW(policy.parameters()), torch.optim.AdamW(value.parameters()))
    stats = update(
        policy, value, copy.deepcopy(policy), optimisers, batch, Settings(), random.Random(1)
    )
    assert stats["skipped_steps"] > 0 and "policy_loss" not in stats and "value_loss" in stats
    assert all(torch.equal(a, b) for a, b in zip(before, policy.parameters(), strict=True))


def test_non_finite_weights_stop_training_before_anything_is_saved():
    net = Net(SMALL)
    with torch.no_grad():
        net.policy.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        selfplay._check_finite(net)


def test_runner_evaluation_matches_the_arena():
    positions = random_positions(6, random.Random(5))
    with runner_for(Net(SMALL)) as runner:
        by_runner = evaluate(runner, RULE, RULE, positions)
    assert by_runner.per_deal == duplicate(RuleBot(), RuleBot(), positions).per_deal == [0.0] * 6


def test_training_runs_and_changes_the_policy(tmp_path):
    torch.manual_seed(5)
    policy = Net(SMALL)
    before = [p.clone() for p in policy.parameters()]
    history = train(policy, critic(), 2, TINY, random.Random(5), out=tmp_path)
    assert len(history) == 2 and all(e["policy_loss"] == e["policy_loss"] for e in history)
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters(), strict=True))
    assert (tmp_path / "policy.npz").exists() and (tmp_path / "log.jsonl").exists()


CHECKPOINTS = [("critic", "pt"), ("policy", "npz"), ("policy", "pt")]


def test_every_snapshot_and_evaluation_is_kept(tmp_path):
    torch.manual_seed(15)
    settings = Settings(deals_per_iteration=8, snapshot_every=2, batch_size=64)
    train(
        Net(SMALL), critic(), 3, settings, random.Random(15), tmp_path, eval_every=3, eval_deals=4
    )
    kept = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert kept == sorted(f"{n}-{i:04d}.{s}" for i in (2, 3) for n, s in CHECKPOINTS)
    evals = [json.loads(line) for line in (tmp_path / "evals.jsonl").read_text().splitlines()]
    assert [e["iteration"] for e in evals] == [3] and len(evals[0]["per_deal"]) == 4


def test_a_new_run_will_not_overwrite_an_old_one(tmp_path):
    train(Net(SMALL), critic(), 1, TINY, random.Random(16), out=tmp_path)
    with pytest.raises(FileExistsError):
        train(Net(SMALL), critic(), 1, TINY, random.Random(16), out=tmp_path)


def test_a_resumed_run_carries_on_exactly_where_it_stopped(tmp_path):
    def weights(net):
        return [p.detach().clone() for p in net.parameters()]

    torch.manual_seed(6)
    start, start_critic = Net(SMALL), critic()
    policy, value = copy.deepcopy(start), copy.deepcopy(start_critic)
    whole = train(policy, value, 3, TINY, random.Random(6), out=tmp_path / "whole")
    expected = weights(policy) + weights(value)

    policy, value = copy.deepcopy(start), copy.deepcopy(start_critic)
    train(policy, value, 2, TINY, random.Random(6), out=tmp_path / "split")
    policy, value = load(str(tmp_path / "split/policy.pt")), load(str(tmp_path / "split/critic.pt"))
    rest = train(policy, value, 3, TINY, random.Random(99), out=tmp_path / "split", resume=True)

    assert [e["iteration"] for e in rest] == [3]
    assert rest[0]["policy_loss"] == pytest.approx(whole[2]["policy_loss"])
    assert all(
        torch.allclose(a, b, atol=1e-6)
        for a, b in zip(expected, weights(policy) + weights(value), strict=True)
    )
    logged = (tmp_path / "split/log.jsonl").read_text().splitlines()
    assert [json.loads(line)["iteration"] for line in logged] == [1, 2, 3]


def test_the_saved_snapshot_pool_holds_the_past_policies(tmp_path):
    torch.manual_seed(18)
    train(Net(SMALL), critic(), 2, TINY, random.Random(18), out=tmp_path)
    state = torch.load(tmp_path / selfplay.STATE, weights_only=True)
    first = torch.load(tmp_path / "checkpoints/policy-0001.pt", weights_only=True)["state"]
    assert all(torch.equal(state["pool"]["snapshot-0"][k], first[k]) for k in first)
    assert not all(torch.equal(state["pool"]["snapshot-0"][k], state["policy"][k]) for k in first)


def test_a_resume_drops_log_lines_after_the_saved_state_and_cut_short(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"iteration": 1}\n{"iteration": 2}\n{"iteration": 3}\n{"itera')
    selfplay._keep_until(log, 2)
    assert log.read_text() == '{"iteration": 1}\n{"iteration": 2}\n'


def test_the_critic_warmup_leaves_the_policy_alone():
    torch.manual_seed(17)
    policy, value = Net(SMALL), critic()
    policy_before = [p.detach().clone() for p in policy.parameters()]
    head_before = value.value.weight.detach().clone()
    settings = Settings(deals_per_iteration=8, batch_size=64, critic_warmup=1)
    (entry,) = train(policy, value, 1, settings, random.Random(17))
    assert entry["warmup"] and "value_loss" in entry and "policy_loss" not in entry
    assert all(torch.equal(a, b) for a, b in zip(policy_before, policy.parameters(), strict=True))
    assert not torch.equal(head_before, value.value.weight)


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
    history = train(Net(SMALL), critic(), 1, settings, random.Random(9))
    assert history[0]["belief_loss"] > 0


def test_exploiter_trains_in_one_seat_against_a_frozen_target():
    from learn.selfplay import exploit_lineups

    lineups = exploit_lineups(20, random.Random(10))
    assert all(lineup.count(LEARNER) == 1 and lineup.count(TARGET) == 3 for lineup in lineups)

    torch.manual_seed(10)
    settings = Settings(deals_per_iteration=8, batch_size=64, games_in_flight=16)
    history = train(
        Net(SMALL),
        critic(),
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
