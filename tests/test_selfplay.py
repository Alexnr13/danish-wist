import copy
import json
import random
from functools import partial

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from danish_wist import Deal  # noqa: E402
from danish_wist.bots import RuleBot  # noqa: E402
from learn import inference, selfplay  # noqa: E402
from learn.arena import duplicate, random_positions  # noqa: E402
from learn.encoding import NUM_ACTIONS, Kind, card_id, encode, encode_oracle  # noqa: E402
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


def one_trajectory(values: list[float], reward: float, lam: float) -> list[float]:
    return advantages(np.array(values), np.array([reward]), np.array([len(values)]), lam).tolist()


def test_advantages_with_lambda_one_are_reward_minus_value():
    assert one_trajectory([1.0, 2.0, 3.0], 10.0, lam=1.0) == [9.0, 8.0, 7.0]


def test_advantages_with_lambda_zero_are_one_step_errors():
    assert one_trajectory([1.0, 2.0, 3.0], 10.0, lam=0.0) == [1.0, 1.0, 7.0]


def plain_advantages(values: list[float], reward: float, lam: float) -> list[float]:
    """GAE for one trajectory, step by step in Python's floats, as `prepare` had it until 30
    September."""
    result, running = [0.0] * len(values), 0.0
    for t in reversed(range(len(values))):
        following = values[t + 1] if t + 1 < len(values) else 0.0
        delta = (reward if t + 1 == len(values) else 0.0) + following - values[t]
        running = delta + lam * running
        result[t] = running
    return result


def test_trajectories_laid_end_to_end_get_the_advantages_each_would_get_alone():
    rng = np.random.default_rng(3)
    lengths = np.array([3, 1, 7, 2, 7])
    values = rng.normal(0, 100, lengths.sum()).astype(np.float32).astype(np.float64)
    rewards = rng.integers(-500, 500, len(lengths)).astype(np.float64)
    found = advantages(values, rewards, lengths, lam=0.95)
    starts = np.cumsum(lengths) - lengths
    expected = [
        a
        for start, length, reward in zip(starts, lengths, rewards, strict=True)
        for a in plain_advantages(values[start : start + length].tolist(), reward, 0.95)
    ]
    assert found.tolist() == expected  # the same numbers, not only close


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


def runner_for(net: Net, workers: int = 1, **agents):
    """A runner with the learner loaded with `net`'s weights, its networks in this process."""
    make = partial(make_agents, slots=2, **agents)
    served = selfplay.networks(net.config, 2, "cpu")
    runner = Runner(make, workers=workers, games_in_flight=16, networks=served)
    served.load(LEARNER, net.state_dict())
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
    ratios, own = {}, {}
    for explore, levels in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5)):
        with runner_for(net, explore=explore, explore_levels=levels) as runner:
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


def test_one_opponent_takes_one_to_three_seats_of_a_league_deal():
    from learn.selfplay import lineups_against

    opponents = [None, RULE, "opponent-3"] * 100
    lineups = lineups_against(opponents, random.Random(3))
    for opponent, lineup in zip(opponents, lineups, strict=True):
        if opponent is None:
            assert lineup == [LEARNER] * 4
        else:
            assert set(lineup) == {LEARNER, opponent} and 1 <= lineup.count(opponent) <= 3
    assert {lineup.count(RULE) for lineup in lineups[1::3]} == {1, 2, 3}


def test_the_learner_wins_a_deal_when_its_seats_score_more_than_nothing():
    from learn.selfplay import outcome

    lineup = [LEARNER, RULE, LEARNER, RULE]
    assert outcome(lineup, [100, -100, 100, -100]) == 1.0
    assert outcome(lineup, [-300, 300, 100, -100]) == 0.0
    assert outcome(lineup, [0, 0, 0, 0]) == 0.5


def test_the_league_draws_the_members_the_learner_struggles_against():
    from learn.selfplay import League

    league = League()
    league.add("weak", {}, 10)
    league.add("strong", {}, 20)
    league.record("weak", [1.0] * 40)  # the learner beats it every time
    league.record("strong", [0.0] * 40)
    drawn = league.draw(3000, random.Random(4))
    assert drawn.count("strong") > 10 * drawn.count("weak") > 0
    assert RULE in drawn  # RuleBot is always a member, at even odds so far
    assert league.draw(5, random.Random(4), exploiters=True) == []
    league.add("exploiter", {}, 30, exploiter=True)
    assert set(league.draw(20, random.Random(4), exploiters=True)) == {"exploiter"}


def test_a_full_league_thins_its_oldest_snapshots_most():
    from learn.selfplay import League

    league = League(size=21)
    league.add("x", {}, 5, exploiter=True)  # exploiters stay
    for iteration in range(10, 1010, 10):
        league.add(f"learner-{iteration:04d}", {}, iteration)
    kept = sorted(m.iteration for n, m in league.members.items() if n.startswith("learner"))
    assert len(league.members) == 21 and RULE in league.members and "x" in league.members
    assert kept[-1] == 1000  # the newest stays
    gaps = [b - a for a, b in zip(kept, kept[1:], strict=False)]
    assert gaps[0] > 5 * gaps[-1]  # sparse long ago, dense lately


def test_a_league_over_its_cap_drops_its_oldest_exploiters_and_their_files(tmp_path):
    from learn.selfplay import League

    league = League(size=8, folder=tmp_path, exploiters=3)
    for iteration in (10, 20, 30):
        league.add(f"learner-{iteration}", {}, iteration)
    for iteration in (5, 15, 25, 35, 45):
        league.add(f"exploiter-{iteration}", {}, iteration, exploiter=True)
    snapshots = {"learner-10", "learner-20", "learner-30"}
    assert set(league.members) == {RULE, "exploiter-25", "exploiter-35", "exploiter-45"} | snapshots
    assert {f.stem for f in tmp_path.iterdir()} == set(league.members) - {RULE}
    for iteration in (40, 50, 60, 70, 80, 90):  # the snapshots are thinned to the size
        league.add(f"learner-{iteration}", {}, iteration)
    assert len(league.members) == 8 and {RULE, "learner-90", "exploiter-45"} <= set(league.members)


def test_the_cap_on_exploiters_comes_from_the_settings_not_the_saved_state(tmp_path):
    from learn.selfplay import League

    league = League(size=50, folder=tmp_path, exploiters=10)  # a run from before the cap
    for iteration in range(10, 110, 10):
        league.add(f"exploiter-{iteration:04d}", {}, iteration, exploiter=True)
    league.add("learner-0100", {}, 100)
    again = League(size=50, folder=tmp_path, exploiters=4)
    again.load_state_dict(league.state_dict())
    assert len(again.members) == 12  # capped at the first member added
    again.add("learner-0110", {}, 110)
    exploiters = sorted(n for n, m in again.members.items() if m.exploiter)
    assert exploiters == ["exploiter-0070", "exploiter-0080", "exploiter-0090", "exploiter-0100"]
    assert {RULE, "learner-0100", "learner-0110"} <= set(again.members)
    assert len(list(tmp_path.iterdir())) == 6


def test_the_league_keeps_its_members_weights_in_files(tmp_path):
    from learn.selfplay import League

    league = League(size=3, folder=tmp_path)
    weights = {"w": torch.arange(3.0)}
    for iteration in (10, 20, 30):
        league.add(f"learner-{iteration}", weights, iteration)
    league.record("learner-30", [1.0, 0.0])
    again = League(folder=tmp_path)
    again.load_state_dict(league.state_dict())
    assert set(again.members) == set(league.members) and len(list(tmp_path.iterdir())) == 2
    assert torch.equal(again.members["learner-30"].weights["w"], weights["w"])
    assert again.win_rate("learner-30") == league.win_rate("learner-30") == 0.5


def test_prepare_normalises_advantages():
    found = some_trajectories(4)
    batch = prepare(found, critic(), Settings())
    assert abs(batch.advantages.mean().item()) < 1e-5
    assert len(batch.actions) == sum(len(t.steps) for t in found)


def plain_padded(sequences: list, device: torch.device) -> selfplay.Padded:
    """`Padded.of` as it was until 30 September: padded by NumPy, then sent."""
    tokens, lengths = inference.pad(sequences, dtype=np.int16)
    tokens = np.pad(
        tokens, ((0, 0), (0, selfplay._rounded(tokens.shape[1]) - tokens.shape[1]), (0, 0))
    )
    return selfplay.Padded(
        torch.from_numpy(tokens).to(device).int(), torch.from_numpy(lengths).to(device)
    )


@torch.no_grad()
def plain_prepare(trajectories: list, critic: Net, settings: Settings) -> selfplay.Batch:
    """`prepare` as it was until 30 September: step by step in Python, and the critic's pass
    compiled (on an NVIDIA GPU) in passes of `chunk` rows, the last filled up with its first."""
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    device = next(critic.parameters()).device
    oracle = plain_padded([s.oracle for s in steps], device)
    critic.eval()
    values = torch.empty(len(steps), device=device)
    values_of, bins = selfplay._compiled(selfplay._values, device), value_bins(device)
    with selfplay._attention(device):
        for start in range(0, len(steps), settings.chunk):
            rows = list(range(start, min(start + settings.chunk, len(steps))))
            index = torch.tensor(rows + rows[:1] * (settings.chunk - len(rows)), device=device)
            values[rows] = values_of(critic, *oracle.take(index), bins)[: len(rows)]
    values = values.cpu()
    found, start = [], 0
    for trajectory in trajectories:
        end = start + len(trajectory.steps)
        found += plain_advantages(
            values[start:end].tolist(), trajectory.reward, settings.gae_lambda
        )
        start = end
    rewards = torch.tensor([t.reward for t in trajectories for _ in t.steps])
    adv = torch.tensor(found, dtype=torch.float32)
    returns = adv + values
    if settings.stake_scaling:
        adv = adv * selfplay.stake_weights(trajectories)
    legal = torch.zeros(len(steps), NUM_ACTIONS, dtype=torch.bool)
    for row, step in enumerate(steps):
        legal[row, [int(i) for i in step.observation.legal]] = True
    return selfplay.Batch(
        plain_padded([s.observation.tokens for s in steps], device),
        legal.to(device),
        oracle,
        torch.from_numpy(np.stack([s.belief for s in steps])).to(device).long(),
        torch.tensor([s.action for s in steps]).to(device),
        torch.tensor([s.policy_log_prob for s in steps]).to(device),
        torch.tensor([s.log_prob for s in steps]).to(device),
        ((adv - adv.mean()) / (adv.std() + 1e-8)).to(device),
        returns.to(device),
        selfplay.explained_variance(values, rewards),
    )


def valued_critic(seed: int) -> Net:
    """A critic whose values differ from deal to deal (a fresh one's are all 0)."""
    torch.manual_seed(seed)
    made = critic()
    torch.nn.init.normal_(made.value.weight, std=0.3)
    return made


def assert_same_batch(found: selfplay.Batch, expected: selfplay.Batch, exact: bool = True):
    """The same batch, to the last bit; or not `exact`, the advantages' normalisation and the
    critic's explained variance to rounding (the GPU's sums, where `prepare` has graphs)."""
    for name in ("inputs", "oracle"):
        for part in ("tokens", "lengths"):
            a, b = getattr(getattr(found, name), part), getattr(getattr(expected, name), part)
            assert a.dtype == b.dtype and a.device == b.device and torch.equal(a, b), (name, part)
    for name in ("legal", "beliefs", "actions", "old_log_probs", "played_log_probs", "returns"):
        a, b = getattr(found, name), getattr(expected, name)
        assert a.dtype == b.dtype and a.device == b.device and torch.equal(a, b), name
    a, b = found.advantages, expected.advantages
    assert a.dtype == b.dtype and a.device == b.device
    if exact:
        assert torch.equal(a, b) and found.value_ev == expected.value_ev
    else:  # normalised by the GPU's mean and deviation: about 1e-8 apart
        assert torch.allclose(a, b, rtol=0, atol=1e-6) and a.abs().max() > 1
        assert found.value_ev == pytest.approx(expected.value_ev, rel=1e-6)


@pytest.mark.parametrize("stake_scaling", [False, True])
def test_prepare_gives_what_it_gave_step_by_step_in_python(stake_scaling):
    """`prepare` gathers the steps with NumPy: the same batch, to the last bit."""
    found = some_trajectories(33, deals=12)
    settings = Settings(chunk=64, stake_scaling=stake_scaling)  # several passes, the last short
    value = valued_critic(33)
    assert_same_batch(prepare(found, value, settings), plain_prepare(found, value, settings))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
@pytest.mark.parametrize("graphed", [False, True])
def test_prepare_on_the_gpu_gives_what_it_gave_step_by_step(graphed):
    """On the GPU too, with the critic's pass compiled; or replayed from CUDA graphs
    (`UpdateGraphs.forward`) that were captured on other rows (the deals in another order),
    and the advantages normalised by the GPU's sums: the same to rounding."""
    found = some_trajectories(34, deals=12)
    settings = Settings(chunk=64)
    value = valued_critic(34).cuda()
    graphs = selfplay.UpdateGraphs() if graphed else None
    if graphed:
        prepare(found[::-1], value, settings, graphs)
    batch = prepare(found, value, settings, graphs)
    assert_same_batch(batch, plain_prepare(found, value, settings), exact=not graphed)
    assert not graphed or len(graphs.passes) == 1  # replayed, not captured again


def log_chances(policy: Net, tokens, padding, legal, actions) -> torch.Tensor:
    """The policy's log-chance of each move: the proximal pass of collecting while updating."""
    logits = policy.heads(selfplay._summarise(policy, tokens, padding), legal)[0]
    return torch.log_softmax(logits, dim=-1).gather(1, actions[:, None]).squeeze(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
def test_a_graphed_forward_pass_computes_what_the_compiled_one_does():
    """`_forward` replayed from CUDA graphs gives what its compiled passes give, to the last bit:
    here the policy's log-chance of each move, on two batches in passes of two sizes, each
    graph captured on other rows (the batch's in reverse) before it is replayed on these."""
    torch.manual_seed(36)
    policy, graphs = Net(SMALL).cuda(), selfplay.UpdateGraphs()
    function = partial(log_chances, policy)
    for seed, size in ((36, 64), (37, 48)):
        batch = prepare(some_trajectories(seed), critic().cuda(), Settings())
        count = len(batch.actions)

        def inputs(rows, batch=batch):
            return (*batch.inputs.take(rows), batch.legal[rows], batch.actions[rows])

        def reversed_inputs(rows, count=count, inputs=inputs):
            return inputs(count - 1 - rows)

        compiled = selfplay._forward(function, inputs, count, size)
        selfplay._forward(function, reversed_inputs, count, size, graphs=graphs)
        assert torch.equal(
            selfplay._forward(function, inputs, count, size, graphs=graphs), compiled
        )
        assert compiled.isfinite().all() and (compiled < 0).any()
    assert len(graphs.passes) == 2  # one for each shape, replayed


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
@pytest.mark.parametrize("bfloat16", [False, True])
def test_the_compiled_update_on_a_gpu_agrees_with_the_cpu(bfloat16, monkeypatch):
    """On CUDA the update is compiled and its trunk runs in bfloat16 (`_summarise`): in
    float32 it is the CPU's update, and in bfloat16 only rounding apart from it."""
    if not bfloat16:
        monkeypatch.setattr(selfplay, "_summarise", lambda net, *inputs: net.summarise(*inputs))
    found = some_trajectories(31)
    torch.manual_seed(31)
    start, start_critic = Net(SMALL), critic()
    stats = {}
    for device in ("cpu", "cuda"):
        policy, value = copy.deepcopy(start).to(device), copy.deepcopy(start_critic).to(device)
        optimisers = (torch.optim.AdamW(policy.parameters()), torch.optim.AdamW(value.parameters()))
        batch = prepare(found, value, Settings())
        settings = Settings(batch_size=64)
        magnet = copy.deepcopy(policy)
        stats[device] = update(policy, value, magnet, optimisers, batch, settings, random.Random(1))
    tolerance = 1e-3 if bfloat16 else 1e-5
    for key in ("policy_loss", "value_loss", "entropy", "magnet_kl", "belief_loss", "approx_kl"):
        assert stats["cuda"][key] == pytest.approx(stats["cpu"][key], abs=tolerance), key


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


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


def weights_of(*nets) -> list[torch.Tensor]:
    return [p.detach().clone() for net in nets for p in net.parameters()]


def gpu_learning(seed: int, settings: Settings, graphed: bool):
    """Fixed networks learning on the GPU, after one eager update on the deals of `seed` (a
    new optimiser makes its state in its first step), and those deals."""
    found = some_trajectories(seed)
    torch.manual_seed(seed)
    learning = selfplay.Learning.start(Net(SMALL), critic(), settings, "cuda", graphed)
    nets = (learning.policy, learning.critic, learning.magnet, learning.optimisers)
    update(*nets, prepare(found, learning.critic, settings), settings, random.Random(0))
    return learning, nets, found


def state_of(learning) -> list[torch.Tensor]:
    """The tensors a learning step changes: the weights and the optimisers' state."""
    nets = (learning.policy, learning.critic, learning.magnet)
    states = [s for optimiser in learning.optimisers for s in optimiser.state.values()]
    return [p.data for net in nets for p in net.parameters()] + [
        t for state in states for t in state.values()
    ]


@cuda
@pytest.mark.parametrize("chunk", [4096, 256])  # the minibatch in one pass, and in three
def test_a_graphed_step_computes_what_today_s_does(chunk):
    """On the same minibatch, a step replayed from CUDA graphs (`UpdateGraphs`) computes what
    `_step` does, whether AdamW works out its step size on the CPU (today's) or on the GPU as
    the graphed step does (`capturable`). The graphs are captured on other advantages first.

    The GPU's sums come in no fixed order, and now and then that flips the step of a few
    weights whose gradient is nearly 0 (Adam scales each weight's step to about the learning
    rate), so the weights are compared by their mean difference."""
    settings = Settings(batch_size=4096, chunk=chunk)  # every decision in one minibatch
    results = {}
    for mode in ("today", "capturable", "graphed"):
        learning, nets, found = gpu_learning(31, Settings(batch_size=64), mode == "graphed")
        for optimiser in learning.optimisers:
            selfplay._set_capturable(optimiser, mode != "today")
        batch = prepare(found, learning.critic, settings)
        if mode == "graphed":  # capture on other inputs, then go back to where the others are
            saved = [t.clone() for t in state_of(learning)]
            other = copy.copy(batch)
            other.advantages = batch.advantages.flip(0)
            update(*nets, other, settings, random.Random(1), graphs=learning.graphs)
            for tensor, value in zip(state_of(learning), saved, strict=True):
                tensor.copy_(value)
        stats = update(*nets, batch, settings, random.Random(1), graphs=learning.graphs)
        results[mode] = stats, weights_of(learning.policy, learning.critic)
    graphed, weights = results["graphed"]
    assert len(learning.graphs.steps[learning.policy].passes) == 1  # replayed, not eager
    for mode in ("today", "capturable"):
        stats, expected = results[mode]
        assert graphed.keys() == stats.keys() and graphed["skipped_steps"] == 0
        for key in stats:
            assert graphed[key] == pytest.approx(stats[key], rel=1e-4, abs=1e-7), (mode, key)
        differences = torch.cat([(a - b).flatten() for a, b in zip(weights, expected, strict=True)])
        assert differences.abs().mean() < 1e-6, mode  # the step moves them by 8e-5 on average


@cuda
def test_a_graphed_step_with_a_non_finite_gradient_is_skipped():
    settings = Settings(batch_size=64)
    learning, nets, found = gpu_learning(14, settings, graphed=True)
    graphs = learning.graphs
    update(
        *nets, prepare(found, learning.critic, settings), settings, random.Random(1), graphs=graphs
    )
    before = weights_of(learning.policy)
    steps = [s["step"].clone() for s in learning.optimisers[0].state.values()]
    batch = prepare(found, learning.critic, settings)
    batch.advantages[:] = float("nan")
    stats = update(*nets, batch, settings, random.Random(2), graphs=graphs)
    assert stats["skipped_steps"] == -(-len(batch.actions) // 64)  # every step of the policy's
    assert "policy_loss" not in stats and "value_loss" in stats
    assert all(torch.equal(a, b) for a, b in zip(before, weights_of(learning.policy), strict=True))
    after = [s["step"] for s in learning.optimisers[0].state.values()]
    assert all(torch.equal(a, b) for a, b in zip(steps, after, strict=True))  # nor AdamW's state
    batch = prepare(found, learning.critic, settings)  # and nothing of the skipped steps stays
    stats = update(*nets, batch, settings, random.Random(3), graphs=graphs)
    assert stats["skipped_steps"] == 0 and torch.isfinite(torch.tensor(stats["policy_grad_norm"]))


@cuda
def test_the_graphed_step_follows_a_change_of_learning_rate():
    """A graph keeps the learning rate it was captured with, so the clip and AdamW are captured
    again when it changes (as a schedule would change it): at 0 the weights stay put."""
    settings = Settings(batch_size=64)
    learning, nets, found = gpu_learning(21, settings, graphed=True)
    update(
        *nets,
        prepare(found, learning.critic, settings),
        settings,
        random.Random(1),
        graphs=learning.graphs,
    )
    before = weights_of(learning.policy, learning.critic)
    for optimiser in learning.optimisers:
        optimiser.param_groups[0]["lr"] = 0.0
    batch = prepare(found, learning.critic, settings)
    stats = update(*nets, batch, settings, random.Random(2), graphs=learning.graphs)
    assert stats["skipped_steps"] == 0 and stats["policy_grad_norm"] > 0
    after = weights_of(learning.policy, learning.critic)
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


@cuda
@pytest.mark.parametrize("graphed", [True, False])
def test_a_run_resumes_on_the_gpu_whether_or_not_its_state_was_saved_graphed(tmp_path, graphed):
    """Earlier runs, saved with AdamW's step counts on the CPU, resume graphed, and a graphed
    run's state (the counts on the GPU) resumes eagerly; the counts carry on either way."""
    torch.manual_seed(6)
    run = partial(train, settings=TINY, device="cuda", out=tmp_path)
    run(Net(SMALL), critic(), 2, rng=random.Random(6), graphed=not graphed)
    policy, value = load(str(tmp_path / "policy.pt")), load(str(tmp_path / "critic.pt"))
    rest = run(policy, value, 3, rng=random.Random(99), resume=True, graphed=graphed)
    assert rest[0]["iteration"] == 3 and rest[0]["skipped_steps"] == 0
    logged = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    steps = sum(-(-entry["decisions"] // TINY.batch_size) for entry in logged)
    state = torch.load(tmp_path / selfplay.STATE, map_location="cpu", weights_only=True)
    for saved in state["optimisers"]:
        assert saved["param_groups"][0]["capturable"] == graphed
        assert {int(s["step"]) for s in saved["state"].values()} == {steps}


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


CHECKPOINTS = [
    ("critic", "pt"),
    ("magnet", "npz"),
    ("magnet", "pt"),
    ("policy", "npz"),
    ("policy", "pt"),
]


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


def test_the_magnet_is_kept_beside_each_checkpoint_as_a_policy(tmp_path):
    from dataclasses import replace

    torch.manual_seed(19)
    train(Net(SMALL), critic(), 2, replace(TINY, magnet_ema=0.5), random.Random(19), tmp_path)
    state = torch.load(tmp_path / selfplay.STATE, weights_only=True)
    kept = torch.load(tmp_path / "checkpoints/magnet-0002.pt", weights_only=True)["state"]
    assert all(torch.equal(kept[k], state["magnet"][k]) for k in kept)
    assert not all(torch.equal(kept[k], state["policy"][k]) for k in kept)  # an average
    exported = load(str(tmp_path / "checkpoints/magnet-0002.npz")).state_dict()
    assert all(torch.equal(exported[k], kept[k]) for k in kept)


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


def test_the_cooldown_takes_the_learning_rates_to_a_tenth_over_the_last_iterations():
    shares = [selfplay.cooldown_share(i, 10, 4) for i in range(1, 11)]
    assert shares[:6] == [1.0] * 6
    assert shares[6:] == pytest.approx([0.775, 0.55, 0.325, 0.1]) and shares[-1] == 0.1
    assert [selfplay.cooldown_share(i, 10, 0) for i in range(1, 11)] == [1.0] * 10


def rates_of_each_update(monkeypatch) -> list[tuple[float, float]]:
    """The policy's and the critic's learning rates at each update of the runs that follow."""
    rates = []

    def update(policy, critic, magnet, optimisers, *args, **kwargs):
        rates.append(tuple(optimiser.param_groups[0]["lr"] for optimiser in optimisers))
        return plain(policy, critic, magnet, optimisers, *args, **kwargs)

    plain = selfplay.update
    monkeypatch.setattr(selfplay, "update", update)
    return rates


def test_a_cooldown_sets_the_rates_of_its_last_iterations_only(tmp_path, monkeypatch):
    from dataclasses import replace

    rates = rates_of_each_update(monkeypatch)
    torch.manual_seed(7)
    history = train(Net(SMALL), critic(), 4, replace(TINY, cooldown=2), random.Random(7), tmp_path)
    base = (TINY.policy_lr, TINY.critic_lr)
    assert rates[:2] == [base, base]
    assert rates[2] == pytest.approx((0.55 * base[0], 0.55 * base[1]))
    assert rates[3] == pytest.approx((0.1 * base[0], 0.1 * base[1]))
    assert [entry["lr_share"] for entry in history] == [1.0, 1.0, 0.55, 0.1]


def test_without_a_cooldown_a_run_is_as_it_was(tmp_path, monkeypatch):
    """`cooldown` 0 changes nothing: the rates are the settings' at every update, and the run
    is the same, bit for bit, as one whose rates are never set."""
    torch.manual_seed(8)
    start, start_critic = Net(SMALL), critic()
    found = {}
    for name in ("never set", "cooldown 0"):
        with monkeypatch.context() as patch:
            if name == "never set":
                patch.setattr(selfplay.Learning, "set_rates", lambda *args: None)
            rates = rates_of_each_update(patch)
            policy, value = copy.deepcopy(start), copy.deepcopy(start_critic)
            history = train(policy, value, 3, TINY, random.Random(8), tmp_path / name)
        found[name] = rates, history, weights_of(policy, value)
    (rates, history, weights), (same_rates, same_history, same_weights) = found.values()
    assert rates == same_rates == [(TINY.policy_lr, TINY.critic_lr)] * 3
    assert all("lr_share" not in entry for entry in same_history)
    for entry in [*history, *same_history]:
        for timing in ("time", "collect_s", "update_s"):
            entry.pop(timing)
    assert history == same_history
    assert all(torch.equal(a, b) for a, b in zip(weights, same_weights, strict=True))


class Stop(Exception):
    pass


@pytest.mark.parametrize(
    "device, graphed",
    [
        ("cpu", False),
        pytest.param("cuda", False, marks=cuda),
        pytest.param("cuda", True, marks=cuda),
    ],
)
def test_a_run_stopped_in_the_middle_of_a_cooldown_carries_on_exactly(
    tmp_path, monkeypatch, device, graphed
):
    """A run to iteration 4 cooling down over its last 3 stops after iteration 2 and is resumed:
    it carries on with the rates and the weights of the run that did not stop. Graphed, each
    rate reaches the graphed optimiser step, captured again for it."""
    from dataclasses import replace

    settings = replace(TINY, cooldown=3)
    rates = rates_of_each_update(monkeypatch)
    captured = []  # the rates the graphed optimiser steps replayed were captured with

    def stepping(self):
        graph = graphed_stepping(self)
        captured.append(self.stepping[0][0]["lr"])
        return graph

    graphed_stepping = selfplay._GraphedStep._stepping
    monkeypatch.setattr(selfplay._GraphedStep, "_stepping", stepping)
    torch.manual_seed(9)
    start, start_critic = Net(SMALL), critic()
    run = partial(train, iterations=4, settings=settings, device=device, graphed=graphed)
    policy, value = copy.deepcopy(start), copy.deepcopy(start_critic)
    whole = run(policy, value, rng=random.Random(9), out=tmp_path / "whole")
    expected = weights_of(policy, value)
    shares = [selfplay.cooldown_share(i, 4, 3) for i in range(1, 5)]
    assert shares == pytest.approx([1.0, 0.7, 0.4, 0.1])
    base = (settings.policy_lr, settings.critic_lr)
    assert rates == [(base[0] * s, base[1] * s) for s in shares]
    assert [entry["lr_share"] for entry in whole] == [round(s, 6) for s in shares]
    if graphed:  # every step but a new optimiser's first (eager) is replayed
        assert set(captured) == {rate * s for rate in base for s in shares}

    learned = selfplay._learn
    calls = []

    def learn(*args, **kwargs):  # stops the run in its third iteration, after its state at 2
        calls.append(1)
        if len(calls) == 3:
            raise Stop
        return learned(*args, **kwargs)

    rates.clear()
    with monkeypatch.context() as patch:
        patch.setattr(selfplay, "_learn", learn)
        policy, value = copy.deepcopy(start), copy.deepcopy(start_critic)
        with pytest.raises(Stop):
            run(policy, value, rng=random.Random(9), out=tmp_path / "split")
    policy, value = load(str(tmp_path / "split/policy.pt")), load(str(tmp_path / "split/critic.pt"))
    rest = run(policy, value, rng=random.Random(99), out=tmp_path / "split", resume=True)

    assert rates == [(base[0] * s, base[1] * s) for s in shares]
    assert [e["iteration"] for e in rest] == [3, 4] and rest[-1]["lr_share"] == pytest.approx(0.1)
    for key in ("policy_loss", "value_loss", "approx_kl"):
        expected_values = [e[key] for e in whole[2:]]
        assert [e[key] for e in rest] == pytest.approx(expected_values, rel=1e-4, abs=1e-7)
    found = weights_of(policy, value)
    differences = torch.cat([(a - b).flatten() for a, b in zip(expected, found, strict=True)])
    if device == "cpu":
        assert differences.abs().max() <= 1e-6
    else:  # the GPU's sums come in no fixed order: `test_a_graphed_step_computes_what_today_s_does`
        assert differences.abs().mean() < 1e-6  # a step moves them by about 1e-4


def test_the_league_holds_the_start_past_policies_and_rulebot(tmp_path):
    torch.manual_seed(18)
    history = train(Net(SMALL), critic(), 2, TINY, random.Random(18), out=tmp_path)
    state = torch.load(tmp_path / selfplay.STATE, weights_only=True)
    members = state["league"]["members"]
    assert set(members) == {RULE, "start", "learner-0001", "learner-0002"}
    first = torch.load(tmp_path / "checkpoints/policy-0001.pt", weights_only=True)["state"]
    kept = torch.load(tmp_path / "league/learner-0001.pt", weights_only=True)
    assert all(torch.equal(kept[k], first[k]) for k in first)
    assert not all(torch.equal(kept[k], state["policy"][k]) for k in first)
    against = history[1]["league"]["against"]  # members met in iteration 2, and win rates
    assert against and all(0 < rate < 1 for _, rate in against.values())


def test_an_exploiter_trains_against_the_learner_and_joins_the_league(tmp_path):
    from dataclasses import replace

    torch.manual_seed(25)
    settings = replace(TINY, exploit_every=2, exploit_iterations=1)
    history = train(Net(SMALL), critic(), 2, settings, random.Random(25), out=tmp_path)
    assert "exploiter_margin" not in history[0] and "exploiter_margin" in history[1]
    state = torch.load(tmp_path / selfplay.STATE, weights_only=True)
    assert state["league"]["members"]["exploiter-0002"]["exploiter"]


def test_the_magnet_follows_the_policy_by_its_moving_average():
    from dataclasses import replace

    from learn.selfplay import Learning

    torch.manual_seed(26)
    learning = Learning.start(Net(SMALL), critic(), Settings(), "cpu")
    before = [p.detach().clone() for p in learning.magnet.parameters()]
    with torch.no_grad():
        for p in learning.policy.parameters():
            p.add_(1.0)
    learning.move_magnet(replace(Settings(), magnet_ema=0.25), 1)
    for old, new, now in zip(
        before, learning.magnet.parameters(), learning.policy.parameters(), strict=True
    ):
        assert torch.allclose(new, old + 0.25 * (now - old))
    learning.move_magnet(Settings(snapshot_every=10), 3)  # no average: a copy every 10 only
    assert not torch.equal(next(learning.magnet.parameters()), next(learning.policy.parameters()))
    learning.move_magnet(Settings(snapshot_every=10), 10)
    assert torch.equal(next(learning.magnet.parameters()), next(learning.policy.parameters()))


def test_a_resume_records_the_settings_it_changed(tmp_path, monkeypatch):
    import sys

    from learn.model import save

    save(Net(SMALL), str(tmp_path / "policy.pt"))
    out = tmp_path / "rl"
    small = ["--deals", "8", "--eval-every", "0", "--workers", "1", "--device", "cpu"]
    argv = ["selfplay", "--init", str(tmp_path / "policy.pt"), "--out", str(out), *small]
    monkeypatch.setattr(sys, "argv", [*argv, "--iterations", "1"])
    selfplay.main()
    for name in ("run.json", "settings.json"):  # as a run from before the cap
        found = json.loads((out / name).read_text())
        found.get("args", found).pop("league_exploiters")
        (out / name).write_text(json.dumps(found))
    resume = ["selfplay", "--resume", str(out), "--iterations", "2", "--workers", "1"]
    monkeypatch.setattr(sys, "argv", [*resume, "--device", "cpu"])
    selfplay.main()
    (resumed,) = json.loads((out / "run.json").read_text())["resumed"]
    assert resumed["iterations"] == 2 and resumed["settings"] == {"league_exploiters": 20}
    assert len((out / "log.jsonl").read_text().splitlines()) == 2


def test_a_setting_given_at_a_resume_is_kept_by_the_next(tmp_path, monkeypatch):
    """A resume takes the run's settings from its run.json. One that the run lacks (newer than
    the run, as `--cooldown` is than rl-006) and that a resume is given is written there, so a
    later resume without it keeps it; the settings left at their defaults are not."""
    import sys

    from learn.model import save

    save(Net(SMALL), str(tmp_path / "policy.pt"))
    out = tmp_path / "rl"
    small = ["--deals", "8", "--eval-every", "0", "--workers", "1", "--device", "cpu"]
    argv = ["selfplay", "--init", str(tmp_path / "policy.pt"), "--out", str(out), *small]
    monkeypatch.setattr(sys, "argv", [*argv, "--iterations", "1"])
    selfplay.main()
    for name in ("run.json", "settings.json"):  # as a run from before the cooldown
        found = json.loads((out / name).read_text())
        found.get("args", found).pop("cooldown")
        (out / name).write_text(json.dumps(found))
    before = json.loads((out / "run.json").read_text())["args"]
    resume = ["selfplay", "--resume", str(out), "--workers", "1", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", [*resume, "--iterations", "3", "--cooldown", "2"])
    selfplay.main()
    run = json.loads((out / "run.json").read_text())
    assert run["args"] == before | {"iterations": 3, "cooldown": 2}
    monkeypatch.setattr(sys, "argv", [*resume, "--iterations", "4"])
    selfplay.main()
    run = json.loads((out / "run.json").read_text())
    assert run["args"] == before | {"iterations": 4, "cooldown": 2}
    assert [resumed["settings"] for resumed in run["resumed"]] == [{"cooldown": 2}] * 2
    logged = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [entry.get("lr_share") for entry in logged] == [None, 0.55, 0.1, 0.1]


@cuda
@pytest.mark.parametrize("first, then", [([], ["--graphed-update"]), (["--graphed-update"], [])])
def test_the_update_is_graphed_only_when_asked_and_a_resume_may_switch(
    tmp_path, monkeypatch, first, then
):
    """A run's update launches its kernels from Python unless `--graphed-update` asks for its
    steps replayed from CUDA graphs (which keep AdamW's step counts on the GPU, `capturable`,
    as the saved state shows), and `prepare`'s critic pass with them; a resume may switch
    either way, recording it."""
    import sys

    from learn.model import save

    def graphed() -> set[bool]:
        state = torch.load(out / selfplay.STATE, map_location="cpu", weights_only=True)
        return {g["capturable"] for saved in state["optimisers"] for g in saved["param_groups"]}

    prepared_with: list[bool] = []  # whether each `prepare` had graphs

    def prepare(trajectories, critic, settings, graphs=None):
        prepared_with.append(graphs is not None)
        return plain(trajectories, critic, settings, graphs)

    plain = selfplay.prepare
    monkeypatch.setattr(selfplay, "prepare", prepare)
    save(Net(SMALL), str(tmp_path / "policy.pt"))
    out = tmp_path / "rl"
    small = ["--deals", "8", "--eval-every", "0", "--workers", "1", "--device", "cuda"]
    argv = ["selfplay", "--init", str(tmp_path / "policy.pt"), "--out", str(out), *small]
    monkeypatch.setattr(sys, "argv", [*argv, "--iterations", "2", *first])
    selfplay.main()
    assert graphed() == {bool(first)} and prepared_with == [bool(first)] * 2
    resume = ["selfplay", "--resume", str(out), "--iterations", "3", *small[4:]]
    monkeypatch.setattr(sys, "argv", [*resume, *then])
    selfplay.main()
    assert graphed() == {bool(then)} and prepared_with[2:] == [bool(then)]
    (resumed,) = json.loads((out / "run.json").read_text())["resumed"]
    assert resumed.get("graphed_update", False) == bool(then)
    assert len((out / "log.jsonl").read_text().splitlines()) == 3


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


def test_collecting_in_worker_processes_is_reproducible():
    """The networks run in this process for all the workers, which keep in step, so the
    networks see the same batches and the learner makes the same decisions every time."""
    torch.manual_seed(11)
    net = Net(SMALL)
    runs = []
    for _ in range(2):
        with runner_for(net, workers=3) as runner:
            found = collect(runner, [[LEARNER, RULE, LEARNER, RULE]] * 9, random.Random(11))
        assert found and all(t.steps and t.steps[0].oracle.dtype.name == "int16" for t in found)
        runs.append(sorted([s.action for s in t.steps] for t in found))
    assert runs[0] == runs[1]


def test_an_exploiter_starts_level_with_its_policy_and_reports_its_margin(tmp_path, monkeypatch):
    import sys

    from learn import exploit
    from learn.model import export, save

    torch.manual_seed(24)
    policy = Net(SMALL)
    save(policy, str(tmp_path / "policy.pt"))
    export(policy, str(tmp_path / "policy.npz"))
    clone = exploit.margin(str(tmp_path / "policy.npz"), str(tmp_path / "policy.pt"), 4, [1])
    assert clone.per_deal == [0.0] * 4  # the same network, greedy: nothing to gain yet

    out = tmp_path / "x"
    argv = ["exploit", str(tmp_path / "policy.pt"), "--out", str(out), "--iterations", "1"]
    argv += ["--deals", "8", "--margin-deals", "3", "--workers", "1", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    exploit.main()
    found = json.loads((out / "margin.json").read_text())
    assert found["deals"] == 3 * len(exploit.SEEDS) and found["iterations"] == 1
    assert len((out / "log.jsonl").read_text().splitlines()) == 1
