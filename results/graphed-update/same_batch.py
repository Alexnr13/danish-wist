"""Graphing the update (TRAINING.md, Results): on rl-006's real networks at 18,000 and one real
batch (1024 deals from its league, as the run draws them), does the graphed update compute what
the update launched from Python does?

- The first minibatch's gradients, parameter by parameter: from `_step`'s loss launched from
  Python, and from the graphed pass (`_GraphedStep`).
- One whole update (about 72 minibatches) from the same state, six times in each of three modes:
  today's (`today`: AdamW's step counts on the CPU), the same with them on the GPU
  (`capturable`, as the graphed update keeps them), and graphed. The update is not repeatable
  run to run (the GPU's sums are in no fixed order, and bfloat16 and Adam carry the
  differences on), so the modes are compared against that spread.

    python results/graphed-update/same_batch.py > results/graphed-update/same_batch.txt
"""

import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/overlap-prototype"))

import timing  # noqa: E402  (restore(), and the served networks' captures made thread-local)

from learn import selfplay as sp  # noqa: E402
from learn.runner import Runner  # noqa: E402

MODES = ["today", "capturable", "graphed"]
REPEATS = 6
KEYS = [
    "approx_kl",
    "clip_fraction",
    "entropy",
    "policy_loss",
    "value_loss",
    "policy_grad_norm",
    "critic_grad_norm",
]


def batch_of_rl006() -> list:
    settings, learning, league, rng = timing.restore("cuda", False)
    make = partial(
        sp.make_agents,
        slots=sp.SLOTS,
        explore=settings.explore_bids,
        explore_levels=settings.explore_levels,
    )
    served = sp.networks(learning.policy.config, sp.SLOTS, "cuda")
    workers, games = timing.WORKERS, settings.games_in_flight
    with Runner(make, workers=workers, games_in_flight=games, networks=served) as runner:
        runner.networks.load(sp.LEARNER, learning.policy.state_dict())
        lineups, _ = sp._league_lineups(runner, league, {}, settings, rng)
        return sp.collect(runner, lineups, rng)


def first_gradients(found: list) -> None:
    settings, learning, _, _ = timing.restore("cuda", True)
    batch = sp.prepare(found, learning.critic, settings)
    rows = torch.arange(settings.batch_size, device="cuda")
    bins = sp.value_bins("cuda")
    losses = {
        "policy": sp._policy_loss(learning.policy, learning.magnet, batch, settings, rows),
        "critic": sp._critic_loss(learning.critic, batch, bins, settings.batch_size),
    }
    optimisers = dict(zip(losses, learning.optimisers, strict=True))
    learning.policy.train(), learning.critic.train(), learning.magnet.eval()
    with sp._attention(torch.device("cuda")):
        for name, loss in losses.items():
            net = learning.policy if name == "policy" else learning.critic
            net.zero_grad(set_to_none=True)
            value, terms = loss.terms(*loss.inputs(rows, len(rows)))
            value.backward()
            eager = {id(p): p.grad.clone() for p in net.parameters() if p.grad is not None}
            del value, terms
            net.zero_grad(set_to_none=True)
            step = sp._GraphedStep(net, optimisers[name])
            inputs = loss.inputs(rows, len(rows))
            graph, statics = step._pass(loss.terms, inputs)
            torch._foreach_zero_([*step.grads, step.totals])
            for static, value in zip(statics, inputs, strict=True):
                if isinstance(value, torch.Tensor):
                    static.copy_(value)
                else:
                    static.fill_(value)
            graph.replay()
            graphed = {id(p): g for p, g in zip(step.used, step.grads, strict=True)}
            worst = max(
                ((eager[i] - graphed[i]).norm() / eager[i].norm().clamp(min=1e-30)).item()
                for i in eager.keys() & graphed.keys()
            )
            norm = torch.stack([g.norm() for g in eager.values()]).norm().item()
            print(
                f"{name}: {len(eager)} parameters with a gradient from Python, {len(graphed)} "
                f"graphed, {len(eager.keys() ^ graphed.keys())} not in both; the norm "
                f"{norm:.6f} and {step.norm.item():.6f}; the largest relative difference in a "
                f"parameter's gradient {worst:.1e}"
            )


def updates(found: list) -> None:
    results: dict[str, list[dict]] = {mode: [] for mode in MODES}
    for mode in MODES * REPEATS:
        settings, learning, _, _ = timing.restore("cuda", mode == "graphed")
        for optimiser in learning.optimisers:
            sp._set_capturable(optimiser, mode != "today")
        batch = sp.prepare(found, learning.critic, settings)
        nets = (learning.policy, learning.critic, learning.magnet, learning.optimisers)
        stats = sp.update(*nets, batch, settings, random.Random(7), graphs=learning.graphs)
        results[mode].append(stats)
    print(f"\none update, {REPEATS} times in each mode: mean (sd), and t against today's")
    print(f"{'':18}" + "".join(f"{mode:>30}" for mode in MODES))
    for key in KEYS:
        base = np.array([stats[key] for stats in results["today"]])
        cells = []
        for mode in MODES:
            values = np.array([stats[key] for stats in results[mode]])
            error = np.sqrt(base.var(ddof=1) / len(base) + values.var(ddof=1) / len(values))
            t = (values.mean() - base.mean()) / error
            cells.append(f"{values.mean():.6g} ({values.std(ddof=1):.1g}) t {t:+.2f}")
        print(f"{key:18}" + "".join(f"{cell:>30}" for cell in cells))


def main() -> None:
    found = batch_of_rl006()
    first_gradients(found)
    updates(found)


if __name__ == "__main__":
    main()
