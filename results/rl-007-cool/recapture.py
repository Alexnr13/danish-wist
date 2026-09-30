"""What a cooldown costs a graphed update (TRAINING.md, Results, "The cooldown check"): the graph of
the clip and AdamW keeps the learning rate it was captured with, so with `--cooldown` it is
captured again at every iteration (`_GraphedStep._stepping`), once for each network.

On rl-006's networks and optimisers at 18,000 (restored as `results/overlap-prototype/timing.py`
does, read only) and one real batch of 1024 deals, time whole graphed updates (`prepare` and
`update`) in turns: at a constant rate, and at a rate changed before each (as a cooldown does).

    python results/rl-007-cool/recapture.py > results/rl-007-cool/recapture.txt
"""

import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/overlap-prototype"))

import timing  # noqa: E402

from learn import selfplay as sp  # noqa: E402
from learn.runner import Runner  # noqa: E402

TURNS = 15


def main() -> None:
    settings, learning, league, rng = timing.restore("cuda", True)
    make = partial(
        sp.make_agents,
        slots=sp.SLOTS,
        explore=settings.explore_bids,
        explore_levels=settings.explore_levels,
    )
    served = sp.networks(learning.policy.config, sp.SLOTS, "cuda")
    with Runner(make, workers=timing.WORKERS, games_in_flight=256, networks=served) as runner:
        runner.networks.load(sp.LEARNER, learning.policy.state_dict())
        lineups, _ = sp._league_lineups(runner, league, {}, settings, rng)
        found = sp.collect(runner, lineups, rng)
    nets = (learning.policy, learning.critic, learning.magnet, learning.optimisers)
    captures = []
    stepping = sp._GraphedStep._stepping

    def counted(self):
        before = self.stepping
        graph = stepping(self)
        captures.append(self.stepping is not before)
        return graph

    sp._GraphedStep._stepping = counted
    times: dict[str, list[float]] = {"constant": [], "changed": []}
    shares = iter(np.linspace(1.0, 0.1, 2 * TURNS + 3))
    for turn in range(-1, TURNS):  # the first turn warms up (compiles and captures)
        for mode in times:
            share = next(shares) if mode == "changed" or turn < 0 else None
            if share is not None:
                learning.set_rates(settings, float(share))
            captures.clear()
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch = sp.prepare(found, learning.critic, settings, learning.graphs)
            sp.update(*nets, batch, settings, random.Random(turn), graphs=learning.graphs)
            torch.cuda.synchronize()
            if turn >= 0:
                times[mode].append(time.perf_counter() - started)
                print(f"turn {turn:2d} {mode:8s} {times[mode][-1]:.4f} s, captures {sum(captures)}")
    for mode, found_times in times.items():
        values = np.array(found_times)
        error = values.std(ddof=1) / np.sqrt(len(values))
        print(f"{mode}: {values.mean():.4f} ± {error:.4f} s (median {np.median(values):.4f})")
    difference = 1000 * (np.array(times["changed"]) - np.array(times["constant"]))
    error = difference.std(ddof=1) / np.sqrt(len(difference))
    print(f"changed - constant, paired by turn: {difference.mean():+.1f} ± {error:.1f} ms")


if __name__ == "__main__":
    main()
