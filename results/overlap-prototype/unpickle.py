"""What moving one iteration's trajectories between processes costs: 1024 self-play deals of
rl-006's policy at 18,000 collected as in training, then pickled and unpickled here (the
workers pickle them the same way to send them to the runner's process).

    python results/overlap-prototype/unpickle.py
"""

import gc
import json
import pickle
import random
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from learn import selfplay as sp  # noqa: E402
from learn.model import load  # noqa: E402
from learn.runner import Runner  # noqa: E402

STATE = ROOT / "runs/rl-006-state-18000"


def main() -> None:
    settings = sp.Settings(**json.loads((STATE / "settings.json").read_text()))
    policy = load(str(STATE / "policy.pt"))
    make = partial(
        sp.make_agents,
        slots=sp.SLOTS,
        explore=settings.explore_bids,
        explore_levels=settings.explore_levels,
    )
    served = sp.networks(policy.config, sp.SLOTS, "cuda")
    served.load(sp.LEARNER, policy.state_dict())
    rng = random.Random(1)
    with Runner(make, workers=22, games_in_flight=256, networks=served) as runner:
        for _ in range(2):
            found = sp.collect(runner, [[sp.LEARNER] * 4] * 1024, rng)
    data = pickle.dumps(found, pickle.HIGHEST_PROTOCOL)
    steps = sum(len(t.steps) for t in found)
    print(f"{steps} decisions, {len(data) / 1e6:.1f} MB pickled")
    for _ in range(3):
        gc.disable()
        started = time.perf_counter()
        pickle.loads(data)
        loads = time.perf_counter() - started
        gc.enable()
        started = time.perf_counter()
        pickle.dumps(found, pickle.HIGHEST_PROTOCOL)
        print(f"unpickled in {loads:.3f} s, pickled in {time.perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()
