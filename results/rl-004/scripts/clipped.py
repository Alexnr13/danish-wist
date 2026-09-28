"""How many recorded auction decisions start outside PPO's clip range when the ratio is
taken against the explored odds (the pre-fix loss), and which way.

usage: python results/rl-004/scripts/clipped.py <policy.pt> [deals]
"""

import random
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repository

import torch  # noqa: E402

from learn.model import load  # noqa: E402
from learn.runner import Runner  # noqa: E402
from learn.selfplay import LEARNER, collect, make_agents, stake_is_fixed  # noqa: E402

if __name__ == "__main__":
    net = load(sys.argv[1])
    deals = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    make = partial(make_agents, config=asdict(net.config), snapshots=1, one_thread=True)
    agents = partial(make, explore=0.15, explore_levels=0.1)
    with Runner(agents, workers=6, games_in_flight=64) as runner:
        runner.broadcast(LEARNER, "load", net.state_dict())
        steps = [
            s for t in collect(runner, [[LEARNER] * 4] * deals, random.Random(5)) for s in t.steps
        ]
    auction = [s for s in steps if int(s.observation.tokens[0][3]) == 1]  # phase 1: the auction
    r = torch.tensor([s.policy_log_prob - s.log_prob for s in auction]).exp()
    above, below = (r > 1.2).float().mean(), (r < 0.8).float().mean()
    print(
        f"{len(steps)} decisions, {len(auction)} in the auction ({len(auction) / len(steps):.0%})"
    )
    print(f"  own/played odds: above 1.2 {above:.1%} (favourites: good results ignored)")
    print(f"                   below 0.8 {below:.1%} (explored: bad results ignored)")
    print(f"                   inside    {1 - above - below:.1%}")
    print(f"  median {r.median():.2f}, min {r.min():.3f}, max {r.max():.2f}")
    fixed = sum(stake_is_fixed(s.observation) for s in steps)
    print(f"  (decisions from the exchange on: {fixed})")
