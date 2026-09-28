"""Decompose a policy's duplicate advantage over RuleBot into bidding and card play.

usage: python results/review-2026-09/scripts/hybrid_arena.py <policy.npz> <deals> <workers> \
           [net net-auction net-prebid net-play]

Hybrid agents: one agent decides the auction, another the contract set-up (the call,
trumps, Flip, the exchange, discards, fucdic), a third the card play.

- `net`: the network throughout (the usual arena).
- `net-auction`: the network bids; RuleBot sets up the contract and plays the cards.
- `net-prebid`: the network bids and sets up the contract; RuleBot plays the cards.
- `net-play`: RuleBot bids and sets up; the network plays the cards. This is the
  **fixed-contract card-play test**: the stake never changes, so it resolves to about
  ±3.5 per deal over 2000 deals, against ±15 for the full arena.
"""

import random
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repository

from danish_wist.bots import RuleBot  # noqa: E402
from danish_wist.game import Phase  # noqa: E402
from learn.arena import random_positions  # noqa: E402
from learn.evaluate import evaluate  # noqa: E402
from learn.inference import NumpyAgent  # noqa: E402
from learn.runner import Runner  # noqa: E402

AUCTION = {Phase.IRON_HAND, Phase.AUCTION}


class Hybrid:
    def __init__(self, auction, contract, play):
        self.parts = {"auction": auction, "contract": contract, "play": play}

    @staticmethod
    def _part(view):
        if view.phase is Phase.PLAY:
            return "play"
        return "auction" if view.phase in AUCTION else "contract"

    def choose(self, view):
        return self.parts[self._part(view)].choose(view)

    def choose_batch(self, views):
        groups = defaultdict(list)
        for i, view in enumerate(views):
            groups[self._part(view)].append(i)
        out = [None] * len(views)
        for name, index in groups.items():
            agent = self.parts[name]
            sub = [views[i] for i in index]
            if hasattr(agent, "choose_batch"):
                actions = agent.choose_batch(sub)
            else:
                actions = [agent.choose(view) for view in sub]
            for i, action in zip(index, actions, strict=True):
                out[i] = action
        return out


def make_agents(worker, path):
    net, rule = NumpyAgent(path), RuleBot()
    return {
        "rule": rule,
        "net": net,
        "net-auction": Hybrid(net, rule, rule),
        "net-prebid": Hybrid(net, net, rule),
        "net-play": Hybrid(rule, rule, net),
    }


if __name__ == "__main__":
    path, deals, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    names = sys.argv[4:] or ["net", "net-auction", "net-prebid", "net-play"]
    positions = random_positions(deals, random.Random(0))
    with Runner(partial(make_agents, path=path), workers=workers, games_in_flight=64) as runner:
        for name in names:
            started = time.time()
            result = evaluate(runner, name, "rule", positions)
            print(f"== {name} vs rule, {deals} deals, {time.time() - started:.0f}s", flush=True)
            print(result.summary(), flush=True)
