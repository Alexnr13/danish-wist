"""Policies against a RuleBot field on the same fresh deals from several seeds, pooled,
each paired against the first.

usage: python results/rl-004/scripts/screen.py <policy> <policy> ... --seeds 21 23

`--seeds 12345` plays the in-run evaluation's deals (`learn.arena --seed 12345` does too).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repository

from learn.curve import paired  # noqa: E402
from learn.evaluate import run  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("policies", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=[21, 23])
    p.add_argument("--deals", type=int, default=2000)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    first = None
    print(f"against RuleBot, {a.deals} deals from each of seeds {a.seeds}, pooled")
    for policy in a.policies:
        per_deal, roles = [], {}
        for seed in a.seeds:
            r = run(policy, "rule", a.deals, seed=seed, workers=a.workers)
            per_deal += r.per_deal
            for k, v in r.by_role.items():
                roles.setdefault(k, []).extend(v)
        n = len(per_deal)
        mean = sum(per_deal) / n
        sd = (sum((x - mean) ** 2 for x in per_deal) / (n - 1)) ** 0.5
        text = f"{policy:48s} {mean:+6.1f} ± {1.96 * sd / n**0.5:4.1f}  "
        text += " ".join(
            f"{k[:4]} {sum(v) / len(v):+.0f}" for k, v in roles.items() if v and k != "redeal"
        )
        if first is None:
            first = per_deal
        else:
            m, h = paired(first, per_deal)
            text += f"   minus the first: {m:+6.1f} ± {h:4.1f}"
        print(text, flush=True)
