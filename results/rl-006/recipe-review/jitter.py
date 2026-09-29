"""How far does a run's score against RuleBot move between evaluations some iterations apart,
beyond what measuring it on the same deals explains? (TRAINING.md, "The recipe for a long run")

    python results/rl-006/recipe-review/jitter.py runs/rl-006/evals.jsonl

Every in-run evaluation plays the same deals, so the change between two is paired deal by deal.
Its spread over all pairs the same distance apart, less the measuring noise of a paired change,
is how much the policy itself moved.
"""

import json
import sys

import numpy as np

evals = [json.loads(line) for line in open(sys.argv[1])]
iterations = np.array([e["iteration"] for e in evals])
per_deal = np.array([e["per_deal"] for e in evals])  # evaluations x deals
means = per_deal.mean(1)
step = iterations[1] - iterations[0]
for gap in (1, 2, 5, 10, 20, 50, 100, 200):
    changes = means[gap:] - means[:-gap]
    paired = per_deal[gap:] - per_deal[:-gap]
    measuring = (paired.var(axis=1, ddof=1) / paired.shape[1]).mean()
    policy = max(changes.var() - measuring, 0.0)
    print(
        f"{gap * step:5d} iterations apart: sd of the change {np.sqrt(changes.var()):5.1f}, "
        f"of measuring it {np.sqrt(measuring):5.1f}, of the policy {np.sqrt(policy):5.1f}"
    )
for start in range(0, len(evals), 50):
    window = means[start : start + 50]
    last = iterations[min(start + 49, len(evals) - 1)]
    print(f"{iterations[start]:5d}-{last:5d}: mean {window.mean():6.1f}, sd {window.std():5.1f}")
