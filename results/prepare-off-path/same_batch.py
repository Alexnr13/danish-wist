"""`prepare` and the proximal pass off the main thread (TRAINING.md, Results): on rl-006's real
networks at 18,000 and one real batch (1024 deals from its league, as the run draws them), does
the new `prepare` give what it gave before, and the graphed proximal pass what the compiled one
gives?

- The batch from `prepare` without graphs (a default run's) and with them (`--graphed-update`),
  field by field against `prepare` as it was (`plain_prepare` in tests/test_selfplay.py: step by
  step in Python, the critic's pass compiled).
- The proximal pass (`timing.proximal`: the policy's log-chance of each move, as collecting
  while updating would recompute it), replayed from graphs against compiled; and, for scale,
  against the log-chances the served networks recorded while playing.
- One whole graphed update (about 72 minibatches) from the same state, six times on each
  batch: `prepare`'s as it was, and the new one's with graphs. The update is not repeatable run
  to run (the GPU's sums are in no fixed order), so the two are compared against that spread.

    python results/prepare-off-path/same_batch.py > results/prepare-off-path/same_batch.txt
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/graphed-update"))
sys.path.insert(0, str(ROOT / "results/overlap-prototype"))
sys.path.insert(0, str(ROOT / "tests"))

import same_batch as earlier  # noqa: E402  (batch_of_rl006)
import timing  # noqa: E402  (restore, proximal)
from test_selfplay import plain_prepare  # noqa: E402

from learn import selfplay as sp  # noqa: E402

REPEATS = 6
KEYS = earlier.KEYS


def fields(found: list) -> None:
    settings, learning, _, _ = timing.restore("cuda", True)
    plain = plain_prepare(found, learning.critic, settings)
    print(f"{len(found)} trajectories, {len(plain.actions)} decisions")
    for label, graphs in (("without graphs", None), ("with graphs", learning.graphs)):
        batch = sp.prepare(found, learning.critic, settings, graphs)
        same = []
        for name in ("legal", "beliefs", "actions", "old_log_probs", "played_log_probs"):
            same.append((name, torch.equal(getattr(batch, name), getattr(plain, name))))
        for name in ("inputs", "oracle"):
            a, b = getattr(batch, name), getattr(plain, name)
            same.append(
                (name, torch.equal(a.tokens, b.tokens) and torch.equal(a.lengths, b.lengths))
            )
        same.append(("returns", torch.equal(batch.returns, plain.returns)))
        gap = (batch.advantages - plain.advantages).abs()
        print(
            f"prepare {label}: "
            + ", ".join(f"{n} {'equal' if s else 'NOT EQUAL'}" for n, s in same)
            + f"; advantages {'equal' if gap.max() == 0 else 'apart'}"
            f" (largest gap {gap.max().item():.1e}, bit for bit in {(gap == 0).float().mean():.1%})"
            f"; value_ev {batch.value_ev!r} against {plain.value_ev!r}"
        )
    compiled = timing.proximal(learning.policy, plain, settings.chunk)
    graphed = timing.proximal(learning.policy, plain, settings.chunk, learning.graphs)
    served = (compiled - plain.old_log_probs).abs()
    print(
        f"proximal pass: graphed and compiled {'apart' if (graphed - compiled).any() else 'equal'}"
        f" (largest gap {(graphed - compiled).abs().max().item():.1e}); for scale, the served"
        f" networks' recorded log-chances are {served.mean().item():.1e} from them on average"
        f" (largest {served.max().item():.1e})"
    )


def updates(found: list) -> None:
    modes = ["before", "after"]
    results: dict[str, list[dict]] = {mode: [] for mode in modes}
    for mode in modes * REPEATS:
        settings, learning, _, _ = timing.restore("cuda", True)
        if mode == "before":
            batch = plain_prepare(found, learning.critic, settings)
        else:
            batch = sp.prepare(found, learning.critic, settings, learning.graphs)
        nets = (learning.policy, learning.critic, learning.magnet, learning.optimisers)
        stats = sp.update(*nets, batch, settings, random.Random(7), graphs=learning.graphs)
        results[mode].append(stats)
    print(f"\none graphed update, {REPEATS} times on each batch: mean (sd), and t against before")
    print(f"{'':18}" + "".join(f"{mode:>30}" for mode in modes))
    for key in KEYS:
        base = np.array([stats[key] for stats in results["before"]])
        cells = []
        for mode in modes:
            values = np.array([stats[key] for stats in results[mode]])
            error = np.sqrt(base.var(ddof=1) / len(base) + values.var(ddof=1) / len(values))
            t = (values.mean() - base.mean()) / error
            cells.append(f"{values.mean():.6g} ({values.std(ddof=1):.1g}) t {t:+.2f}")
        print(f"{key:18}" + "".join(f"{cell:>30}" for cell in cells))


def main() -> None:
    found = earlier.batch_of_rl006()
    fields(found)
    updates(found)


if __name__ == "__main__":
    main()
