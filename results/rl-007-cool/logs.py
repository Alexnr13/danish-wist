"""The logs of the cooldown check (TRAINING.md, Results, "The cooldown check"): rl-007, rl-006's
state at 18,000 carried on to 19,000 at its constant learning rates (the control), and
rl-007-cool, a copy of the same state resumed to 19,000 with `--cooldown 1000` (the branch: both
rates falling linearly to a tenth over 18,001 to 19,000), both with `--graphed-update`, run one
after the other on the workstation.

    python results/rl-007-cool/logs.py > results/rl-007-cool/logs.txt
    python results/rl-007-cool/logs.py runs/rl-007 runs/rl-007-cool     # while they run

Reads each arm's `log.jsonl` and `evals.jsonl` from 18,001 on (results/rl-007/ and this folder).
Window means are over 100 iterations; the in-run evaluations (every 10 iterations, the same 2000
deals each time) are paired deal by deal, between the arms and between evaluations of one arm.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ARMS = {"control": HERE.parent / "rl-007", "cooldown": HERE}
if len(sys.argv) == 3:
    ARMS = dict(zip(ARMS, map(Path, sys.argv[1:]), strict=True))
FIRST, LAST, WINDOW = 18001, 19000, 100
KEYS = [
    "lr_share",
    "approx_kl",
    "clip_fraction",
    "magnet_kl",
    "entropy",
    "policy_loss",
    "value_ev",
    "value_loss",
    "belief_loss",
    "policy_grad_norm",
    "critic_grad_norm",
    "decisions",
    "mean_reward",
    "level",
    "made",
]


def lines(path: Path) -> list[dict]:
    zipped = path.with_name(path.name + ".gz")
    text = path.read_text() if path.exists() else gzip.decompress(zipped.read_bytes()).decode()
    found = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [e for e in found if FIRST <= e["iteration"] <= LAST]


def paired(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    changes = after - before
    return changes.mean(), 1.96 * changes.std(ddof=1) / np.sqrt(len(changes))


def main() -> None:
    logs = {arm: lines(folder / "log.jsonl") for arm, folder in ARMS.items()}
    evals = {arm: lines(folder / "evals.jsonl") for arm, folder in ARMS.items()}
    for arm, found in logs.items():
        iterations = [e["iteration"] for e in found]
        print(f"{arm}: iterations {iterations[0]} to {iterations[-1]} ({len(found)} logged)")
    starts = range(FIRST, LAST + 1, WINDOW)

    print("\n== window means (the cooldown's over the control's in brackets)")
    for key in KEYS:
        print(f"\n{key}")
        print(f"{'iterations':>14} {'control':>12} {'cooldown':>12}")
        for start in starts:
            cells = []
            for arm in ARMS:
                window = [e for e in logs[arm] if start <= e["iteration"] < start + WINDOW]
                values = [e.get(key, 1.0 if key == "lr_share" else None) for e in window]
                values = [v for v in values if v is not None]
                cells.append(np.mean(values) if values else np.nan)
            ratio = f"({cells[1] / cells[0]:.3f})" if cells[0] and np.isfinite(cells[0]) else ""
            end = start + WINDOW - 1
            print(f"{start:6d}-{end:6d} {cells[0]:12.5g} {cells[1]:12.5g}  {ratio}")

    print("\n== health")
    for arm, found in logs.items():
        bad = [
            (e["iteration"], k)
            for e in found
            for k, v in e.items()
            if isinstance(v, float) and not np.isfinite(v)
        ]
        skipped = sum(e.get("skipped_steps", 0) for e in found)
        print(f"{arm}: non-finite values {bad or 'none'}, skipped steps {skipped}")
        largest = [f"{k} {max(e[k] for e in found):.4g}" for k in ("approx_kl", "magnet_kl")]
        print("  largest " + ", ".join(largest))

    print("\n== in-run exploiters (100 iterations each, every 200)")
    for arm, found in logs.items():
        margins = [
            f"{e['iteration']}: {e['exploiter_margin']:+.1f} ± {e['exploiter_margin_ci95']:.1f}"
            for e in found
            if "exploiter_margin" in e
        ]
        print(f"{arm}: " + ", ".join(margins))

    print("\n== in-run evaluations against RuleBot (the policy; 2000 deals, the same each time)")
    by_iteration = {
        arm: {e["iteration"]: np.array(e["per_deal"]) for e in evals[arm]} for arm in ARMS
    }
    common = sorted(set(by_iteration["control"]) & set(by_iteration["cooldown"]))
    print(f"{'iterations':>14} {'control':>9} {'cooldown':>9}  the cooldown - the control, paired")
    for start in starts:
        its = [i for i in common if start <= i < start + WINDOW]
        if not its:
            continue
        means = [np.mean([by_iteration[arm][i].mean() for i in its]) for arm in ARMS]
        pooled = [np.concatenate([by_iteration[arm][i] for i in its]) for arm in ARMS]
        change, half = paired(*pooled)
        print(
            f"{start:6d}-{start + WINDOW - 1:6d} {means[0]:+9.1f} {means[1]:+9.1f}"
            f"  {change:+.1f} ± {half:.1f} (the {len(its)} evaluations pooled)"
        )
    if LAST in common:
        change, half = paired(by_iteration["control"][LAST], by_iteration["cooldown"][LAST])
        print(f"at {LAST}: {change:+.1f} ± {half:.1f}")

    print("\n== how far the policy moves between evaluations 10 iterations apart, against RuleBot")
    print("(the sd of the paired change, less the part measuring explains: TRAINING.md, 'The")
    print("recipe for a long run', point 1; rl-006 over 6000 iterations: 16.0)")
    for first, last in ((18010, 18500), (18510, 18700), (18710, 19000)):
        cells = []
        for arm in ARMS:
            its = [i for i in sorted(by_iteration[arm]) if first <= i <= last]
            per_deal = np.array([by_iteration[arm][i] for i in its])
            changes = per_deal[1:] - per_deal[:-1]
            measuring = (changes.var(axis=1, ddof=1) / changes.shape[1]).mean()
            moved = max(changes.mean(1).var() - measuring, 0.0)
            cells.append(f"{arm} {np.sqrt(moved):5.1f} (of {np.sqrt(changes.mean(1).var()):5.1f})")
        print(f"{first:6d}-{last:6d}: " + ", ".join(cells))

    print("\n== time")
    for arm, found in logs.items():
        run = json.loads((ARMS[arm] / "run.json").read_text())
        started = run["resumed"][-1]["time"]
        times = np.array([e["time"] for e in found])
        plain = [e for e in found if "exploiter_s" not in e]
        exploiters = [e["exploiter_s"] for e in found if "exploiter_s" in e]
        print(
            f"{arm}: resumed {started}, {len(found)} iterations; from the first to the last "
            f"{(times[-1] - times[0]) / (len(times) - 1):.3f} s an iteration; collecting "
            f"{np.mean([e['collect_s'] for e in plain]):.3f} s, updating "
            f"{np.mean([e['update_s'] for e in plain]):.3f} s (means of values rounded to 0.1 s, "
            f"without the exploiter phases), exploiter phases {np.mean(exploiters):.1f} s "
            f"({len(exploiters)}), graphed: {run['resumed'][-1].get('graphed_update', False)}"
        )


if __name__ == "__main__":
    main()
