"""A run's evaluation curve, with paired changes (TRAINING.md §5).

    python -m learn.curve runs/rl-003            # the curve so far
    python -m learn.curve runs/rl-003 --follow   # then each new evaluation, and alerts
    python -m learn.curve results/rl-006         # a run's copy, its evals.jsonl gzipped or not

Every evaluation in a run plays the same deals, so the change between two is
a paired difference, per deal, with a much tighter confidence interval than
either result's own. A fall is significant when its whole interval is below
0; three in a row is TRAINING.md's stop rule. Each line also shows the
results by role and the kinds of contract the learner declared.

`--follow` keeps watching: new evaluations, any non-finite value or skipped
step, a log that stops changing, and the run reaching its last iteration.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path


def paired(before: list[float], after: list[float]) -> tuple[float, float]:
    """Mean change per deal and the half-width of its 95% confidence interval."""
    changes = [b - a for a, b in zip(before, after, strict=True)]
    mean = sum(changes) / len(changes)
    variance = sum((c - mean) ** 2 for c in changes) / (len(changes) - 1)
    return mean, 1.96 * math.sqrt(variance / len(changes))


def _lines(path: Path) -> list[dict]:
    """A JSON-lines file, or its gzipped copy (`name.gz`, as a long run's is in results/)."""
    zipped = path.with_name(path.name + ".gz")
    if path.exists():
        text = path.read_text()
    elif zipped.exists():
        text = gzip.decompress(zipped.read_bytes()).decode()
    else:
        return []
    found = []
    for line in text.splitlines():
        try:
            found.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line being written
    return found


def describe(entry: dict, change: tuple[float, float] | None, falls: int) -> str:
    name = "rulebot" if "vs_rulebot" in entry else "target"
    text = f"{entry['iteration']:4d}  {entry[f'vs_{name}']:+7.1f} ± {entry[f'vs_{name}_ci95']:4.1f}"
    text += f"  {f'{change[0]:+.1f} ± {change[1]:.1f}':>13}" if change else f"{'':15}"
    text += f"  {falls}" if change else "   "
    roles = entry.get(f"vs_{name}_roles", {})
    text += "  " + " ".join(f"{r[:4]} {v:+.0f}" for r, v in roles.items() if r != "redeal")
    if declared := entry.get("declared"):
        text += "  |  " + " ".join(f"{k} {v:.0%}" for k, v in declared.items())
        text += f"  level {entry.get('level')}  made {entry.get('made', 0):.0%}"
    return text


def curve(run: Path) -> list[str]:
    """One line per evaluation: result, paired change from the one before, falls in a row."""
    log = {e["iteration"]: e for e in _lines(run / "log.jsonl")}
    evals = _lines(run / "evals.jsonl")
    lines = ["iter  result          paired change  falls  by role  |  contracts declared"]
    falls, before = 0, None
    for evaluation in evals:
        change = None
        if before is not None:
            change = paired(before["per_deal"], evaluation["per_deal"])
            falls = falls + 1 if change[0] + change[1] < 0 else 0
        if evaluation["iteration"] in log:
            lines.append(describe(log[evaluation["iteration"]], change, falls))
        before = evaluation
    return lines


def follow(run: Path, stale_after: float = 600) -> None:
    """Print new evaluations and alerts until the run reaches its last iteration."""
    total = json.loads((run / "run.json").read_text())["args"]["iterations"]
    shown, seen, stale = len(curve(run)), len(_lines(run / "log.jsonl")), False
    while True:
        entries = _lines(run / "log.jsonl")
        for entry in entries[seen:]:
            bad = [k for k, v in entry.items() if isinstance(v, float) and not math.isfinite(v)]
            if bad or entry.get("skipped_steps"):
                print(
                    f"ALERT iteration {entry['iteration']}: non-finite {bad}, "
                    f"skipped steps {entry.get('skipped_steps')}",
                    flush=True,
                )
        seen = len(entries)
        lines = curve(run)
        for line in lines[shown:]:
            print(line, flush=True)
        shown = len(lines)
        if entries and entries[-1]["iteration"] >= total:
            print(f"the run reached its last iteration, {total}", flush=True)
            return
        age = time.time() - (run / "log.jsonl").stat().st_mtime if entries else 0
        if age > stale_after and not stale:
            print(f"STALL: the log has not changed for {age:.0f} s", flush=True)
        stale = age > stale_after
        time.sleep(20)


def main() -> None:
    parser = argparse.ArgumentParser(description="A run's evaluation curve, with paired changes.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--follow", action="store_true", help="keep watching the run")
    args = parser.parse_args()
    print("\n".join(curve(args.run)), flush=True)
    if args.follow:
        follow(args.run)


if __name__ == "__main__":
    main()
