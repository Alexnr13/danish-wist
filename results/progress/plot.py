"""Draw the workstation line's training progress as one page: results/progress/progress.html.

    python results/progress/plot.py

It reads the runs' logs in results/ (the in-run evaluation and the league's exploiters), the
exploiter margins of `learn.exploit` (results/x-*/margin.json) and the checkpoint sweep that
sweep.sh writes, and fills template.html with them. The x axis counts iterations from rl-004d's
10: rl-005 runs from 0 to 4000 and rl-006 from its 3600, so rl-006's iteration i is at 3600 + i.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent
RUNS = {"rl-005": 0, "rl-006": 3600}  # where each run starts on the x axis
START = "runs/rl-004d/checkpoints/policy-0010"  # rl-005's start, at 0
CHOSEN = {"rl-005": (2000, 3600), "rl-006": (2000,)}


def position(policy: str) -> tuple[str, float] | None:
    """The run and x of a checkpoint path, or None if it is not on the line."""
    if policy.startswith(START):
        return "rl-005", 0
    found = re.search(r"runs/(rl-00[56])/checkpoints/policy-(\d+)", policy)
    return (found[1], RUNS[found[1]] + int(found[2])) if found else None


def sweep(name: str) -> dict[str, tuple[float, float]]:
    """Each candidate's advantage and interval in an arena output, by path without `play:`."""
    path = HERE / name
    if not path.exists():
        return {}
    pattern = r"^(?:play:)?(\S+) against a field.*\nadvantage per deal: ([+-][\d.]+) ± ([\d.]+)"
    return {m[1]: (float(m[2]), float(m[3])) for m in re.finditer(pattern, path.read_text(), re.M)}


def smoothed(points: list[tuple[float, float]], half: int = 5) -> list[list[float]]:
    """A centred mean over 2 * half + 1 evaluations (100 iterations at one every 10)."""
    out = []
    for i, (x, _) in enumerate(points):
        window = [v for _, v in points[max(0, i - half) : i + half + 1]]
        out.append([x, round(sum(window) / len(window), 1)])
    return out


def data() -> dict:
    runs = {}
    for run, offset in RUNS.items():
        log = [json.loads(line) for line in (RESULTS / run / "log.jsonl").read_text().splitlines()]
        evals = [(offset + e["iteration"], e["vs_rulebot"]) for e in log if "vs_rulebot" in e]
        settings = json.loads((RESULTS / run / "run.json").read_text())["args"]
        runs[run] = {
            "start": offset,
            "exploit_iterations": settings["exploit_iterations"],
            "in_run": smoothed(evals),
            "league_exploiters": [
                [
                    offset + e["iteration"],
                    round(e["exploiter_margin"], 1),
                    round(e["exploiter_margin_ci95"], 1),
                ]
                for e in log
                if "exploiter_margin" in e
            ],
        }
    card, full = sweep("card-play.txt"), sweep("full-game.txt")
    checkpoints = []
    for policy in sorted(set(card) | set(full), key=lambda p: position(p) or ("", 0)):
        run, x = position(policy)
        own = 0 if x == 0 else x - RUNS[run]
        checkpoints.append(
            {
                "run": run,
                "x": x,
                "label": "rl-004d's 10" if x == 0 else f"{run}'s {own}",
                "chosen": own in CHOSEN.get(run, ()),
                "card": card.get(policy),
                "full": full.get(policy),
            }
        )
    margins = []
    for path in sorted(RESULTS.glob("x-*/margin.json")):
        margin = json.loads(path.read_text())
        spot = position(margin["policy"])
        if spot and margin["iterations"] == 100:  # the standard budget only
            seed = json.loads((path.parent / "run.json").read_text())["args"]["seed"]
            name = path.parent.name
            margins.append([spot[1], margin["margin"], margin["ci95"], name, seed])
    return {"runs": runs, "checkpoints": checkpoints, "exploiters": margins}


def main() -> None:
    page = (HERE / "template.html").read_text().replace("/*DATA*/null", json.dumps(data()))
    (HERE / "progress.html").write_text(page)
    print(HERE / "progress.html")


if __name__ == "__main__":
    main()
