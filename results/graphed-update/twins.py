"""Compare the twins of "Graphing the update" (TRAINING.md, Results): four copies of rl-006's
state at 18,000 resumed to 18,200 one after the other, two with the update launched from
Python as before (`--eager-update`: gu-eager, gu-eager2) and two with its steps replayed from
CUDA graphs (gu-graphed, gu-graphed2). Reads the `*-log.jsonl` beside this file.

    python results/graphed-update/twins.py > results/graphed-update/twins.txt

For each diagnostic: its mean over the 200 iterations with the standard error of that mean (as
if the iterations were independent; they are not, so it understates how far two runs can
drift apart), then the first iteration, whose batch is the same in all four, and the times.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TWINS = ["gu-eager", "gu-eager2", "gu-graphed", "gu-graphed2"]
KEYS = [
    "approx_kl",
    "clip_fraction",
    "entropy",
    "value_ev",
    "policy_loss",
    "value_loss",
    "magnet_kl",
    "belief_loss",
    "policy_grad_norm",
    "critic_grad_norm",
    "skipped_steps",
    "decisions",
    "mean_reward",
]


def rows(name: str) -> list[dict]:
    lines = (HERE / f"{name}-log.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if 18000 < r["iteration"] <= 18200]


def main() -> None:
    twins = {name: rows(name) for name in TWINS}
    print(f"{'':18}" + "".join(f"{name:>24}" for name in twins))
    for key in KEYS:
        cells = []
        for found in twins.values():
            values = np.array([r[key] for r in found], dtype=float)
            error = values.std(ddof=1) / np.sqrt(len(values))
            cells.append(f"{values.mean():.5g} ± {error:.2g}")
        print(f"{key:18}" + "".join(f"{cell:>24}" for cell in cells))
    print("\nthe first iteration, the same batch in all four:")
    for key in ["approx_kl", "clip_fraction", "entropy", "policy_loss", "value_loss"]:
        print(f"{key:18}" + "".join(f"{found[0][key]:>24.7g}" for found in twins.values()))
    print("\ntimes, s:")
    for name, found in twins.items():
        plain = found[:-1]  # 18,200 includes the exploiter phase
        wall = (found[-2]["time"] - found[0]["time"]) / (len(found) - 2)
        whole = (found[-1]["time"] - found[0]["time"]) / (len(found) - 1)
        evals = [r["vs_rulebot"] for r in found if "vs_rulebot" in r]
        last = found[-1]
        print(
            f"{name:12} collect_s {np.mean([r['collect_s'] for r in plain]):.3f}"
            f"  update_s {np.mean([r['update_s'] for r in plain]):.3f}"
            f"  eval_s {np.mean([r['eval_s'] for r in found if 'eval_s' in r]):.2f} (every 10th)"
            f"  an iteration {wall:.3f} (18,001 to 18,199), {whole:.3f} with the exploiter phase"
            f"  exploiter_s {last['exploiter_s']}  exploiter_margin {last['exploiter_margin']}"
            f"  vs_rulebot {np.mean(evals):.1f} (mean of {len(evals)})"
        )


if __name__ == "__main__":
    main()
