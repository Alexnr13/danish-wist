"""A summary of a run's log past an iteration: health, window means and the in-run exploiters.

python results/rl-007/health.py runs/rl-007/log.jsonl 19000 1000
"""

import json
import math
import statistics as s
import sys

path, after, size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(line) for line in open(path)]
new = [r for r in rows if r["iteration"] > after]
wall = (new[-1]["time"] - new[0]["time"]) / max(1, len(new) - 1)
print("at", new[-1]["iteration"], "wall/it", round(wall, 3))
bad = [
    r["iteration"]
    for r in new
    if r.get("skipped_steps")
    or any(isinstance(v, float) and not math.isfinite(v) for v in r.values())
]
print("skipped or non-finite at", bad[:10])
for lo in range(after + 1, new[-1]["iteration"] + 1, size):
    w = [r for r in new if lo <= r["iteration"] < lo + size]
    ev = [r["vs_rulebot"] for r in w if "vs_rulebot" in r]

    def m(k: str, w: list = w) -> float:
        return s.mean(r[k] for r in w if k in r)

    share = f" lr {m('lr_share'):.2f}" if "lr_share" in w[0] else ""
    print(
        f"{lo}-{w[-1]['iteration']}: rule {s.mean(ev) if ev else float('nan'):6.1f} "
        f"ent {m('entropy'):.3f} ev {m('value_ev'):.3f} kl {m('approx_kl'):.5f} "
        f"clip {m('clip_fraction'):.4f} mkl {m('magnet_kl'):.4f} belief {m('belief_loss'):.3f} "
        f"level {m('level'):.2f} made {m('made'):.3f} "
        f"flip {s.mean(r['declared']['flip'] for r in w):.3f} "
        f"halves {s.mean(r['declared']['halves'] for r in w):.3f} "
        f"collect {m('collect_s'):.2f} update {m('update_s'):.2f}{share}"
    )
print(
    "exploiters",
    [
        (r["iteration"], round(r["exploiter_margin"], 1), round(r["exploiter_margin_ci95"], 1))
        for r in new
        if "exploiter_margin" in r
    ],
)
