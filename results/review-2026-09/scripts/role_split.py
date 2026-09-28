"""From a recorded arena (`learn.arena --record`): the candidate's advantage over RuleBot in
each role, split by whether its bidding changed the contract from the all-RuleBot baseline.

With the same contract, the difference is the candidate's card play (and contract set-up)
alone. With a changed contract, it includes the auction's effect on the stake.

usage: python results/review-2026-09/scripts/role_split.py runs/arena/<policy>-vs-rule.jsonl
"""

import json
import math
import sys
from collections import defaultdict
from statistics import mean, stdev


def role(outcome, seat):
    if seat == outcome["declarer"]:
        return "alone" if outcome["partner"] == seat else "declarer"
    return "partner" if seat == outcome["partner"] else "defender"


def contract(outcome):
    return (outcome["declarer"], outcome["bid"], outcome["called"], outcome["trumps"])


def main(path):
    records = [json.loads(line) for line in open(path)]
    by_position = defaultdict(list)
    for record in records:
        by_position[record["meta"]["position"]].append(record)
    candidate = next(n for n in records[1]["meta"]["seats"] if n != "rule")
    buckets = defaultdict(list)
    for games in by_position.values():
        baseline = games[0]["outcome"]
        for game in games[1:]:
            seat = game["meta"]["seats"].index(candidate)
            outcome = game["outcome"]
            if outcome["redeal"] or baseline["redeal"]:
                continue
            advantage = outcome["scores"][seat] - baseline["scores"][seat]
            same = "same contract" if contract(outcome) == contract(baseline) else "changed"
            for r in (role(outcome, seat), "all"):
                buckets[(r, same)].append(advantage)
                buckets[(r, "all")].append(advantage)
    print(f"{candidate}: advantage over RuleBot in the same seat and cards (points per seat)")
    print(f"{'role':>9} {'contract':>14} {'seats':>6} {'advantage (95% CI)':>20}")
    for (r, same), values in sorted(buckets.items()):
        ci = 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.nan
        print(f"{r:>9} {same:>14} {len(values):6d} {mean(values):+9.1f} ± {ci:5.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
