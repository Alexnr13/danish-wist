"""From a recorded arena: is the candidate's deficit as defender its card play or its auction?

Splits its defender seats by whether it bid at all and whether the contract is the one the
all-RuleBot baseline played, and shows the change in contract level.

usage: python results/review-2026-09/scripts/defender_split.py runs/arena/<policy>-vs-rule.jsonl
"""

import json
import sys
from collections import defaultdict
from statistics import mean


def level(outcome):
    return int(outcome["bid"].split()[0])


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
            if seat in (outcome["declarer"], outcome["partner"]):
                continue
            bid = any(s == seat and a.startswith("bid") for s, a in game["actions"])
            same = "same contract" if contract(outcome) == contract(baseline) else "changed"
            row = (
                outcome["scores"][seat] - baseline["scores"][seat],
                level(outcome) - level(baseline),
                outcome["scores"][seat],
                baseline["scores"][seat],
            )
            buckets[("bid" if bid else "passed", same)].append(row)
            buckets[("all", "")].append(row)
    print(f"{candidate} as defender against the baseline RuleBot in the same seat")
    print(
        f"{'candidate':>10} {'contract':>14} {'seats':>6} {'adv':>7} {'dLevel':>7} "
        f"{'cand':>7} {'base':>7}"
    )
    for (bid, same), rows in sorted(buckets.items()):
        columns = [mean(row[i] for row in rows) for i in range(4)]
        print(
            f"{bid:>10} {same:>14} {len(rows):6d} {columns[0]:+7.1f} {columns[1]:+7.2f} "
            f"{columns[2]:+7.1f} {columns[3]:+7.1f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
