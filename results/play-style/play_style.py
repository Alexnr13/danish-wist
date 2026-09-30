"""How a policy bids and what its hands are worth, from its self-play: results/play-style/.

    P=runs/rl-006/checkpoints/magnet-18000.npz
    python -m learn.arena --candidate $P --field $P --deals 20000 --record runs/arena/self.jsonl
    cd results/play-style && python play_style.py ../../runs/arena/self.jsonl > play-style.txt

Every lineup of a deal is the same policy in all four seats, and it plays greedily, so the five
games `learn.arena` records for a deal are one game: the first is kept. Deals thrown in for an
iron hand are left out; a deal all four passed counts, as a deal without a contract.

It prints the tables and fills template.html into play-style.html:

- the contracts that win the auction, by level and attachment: their share of the deals, how
  often they are made and the declarer's mean score (alone contracts pay the declarer triple);
- what each card and each extra card of length is worth: least-squares fits over all seats of
  the seat's score, and of its chance to win the auction, on its original 13 cards (Jokers,
  aces, kings, queens, jacks, the two longest suits) and its turn to speak, with 95% intervals
  clustered by deal (the four seats of a deal share one result);
- bidding and results by Jokers, aces, the longest suit and the shape of the two longest suits.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository

from danish_wist.cards import ACE, JACK, KING, QUEEN, Suit, parse_cards  # noqa: E402

HERE = Path(__file__).parent
ATTACHMENTS = ("plain", "clubs", "flip", "halves")
LEVELS = range(7, 14)
FEATURES = ["jokers", "aces", "kings", "queens", "jacks", "long", "second"]
NAMES = {
    "jokers": "a Joker",
    "aces": "an ace",
    "kings": "a king",
    "queens": "a queen",
    "jacks": "a jack",
    "long": "a card in the longest suit",
    "second": "a card in the second suit",
}


def deals(path):
    """One record per deal: the first of its lineups, all of which are the same game."""
    seen = set()
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            key = (record["meta"].get("seed"), record["meta"]["position"])
            if key in seen:
                continue
            seen.add(key)
            if len(set(record["meta"]["seats"])) != 1:
                raise SystemExit(f"{path} is not self-play: {record['meta']['seats']}")
            yield record


def seats(record, deal):
    """One row per seat of a deal that had an auction: its hand, its bids and its result."""
    outcome = record["outcome"]
    forehand = (record["dealer"] + 1) % 4
    highest = defaultdict(int)
    for seat, action in record["actions"]:
        if action.startswith("bid "):
            highest[seat] = max(highest[seat], int(action.split()[1]))
    for seat in range(4):
        cards = parse_cards(record["hands"][seat])
        lengths = sorted((sum(c.suit is s for c in cards) for s in Suit), reverse=True)
        declared = not outcome["redeal"] and outcome["declarer"] == seat
        yield {
            "deal": deal,
            "jokers": sum(c.is_joker for c in cards),
            "aces": sum(c.rank == ACE for c in cards),
            "kings": sum(c.rank == KING for c in cards),
            "queens": sum(c.rank == QUEEN for c in cards),
            "jacks": sum(c.rank == JACK for c in cards),
            "long": lengths[0],
            "second": lengths[1],
            "speaks": (seat - forehand) % 4 + 1,
            "bid": seat in highest,
            "declared": declared,
            "made": declared and made(outcome),
            "score": 0 if outcome["redeal"] else outcome["scores"][seat],
        }


def made(outcome):
    level = int(outcome["bid"].split()[0])
    declarer, partner = outcome["declarer"], outcome["partner"]
    taken = outcome["tricks_won"][declarer]
    if partner != declarer:
        taken += outcome["tricks_won"][partner]
    return taken >= level


def mean_ci(values):
    """Mean and the half-width of its 95% interval."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()) if len(values) else float("nan"), float("nan")
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def fit(rows, target, features=FEATURES, residuals=False):
    """Least squares of `target` on `features` and the turn to speak, clustered by deal.

    Each feature's coefficient and 95% interval; with `residuals`, each row's residual instead.
    """
    X = np.array(
        [[1.0] + [r[f] for f in features] + [r["speaks"] == k for k in (2, 3, 4)] for r in rows]
    )
    y = np.array([float(r[target]) for r in rows])
    groups = np.array([r["deal"] for r in rows])
    inverse = np.linalg.inv(X.T @ X)
    beta = inverse @ X.T @ y
    if residuals:
        return y - X @ beta
    scores = X * (y - X @ beta)[:, None]
    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    summed = np.add.reduceat(scores[order], starts)  # each deal's contribution
    clusters, n, k = len(starts), len(y), X.shape[1]
    meat = summed.T @ summed * clusters / (clusters - 1) * (n - 1) / (n - k)
    se = np.sqrt(np.diag(inverse @ meat @ inverse))
    names = [*features, "speaks second", "speaks third", "speaks fourth"]
    return {
        name: (float(b), float(1.96 * s))
        for name, b, s in zip(names, beta[1:], se[1:], strict=True)
    }


def by(rows, key, groups):
    """Bidding and results for each group of seats: (label, test) pairs over the key."""
    out = []
    for label, test in groups:
        group = [r for r in rows if test(r[key])]
        declared = [r for r in group if r["declared"]]
        out.append(
            {
                "label": label,
                "seats": len(group),
                "bid": sum(r["bid"] for r in group) / len(group),
                "declared": len(declared) / len(group),
                "made": sum(r["made"] for r in declared) / len(declared) if declared else None,
                "score": mean_ci([r["score"] for r in group]),
                "score_declaring": mean_ci([r["score"] for r in declared]),
            }
        )
    return out


def exactly(*values):
    """One group for each count."""
    return [(str(v), lambda x, v=v: x == v) for v in values]


def analyse(path):
    """Everything the page shows, from the recorded deals."""
    contracts = defaultdict(list)  # (level, attachment) -> [(made, declarer's score, alone)]
    rows, total, passed, iron, policy = [], 0, 0, 0, None
    for deal, record in enumerate(deals(path)):
        policy = record["meta"]["seats"][0]
        outcome = record["outcome"]
        if outcome["redeal"] and any(a == "iron-hand" for _, a in record["actions"]):
            iron += 1
            continue
        total += 1
        rows.extend(seats(record, deal))
        if outcome["redeal"]:
            passed += 1
            continue
        level, *attachment = outcome["bid"].split()
        key = (int(level), attachment[0] if attachment else "plain")
        alone = outcome["partner"] == outcome["declarer"]
        contracts[key].append((made(outcome), outcome["scores"][outcome["declarer"]], alone))

    grid = [
        {
            "level": level,
            "attachment": attachment,
            "deals": len(found),
            "share": len(found) / total,
            "made": sum(m for m, _, _ in found) / len(found),
            "score": mean_ci([s for _, s, _ in found]),
            "alone": sum(a for _, _, a in found) / len(found),
        }
        for (level, attachment), found in sorted(contracts.items())
    ]
    # What the shape adds to the score beyond the high cards: the residuals of a fit on them.
    beyond = fit(rows, "score", FEATURES[:5], residuals=True)
    shapes = defaultdict(list)  # (longest, second) -> rows, 8+ and 5+ pooled
    for row, residual in zip(rows, beyond, strict=True):
        row["beyond"] = residual
        shapes[min(row["long"], 8), min(row["second"], 5)].append(row)
    shape = [
        {
            "long": long,
            "second": second,
            "seats": len(group),
            "declared": sum(r["declared"] for r in group) / len(group),
            "score": mean_ci([r["score"] for r in group]),
            "beyond": mean_ci([r["beyond"] for r in group]),
            "jokers": sum(r["jokers"] for r in group) / len(group),
        }
        for (long, second), group in sorted(shapes.items())
        if long >= 4 and len(group) >= 50
    ]
    declarers = [r for r in rows if r["declared"]]
    share = defaultdict(float)
    for c in grid:
        share[c["attachment"]] += c["share"]
        share[c["level"]] += c["share"]
    longest = [("3–4", lambda v: v <= 4), *exactly(5, 6, 7), ("8+", lambda v: v >= 8)]
    return {
        "policy": policy,
        "deals": total,
        "iron": iron,
        "passed": passed,
        "seats": len(rows),
        "declarer_made": sum(r["made"] for r in declarers) / len(declarers),
        "contracts": grid,
        "attachments": {a: share[a] for a in ATTACHMENTS},
        "levels": {level: share[level] for level in LEVELS},
        "worth": {"score": fit(rows, "score"), "declared": fit(rows, "declared")},
        "jokers": by(rows, "jokers", exactly(0, 1, 2, 3)),
        "aces": by(rows, "aces", exactly(0, 1, 2, 3, 4)),
        "long": by(rows, "long", longest),
        "shape": shape,
    }


def report(data):
    """The page's numbers as text."""
    print(f"{data['policy']} in self-play: {data['deals']} deals", end=" ")
    print(f"({data['iron']} more thrown in for an iron hand), {data['passed']} passed out")
    print("\nContracts that won the auction: share, deals, made, declarer's mean score, alone")
    for c in data["contracts"]:
        if c["deals"] >= 20:
            score = f"{c['score'][0]:+7.0f} ± {c['score'][1]:4.0f}"
            print(
                f"  {c['level']:2d} {c['attachment']:6s} {c['share']:6.1%} {c['deals']:6d}", end=""
            )
            print(f"  made {c['made']:4.0%}  {score}  alone {c['alone']:4.0%}")
    print(f"  made in all: {data['declarer_made']:.1%}")
    print("  by attachment:", ", ".join(f"{a} {s:.1%}" for a, s in data["attachments"].items()))
    print("  by level:", ", ".join(f"{lv} {s:.1%}" for lv, s in data["levels"].items() if s))
    for target, title, scale in (("score", "points per deal", 1), ("declared", "percent", 100)):
        print(f"\nWhat one more of each is worth ({title}; 95% intervals clustered by deal)")
        for name, (b, ci) in data["worth"][target].items():
            print(f"  {NAMES.get(name, name):28s} {b * scale:+7.1f} ± {ci * scale:5.1f}")
    for key in ("jokers", "aces", "long"):
        print(f"\nBy {key}: seats, bid, won the auction, made as declarer, score, as declarer")
        for g in data[key]:
            rate = "   -" if g["made"] is None else f"{g['made']:4.0%}"
            print(
                f"  {g['label']:4s} {g['seats']:6d}  {g['bid']:4.0%}  {g['declared']:4.0%}", end=""
            )
            print(f"  {rate}  {g['score'][0]:+6.0f} ± {g['score'][1]:3.0f}", end="")
            print(f"  {g['score_declaring'][0]:+6.0f}")
    print(
        "\nBy the two longest suits: seats, Jokers, won the auction, score, beyond the high cards"
    )
    for s in data["shape"]:
        print(f"  {s['long']}-{s['second']}  {s['seats']:6d}  {s['jokers']:.1f}", end="")
        print(f"  {s['declared']:4.0%}  {s['score'][0]:+6.0f} ± {s['score'][1]:3.0f}", end="")
        print(f"  {s['beyond'][0]:+6.0f} ± {s['beyond'][1]:3.0f}")


def main():
    data = analyse(sys.argv[1])
    report(data)
    page = (HERE / "template.html").read_text().replace("/*DATA*/null", json.dumps(data))
    (HERE / "play-style.html").write_text(page)


if __name__ == "__main__":
    main()
