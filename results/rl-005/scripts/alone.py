"""Alone contracts: why an agent played 1 v 3 and what it cost, from recorded deals (REVIEW.md
T4.3).

A declarer plays alone (RULES.md §5) when it calls an ace it holds ("own ace") or the called ace
is in the cat ("ace in cat"); `learn.report` says which, and whether it "had a choice" of a
partner: some legal call's ace was not in its hand (any suit, but not clubs in a Clubs
contract). The call comes after the auction and before trumps, the Flip and the exchange, so no
declarer can know at the call whether that ace is in the cat. It learns it when the Flip turns
the ace up, when it is asked to name trumps in Halves because nobody else can (§6), or when it
picks up the cat (§8).

Each agent is set against the field (RuleBot) in the same seats of the same deals: with the
agent in seat s of deal p, the field's seat s in the all-field record of deal p, the duplicate
baseline that `learn.arena --record` writes.

usage: python results/rl-005/scripts/alone.py runs/arena/<policy>-vs-rule.jsonl...
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repository

from learn.report import seat_rows  # noqa: E402

CAT_CHANCE = 3 / 42  # an ace not in the declarer's hand: the other 42 cards go 13/13/13/3
KINDS = ("plain", "clubs", "flip", "halves")
ROLES = ("declarer", "alone", "partner", "defender", "redeal")
LEARNT = ("turned up in Flip", "named Halves trumps", "picked up the cat", "never")


def contract(record, row):
    """The declarer's contract: learn.report's row for its seat (level, attachment, made, score,
    and when alone the cause and whether it had a choice), with what it held, the tricks its side
    took, and when it learnt the called ace was in the cat, if it was."""
    outcome = record["outcome"]
    declarer, kind = outcome["declarer"], row["attachment"]
    hand, cat = record["hands"][declarer].split(), record["cat"].split()
    ace = "A" + outcome["called"]
    own = [text for seat, text in record["actions"] if seat == declarer]
    turned = 1 + own.count("flip next") if kind == "flip" else 0  # the cards turned face up
    if ace in cat[:turned]:
        learnt = LEARNT[0]
    elif kind == "halves":  # the declarer names trumps itself: nobody else can
        learnt = LEARNT[1]
    else:
        learnt = LEARNT[2] if row["exchanged"] else LEARNT[3]
    return row | {
        "cause": row.get("cause"),
        "aces": sum(card[0] == "A" for card in hand),
        "jokers": hand.count("JK"),
        "ace in hand": ace in hand,
        "learnt": learnt,
        "taken": sum(outcome["tricks_won"][s] for s in {declarer, outcome["partner"]}),
    }


def seat(record, s):
    """Seat s of a deal: its role, its score, and its contract when it declared."""
    row = seat_rows(record)[s]
    if row["redeal"]:
        return {"role": "redeal", "score": 0, "contract": None}
    own = contract(record, row) if row["role"] in ("declarer", "alone") else None
    return {"role": row["role"], "score": row["score"], "contract": own}


def side(name, seats):
    played = [s for s in seats if s["role"] != "redeal"]
    contracts = [s["contract"] for s in played if s["contract"]]
    return {"name": name, "played": played, "contracts": contracts}


def avg(values, spec="+6.0f"):
    """The mean in `spec`, or a dash as wide."""
    return format(mean(values), spec) if values else "-".rjust(len(format(0, spec)))


def ratio(n, total, spec="6.1%"):
    return format(n / total, spec) if total else "-".rjust(len(format(0, spec)))


def table(title, header, rows, sides, cells):
    """Each row's label, then `cells(side, group)` for each side side by side, where the group
    is that side's contracts passing the row's test. Rows no side has any contract in are left
    out."""
    print(f"\n{title}")
    print(" " * 20 + "  ".join(f"{s['name']:<{len(header)}}" for s in sides).rstrip())
    print(" " * 20 + "  ".join(header for _ in sides))
    for label, test in rows:
        groups = [[c for c in s["contracts"] if test(c)] for s in sides]
        if any(groups):
            line = "  ".join(cells(s, g) for s, g in zip(sides, groups, strict=True))
            print(f"{label:<20}{line}")


def by_cause(s, group):
    n, played = len(group), len(s["played"])
    scores = [c["score"] for c in group]
    return (
        f"{n:5d} {ratio(n, played)} {ratio(n, len(s['contracts']))} {avg(scores)}"
        f" {avg([c['made'] for c in group], '5.0%')} {sum(scores) / played:+8.1f}"
    )


def alone_rate(s, group):
    alone = [c["score"] for c in group if c["cause"]]
    other = [c["score"] for c in group if not c["cause"]]
    rate = ratio(len(alone), len(group))
    return f"{len(group):5d} {len(alone):5d} {rate} {avg(alone)} {avg(other)}"


def by_result(s, group):
    short = [c["level"] - c["taken"] for c in group if not c["made"]]
    return (
        f"{len(group):5d} {avg([c['score'] for c in group])}"
        f" {avg([c['taken'] for c in group], '6.2f')} {avg(short, '6.2f')}"
    )


def by_knowledge(s, group):
    in_cat = [c for c in s["contracts"] if c["cause"] == "ace in cat"]
    share, made = ratio(len(group), len(in_cat)), avg([c["made"] for c in group], "5.0%")
    return f"{len(group):5d} {share} {avg([c['score'] for c in group])} {made}"


def report(candidate, field, pairs):
    short = Path(candidate).stem if candidate.endswith((".npz", ".pt")) else candidate
    sides = [side(short, [a for a, _ in pairs]), side(field, [b for _, b in pairs])]
    print(f"== {candidate} against {field} in the same seats of the same deals")
    for s in sides:
        alone = [c for c in s["contracts"] if c["cause"]]
        print(
            f"{s['name']}: {len(pairs)} seats, {len(s['played'])} played, "
            f"{len(s['contracts'])} contracts, {len(alone)} alone"
        )

    is_own, is_cat = (lambda c: c["cause"] == "own ace"), (lambda c: c["cause"] == "ace in cat")
    table(
        "Alone contracts by cause (share of played seats and of contracts; points per played seat)",
        "count  seats contr.   mean  made per seat",
        [
            ("own ace", is_own),
            ("  with a choice", lambda c: is_own(c) and c["choice"]),
            ("  forced", lambda c: is_own(c) and not c["choice"]),
            ("ace in cat", is_cat),
            ("all alone", lambda c: c["cause"]),
            ("with a partner", lambda c: not c["cause"]),
        ],
        sides,
        by_cause,
    )

    header = "contr alone   rate  alone  other"
    note = "(mean scores of its alone and other contracts)"
    table(
        f"By the declarer's aces {note}",
        header,
        [(f"{n} ace" + "s" * (n != 1), lambda c, n=n: c["aces"] == n) for n in range(5)],
        sides,
        alone_rate,
    )
    table(
        "By its Jokers",
        header,
        [(f"{n} Joker" + "s" * (n != 1), lambda c, n=n: c["jokers"] == n) for n in range(4)],
        sides,
        alone_rate,
    )
    table(
        "By kind and level",
        header,
        [(k, lambda c, k=k: c["attachment"] == k) for k in KINDS]
        + [(f"level {n}", lambda c, n=n: c["level"] == n) for n in range(7, 14)],
        sides,
        alone_rate,
    )
    table(
        "By result (mean tricks taken by the declaring side; mean shortfall when failed)",
        "count   mean tricks  short",
        [
            ("alone, made", lambda c: c["cause"] and c["made"]),
            ("alone, failed", lambda c: c["cause"] and not c["made"]),
            ("partnered, made", lambda c: not c["cause"] and c["made"]),
            ("partnered, failed", lambda c: not c["cause"] and not c["made"]),
        ],
        sides,
        by_result,
    )
    table(
        "Ace in the cat: when the declarer learnt it was alone (nobody can know at the call)",
        "count  share   mean  made",
        [(w, lambda c, w=w: is_cat(c) and c["learnt"] == w) for w in LEARNT],
        sides,
        by_knowledge,
    )
    print(f"The called ace not in the declarer's hand: in the cat (chance {CAT_CHANCE:.1%})")
    for s in sides:
        elsewhere = [c for c in s["contracts"] if not c["ace in hand"]]
        n, k = len(elsewhere), sum(map(is_cat, elsewhere))
        ci = 1.96 * math.sqrt(k / n * (1 - k / n) / n) if n else math.nan
        print(f"  {s['name']:<18}{k:5d} of {n:5d} {ratio(k, n)} ± {ci:.1%}")

    groups = defaultdict(list)
    for a, b in pairs:
        role = f"alone, {a['contract']['cause']}" if a["role"] == "alone" else a["role"]
        groups[role].append((a, b))
        groups["all seats"].append((a, b))
    print(f"\nThe same seats in the all-{field} deals, by {short}'s role: its mean score (agent),")
    print(f"{field}'s there (field), the difference, the difference per seat of all {len(pairs)},")
    print(f"and {field}'s role there")
    print(f"{'seats':>26} {'agent':>7} {'field':>7}   diff  /seat  " + " ".join(ROLES))
    for label in ("alone, own ace", "alone, ace in cat", *ROLES[:1], *ROLES[2:], "all seats"):
        rows = groups.get(label)
        if not rows:
            continue
        mine, theirs = [a["score"] for a, _ in rows], [b["score"] for _, b in rows]
        diff = [x - y for x, y in zip(mine, theirs, strict=True)]
        roles = Counter(b["role"] for _, b in rows)
        print(
            f"{label:<20}{len(rows):6d} {avg(mine, '+7.0f')} {avg(theirs, '+7.0f')} "
            f"{avg(diff)} {sum(diff) / len(pairs):+6.1f}  "
            + " ".join(ratio(roles[r], len(rows), f"{len(r)}.0%") for r in ROLES)
        )
    print()


def deal(record):
    return record["meta"].get("seed"), record["meta"]["position"]


def main(paths):
    records = [json.loads(line) for path in paths for line in Path(path).read_text().splitlines()]
    baselines = {}
    for r in records:
        if len(set(r["meta"]["seats"])) == 1:
            baselines[deal(r), r["meta"]["seats"][0]] = r
    pairs = defaultdict(list)  # (candidate, field): (its seat, the field's seat)
    for r in records:
        names = Counter(r["meta"]["seats"]).most_common()
        if [n for _, n in names] != [3, 1] or (deal(r), names[0][0]) not in baselines:
            continue
        (field, _), (candidate, _) = names
        s = r["meta"]["seats"].index(candidate)
        pairs[candidate, field].append((seat(r, s), seat(baselines[deal(r), field], s)))
    for (candidate, field), found in sorted(pairs.items()):
        report(candidate, field, found)


if __name__ == "__main__":
    main(sys.argv[1:])
