"""How much each agent's bidding reads its hand, from recorded deals (REVIEW.md T4.2).

For every agent in `meta.seats` other than RuleBot, pooling its seats: how bidding at all, the
level bid and the contracts that followed depend on its original 13 cards (before any exchange).
Where the same deal was also played by four RuleBots (the all-"rule" lineup `learn.arena
--record` writes), the agent is also set against RuleBot in the same seats and deals.

usage: python results/rl-005/scripts/hand_reading.py runs/arena/<policy>-vs-rule.jsonl...
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import StatisticsError, correlation, linear_regression, mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repository

from danish_wist.cards import ACE, JACK, KING, QUEEN, Suit, parse_cards  # noqa: E402

BASELINE = "rule"
POINTS = {ACE: 4, KING: 3, QUEEN: 2, JACK: 1}  # and 4 for a Joker
ATTACHMENTS = ("plain", "flip", "clubs", "halves")
NOTES = """\
Only seats that spoke in the auction count (not those of a deal redealt for an iron hand).
Hand features come from the original 13 cards, before any exchange. HCP: high-card points,
A=4 K=3 Q=2 J=1 Joker=4 (52 in the deck, 12.3 a hand on average). Top cards: aces + Jokers.
Long suit: cards in the longest suit. Highest bid: the level of the seat's highest bid, over
the seats that bid.
Contracts, made and mean score: the seat as declarer (alone included), with its own score.
Alone: those contracts it played alone (RULES.md §5), and their mean score; alone.py splits
them by cause.
'rule, same seats': RuleBot with the same cards in the same seat, in the deal's all-rule lineup.
"""


def ranges(*starts):
    """Labelled tests for the ranges from each start to the next, the last one open-ended."""
    ends = [start - 1 for start in starts[1:]] + [None]
    return [
        (
            f"{lo}+" if hi is None else f"{lo}" if lo == hi else f"{lo}-{hi}",
            lambda v, lo=lo, hi=hi: lo <= v and (hi is None or v <= hi),
        )
        for lo, hi in zip(starts, ends, strict=True)
    ]


HCP_BANDS = ranges(0, 7, 10, 13, 16, 19)
FEATURES = [  # table title, row field, column header, groups
    ("high-card points", "hcp", "HCP", HCP_BANDS),
    ("top cards (aces + Jokers)", "top", "top", ranges(0, 1, 2, 3, 4, 5)),
    ("longest suit", "long", "long", ranges(3, 5, 6, 7)),
]


def seat_rows(record):
    """One row per seat that spoke in the auction: its hand, its bids, and its contract if any."""
    outcome = record["outcome"]
    forehand = (record["dealer"] + 1) % 4
    heard, levels, anyone_bid = {}, defaultdict(list), False
    for seat, text in record["actions"]:
        if text == "pass" or text.startswith("bid "):
            heard.setdefault(seat, anyone_bid)  # had anyone bid before this seat first spoke?
            if text != "pass":
                levels[seat].append(int(text.split()[1]))
                anyone_bid = True
    rows = []
    for seat, earlier_bid in heard.items():
        cards = parse_cards(record["hands"][seat])
        row = {
            "agent": record["meta"]["seats"][seat],
            "place": (record["dealer"], *record["hands"], record["cat"], seat),
            "hcp": sum(4 if c.is_joker else POINTS.get(c.rank, 0) for c in cards),
            "top": sum(c.is_joker or c.rank == ACE for c in cards),
            "long": max(sum(c.suit is suit for c in cards) for suit in Suit),
            "speaks": (seat - forehand) % 4 + 1,
            "earlier_bid": earlier_bid,
            "bid": bool(levels[seat]),
            "high": max(levels[seat], default=None),
            "score": 0 if outcome["redeal"] else outcome["scores"][seat],
            "declared": not outcome["redeal"] and outcome["declarer"] == seat,
        }
        if row["declared"]:
            row |= contract(outcome, seat, cards)
        rows.append(row)
    return rows


def contract(outcome, declarer, cards):
    level, *attachment = outcome["bid"].split()
    partner = outcome["partner"]
    taken = outcome["tricks_won"][declarer]
    if partner != declarer:
        taken += outcome["tricks_won"][partner]
    trumps = Suit(outcome["trumps"]) if outcome["trumps"] else None  # None: no trumps
    return {
        "level": int(level),
        "attachment": attachment[0] if attachment else "plain",
        "made": taken >= int(level),
        "trumps_held": None if trumps is None else sum(c.suit is trumps for c in cards),
        "alone": partner == declarer,
    }


def r(rows, x, y):
    """Pearson correlation of two row fields, or nan where it is undefined."""
    try:
        return correlation([row[x] for row in rows], [float(row[y]) for row in rows])
    except StatisticsError:  # fewer than two rows, or a constant
        return math.nan


def slope(rows):
    """Change in the highest bid per 10 HCP (least squares)."""
    try:
        fit = linear_regression([row["hcp"] for row in rows], [row["high"] for row in rows])
    except StatisticsError:
        return math.nan
    return 10 * fit.slope


def bidders(rows):
    return [row for row in rows if row["bid"]]


def pct(part, whole):
    return f"{part / whole:6.1%}" if whole else "     -"


def avg(values, spec, width):
    return format(mean(values), spec).rjust(width) if values else "-".rjust(width)


def corr(x, y, among=lambda rows: rows):
    """A summary value: r(x, y) over a group's seats, or over those `among` picks from them."""
    return lambda rows: f"{r(among(rows), x, y):.2f}"


SUMMARY = [  # what, and its value for a group of seats
    ("seats", lambda rows: str(len(rows))),
    ("bid at all", lambda rows: pct(len(bidders(rows)), len(rows)).strip()),
    ("r(HCP, bid at all)", corr("hcp", "bid")),
    ("r(top cards, bid at all)", corr("top", "bid")),
    ("r(long suit, bid at all)", corr("long", "bid")),
    ("seats that bid: r(HCP, highest bid)", corr("hcp", "high", bidders)),
    ("seats that bid: r(top cards, highest bid)", corr("top", "high", bidders)),
    ("seats that bid: r(long suit, highest bid)", corr("long", "high", bidders)),
    ("seats that bid: highest bid per 10 HCP", lambda rows: f"{slope(bidders(rows)):+.2f}"),
]


def agent_tables(rows):
    lines = []
    for title, key, header, groups in FEATURES:
        lines += [
            f"By {title}:",
            f"  {header:>5}  seats  bid at all  highest  contracts   made  mean score"
            "  alone   mean",
        ]
        for label, test in groups:
            group = [row for row in rows if test(row[key])]
            declared = [row for row in group if row["declared"]]
            alone = [row for row in declared if row["alone"]]
            lines.append(
                f"  {label:>5}{len(group):>7}      {pct(len(bidders(group)), len(group))}"
                f"{avg([row['high'] for row in bidders(group)], '.2f', 9)}{len(declared):>11}"
                f" {pct(sum(row['made'] for row in declared), len(declared))}"
                f"{avg([row['score'] for row in declared], '+.0f', 12)}{len(alone):>7}"
                f"{avg([row['score'] for row in alone], '+.0f', 7)}"
            )

    lines += [
        "Bid at all by turn to speak, and whether an earlier seat had bid by then:",
        "  speaks  earlier bid  seats  bid at all  mean HCP: bidding  passing",
    ]
    turns = [
        (
            f"{speaks:>6}  {'yes' if earlier else 'no':>11}",
            [row for row in rows if (row["speaks"], row["earlier_bid"]) == (speaks, earlier)],
        )
        for speaks in range(1, 5)
        for earlier in (False, True)
    ]
    for label, group in turns + [(f"{'all':>6}{'':>13}", rows)]:
        if group:
            bid_hcp = [row["hcp"] for row in group if row["bid"]]
            pass_hcp = [row["hcp"] for row in group if not row["bid"]]
            lines.append(
                f"  {label}{len(group):>7}      {pct(len(bid_hcp), len(group))}"
                f"{avg(bid_hcp, '.1f', 19)}{avg(pass_hcp, '.1f', 9)}"
            )

    declared = [row for row in rows if row["declared"]]
    lines += [
        "As declarer, by attachment (trumps held: cards of the trump suit in the original hand):",
        "  attachment  contracts  level   HCP   top  trumps held   made  mean score",
    ]
    for name in (*ATTACHMENTS, "all"):
        group = [row for row in declared if name in (row["attachment"], "all")]
        if group:
            held = [row["trumps_held"] for row in group if row["trumps_held"] is not None]
            lines.append(
                f"  {name:<10}{len(group):>11}{avg([row['level'] for row in group], '.2f', 7)}"
                f"{avg([row['hcp'] for row in group], '.1f', 6)}"
                f"{avg([row['top'] for row in group], '.2f', 6)}{avg(held, '.2f', 13)}"
                f" {pct(sum(row['made'] for row in group), len(group))}"
                f"{avg([row['score'] for row in group], '+.0f', 12)}"
            )
    return lines


def paired_line(label, group, total):
    """One row of the paired table: `group` is (agent's row, RuleBot's row) pairs."""
    declared = [a for a, _ in group if a["declared"]]
    diffs = [a["score"] - b["score"] for a, b in group]
    ci = 1.96 * stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else math.nan
    return (
        f"  {label:<16}{len(group):>6} {pct(len(group), total)}"
        f"{avg([a['hcp'] for a, _ in group], '.1f', 6)}"
        f"       {pct(len(declared), len(group))}"
        f" {pct(sum(a['made'] for a in declared), len(declared))}"
        f"{avg([a['score'] for a, _ in group], '+.0f', 14)}"
        f"{avg([b['score'] for _, b in group], '+.0f', 7)}"
        + (f"{mean(diffs):>+8.1f} ± {ci:.1f}" if diffs else "")
    )


def paired_table(agent, pairs):
    """The agent against RuleBot with the same cards in the same seat, split by who bid at all,
    and where only the agent bid, by whether it declared and by HCP."""
    lines = [
        f"{agent} against RuleBot with the same cards in the same seat",
        "(the all-rule lineup), by who bid at all. Declares and made: the agent's. Score: the",
        "seat's, whatever its role. Advantage: the agent's score minus RuleBot's, ± 95% interval:",
        "what its bidding and its card play gain together.",
        "  agent   rule     seats  share   HCP  agent declares   made  score: agent   rule"
        "   advantage",
    ]
    for mine, theirs in ((True, True), (True, False), (False, True), (False, False)):
        group = [(a, b) for a, b in pairs if (a["bid"], b["bid"]) == (mine, theirs)]
        label = f"{'bid' if mine else 'passed':<8}{'bid' if theirs else 'passed'}"
        lines.append(paired_line(label, group, len(pairs)))
        if mine and not theirs:
            for name, declared in (("  declared", True), ("  outbid", False)):
                subgroup = [(a, b) for a, b in group if a["declared"] == declared]
                lines.append(paired_line(name, subgroup, len(pairs)))
            for band, test in HCP_BANDS:
                subgroup = [(a, b) for a, b in group if test(a["hcp"])]
                lines.append(paired_line(f"  HCP {band}", subgroup, len(pairs)))
    lines.append(paired_line("all", pairs, len(pairs)))
    both = [(a, b) for a, b in pairs if a["bid"] and b["bid"]]
    if both:
        lines.append(
            f"  when both bid, highest level: agent {mean(a['high'] for a, _ in both):.2f},"
            f" rule {mean(b['high'] for _, b in both):.2f}; agent higher"
            f" {pct(sum(a['high'] > b['high'] for a, b in both), len(both)).strip()},"
            f" lower {pct(sum(a['high'] < b['high'] for a, b in both), len(both)).strip()}"
        )
    return lines


def main(paths):
    records = [json.loads(line) for path in paths for line in open(path)]
    by_agent, baseline = defaultdict(list), {}  # baseline: the all-rule lineup's rows, by place
    for record in records:
        all_rule = all(name == BASELINE for name in record["meta"]["seats"])
        for row in seat_rows(record):
            if all_rule:
                baseline[row["place"]] = row
            elif row["agent"] != BASELINE:
                by_agent[row["agent"]].append(row)
    agents = sorted(by_agent)
    groups, pairs = [], {}  # groups: (agent, label, rows), RuleBot in its seats after each agent
    for agent in agents:
        groups.append((agent, "itself", by_agent[agent]))
        pairs[agent] = [
            (row, baseline[row["place"]]) for row in by_agent[agent] if row["place"] in baseline
        ]
        if pairs[agent]:
            groups.append((agent, "rule, same seats", [b for _, b in pairs[agent]]))

    redeals = sum(record["outcome"]["redeal"] for record in records)
    out = [
        f"Hand reading: {len(records)} records from {', '.join(paths)}, {redeals} redealt.",
        NOTES,
        "Summary. r: Pearson correlation over seats. Per 10 HCP: least-squares slope.",
    ]
    width = max([len(what) + 4 for what, _ in SUMMARY] + [len(agent) + 1 for agent in agents])
    for agent in agents:
        columns = [(label, rows) for name, label, rows in groups if name == agent]
        out.append(f"{agent:<{width}}" + "".join(f"{label:>18}" for label, _ in columns))
        for what, value in SUMMARY:
            values = "".join(f"{value(rows):>18}" for _, rows in columns)
            out.append(f"  {what:<{width - 2}}{values}")

    for agent, label, rows in groups:
        same_seats = label == "rule, same seats"
        title = f"rule in the same seats and deals as {agent}" if same_seats else agent
        out += ["", f"== {title}: {len(rows)} seats =="] + agent_tables(rows)
        if same_seats:
            out += [""] + paired_table(agent, pairs[agent])
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
