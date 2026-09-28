"""What agents actually do: a report from recorded deals.

Reads records written by `python -m learn.arena ... --record games.jsonl`
(or any `danish_wist.record` records whose `meta.seats` names the agent in
each seat) and summarises each agent's bidding and results:

    python -m learn.report games.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from danish_wist.bidding import NUM_PLAYERS
from danish_wist.cards import ACE, Suit, parse_cards

ATTACHMENTS = ("plain", "flip", "clubs", "halves")


def top_cards(hand_text: str) -> int:
    """Aces and Jokers in a hand: a rough measure of its strength."""
    return sum(card.is_joker or card.rank == ACE for card in parse_cards(hand_text))


def seat_rows(record: dict) -> list[dict]:
    """One row per seat: who played it, what they held and bid, and how it went."""
    outcome = record["outcome"]
    rows = []
    highest: dict[int, int] = {}
    for seat, text in record["actions"]:
        if text.startswith("bid "):
            highest[seat] = max(highest.get(seat, 0), int(text.split()[1]))
    for seat in range(NUM_PLAYERS):
        row = {
            "agent": record["meta"]["seats"][seat],
            "top_cards": top_cards(record["hands"][seat]),
            "highest_bid": highest.get(seat),
            "redeal": outcome["redeal"],
        }
        if not outcome["redeal"]:
            declarer, partner = outcome["declarer"], outcome["partner"]
            row["score"] = outcome["scores"][seat]
            if seat == declarer:
                row["role"] = "alone" if partner == declarer else "declarer"
            else:
                row["role"] = "partner" if seat == partner else "defender"
            if seat == declarer:
                level, *attachment = outcome["bid"].split()
                taken = outcome["tricks_won"][declarer]
                if partner != declarer:
                    taken += outcome["tricks_won"][partner]
                row |= {
                    "level": int(level),
                    "attachment": attachment[0] if attachment else "plain",
                    "made": taken >= int(level),
                    "exchanged": any(a == [seat, "take-cat"] for a in record["actions"]),
                    "fucdic": outcome.get("fucdic") is not None,
                }
                if partner == declarer:
                    # RULES.md §5: alone by calling an ace it held, or one in the cat. It had a
                    # choice if it could have called an ace it did not hold (not clubs in Clubs).
                    hand = record["hands"][seat].split()
                    clubs = row["attachment"] == "clubs"
                    suits = [s.value for s in Suit if not (clubs and s is Suit.CLUBS)]
                    row["cause"] = "own ace" if "A" + outcome["called"] in hand else "ace in cat"
                    row["choice"] = any("A" + suit not in hand for suit in suits)
        rows.append(row)
    return rows


def _share(rows: list[dict], test) -> str:
    return f"{sum(bool(test(row)) for row in rows) / len(rows):6.1%}" if rows else "     -"


def _mean(values: list[float], fmt: str = "{:+8.0f}") -> str:
    return fmt.format(mean(values)) if values else "       -"


def report(records: list[dict]) -> str:
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        for row in seat_rows(record):
            by_agent[row["agent"]].append(row)

    lines = []
    for agent, rows in sorted(by_agent.items()):
        played = [r for r in rows if not r["redeal"]]
        declaring = [r for r in played if r["role"] in ("declarer", "alone")]
        lines += [
            f"== {agent}: {len(rows)} seats, {len(played)} played ==",
            "Role        share   mean score",
        ]
        for name in ("declarer", "alone", "partner", "defender"):
            scores = [r["score"] for r in played if r["role"] == name]
            share = _share(played, lambda r, n=name: r["role"] == n)
            lines.append(f"  {name:<9}{share}  {_mean(scores)}")

        if declaring:
            lines += [
                "As declarer:",
                f"  mean level {mean(r['level'] for r in declaring):.2f}, "
                f"made {_share(declaring, lambda r: r['made']).strip()}, "
                f"exchanged {_share(declaring, lambda r: r['exchanged']).strip()}, "
                f"fucdic {_share(declaring, lambda r: r['fucdic']).strip()}",
                "  attachment "
                + ", ".join(
                    f"{a} {_share(declaring, lambda r, a=a: r['attachment'] == a).strip()}"
                    for a in ATTACHMENTS
                ),
                "  level      "
                + ", ".join(
                    f"{lv}: {sum(r['level'] == lv for r in declaring)}" for lv in range(7, 14)
                ),
            ]

        if any(r["role"] == "alone" for r in played):
            lines.append("Alone by cause  count   share   mean score")
            for cause in ("own ace", "ace in cat"):
                group = [r for r in played if r.get("cause") == cause]
                share = _share(played, lambda r, c=cause: r.get("cause") == c)
                scores = [r["score"] for r in group]
                line = f"  {cause:<12}{len(group):>7}  {share}  {_mean(scores)}"
                if cause == "own ace":
                    line += f"  ({sum(r['choice'] for r in group)} with a choice)"
                lines.append(line)

        lines.append("Bidding by top cards (aces + Jokers):")
        lines.append("  top  seats  bid at all  mean highest bid")
        for top in range(8):
            group = [r for r in rows if r["top_cards"] == top]
            if group:
                bids = [r["highest_bid"] for r in group if r["highest_bid"]]
                lines.append(
                    f"  {top:>3}  {len(group):>5}  {_share(group, lambda r: r['highest_bid'])}"
                    f"      {_mean(bids, '{:5.2f}')}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise what agents do in recorded deals.")
    parser.add_argument("records", type=Path, nargs="+")
    args = parser.parse_args()
    records = [json.loads(line) for path in args.records for line in path.read_text().splitlines()]
    print(report(records))


if __name__ == "__main__":
    main()
