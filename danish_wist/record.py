"""Complete, replayable records of deals, for analysis and debugging.

A record is a JSON-ready dict holding the deal's starting position and every
action in the order it was taken. Replaying it through the engine reproduces
the deal exactly, so a record is all that needs storing. A summary of the
outcome is included too, so large collections can be analysed without
replaying, and replay checks the summary still matches.

    {
      "version": 2,
      "dealer": 0,
      "hands": ["2C 5C ... JK", ...],     # the four hands as dealt
      "cat": "QD 5C JK",                  # in order, which matters for Flip
      "actions": [[1, "bid 8"], [2, "pass"], ..., [3, "play AH"], ...],
      "outcome": {"redeal": false, "declarer": 1, "partner": 3, "bid": "8",
                  "called": "H", "trumps": "S", "tricks_won": [..], "scores": [..]},
      "meta": {...}                       # anything the caller adds: match id, agents...
    }

The engine stays free of I/O: callers write records with `json.dumps`,
typically one per line (JSON Lines).
"""

from __future__ import annotations

from typing import Any

from .actions import decode, encode
from .cards import parse_cards
from .game import Deal

FORMAT_VERSION = 2  # 2: discards are made before the cat is picked up


def to_record(deal: Deal, **meta: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "version": FORMAT_VERSION,
        "dealer": deal.dealer,
        "hands": [_cards_text(hand) for hand in deal.initial_hands],
        "cat": _cards_text(deal.cat),
        "actions": [[seat, encode(action)] for seat, action in deal.history],
        "outcome": _outcome(deal),
    }
    if meta:
        record["meta"] = meta
    return record


def replay(record: dict[str, Any]) -> Deal:
    """Rebuild a deal from its record, checking every step along the way."""
    if record["version"] != FORMAT_VERSION:
        raise ValueError(f"unsupported record version {record['version']}")
    hands = [parse_cards(text) for text in record["hands"]]
    deal = Deal(record["dealer"], hands, parse_cards(record["cat"]))
    for seat, text in record["actions"]:
        if seat != deal.to_act:
            raise ValueError(f"record has seat {seat} acting, but it is seat {deal.to_act}'s turn")
        deal.apply(decode(text))
    if _outcome(deal) != record["outcome"]:
        raise ValueError("replayed outcome does not match the record")
    return deal


def _outcome(deal: Deal) -> dict[str, Any] | None:
    if not deal.is_over:
        return None
    if deal.redeal:
        return {"redeal": True}
    return {
        "redeal": False,
        "declarer": deal.declarer,
        "partner": deal.partner,
        "fucdic": str(deal.fucdic) if deal.fucdic else None,
        "bid": str(deal.bid),
        "called": str(deal.called_suit),
        "trumps": str(deal.trumps) if deal.trumps else None,
        "tricks_won": list(deal.tricks_won),
        "scores": list(deal.scores),
    }


def _cards_text(cards) -> str:
    return " ".join(str(card) for card in cards)
