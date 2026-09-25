"""Replay deals recorded before the engine was optimised, checking every step.

`data/golden_deals.jsonl` holds records made by the original engine. Each one
carries a digest of what every seat saw at every step (all four views, with
legal actions in order), so any change in behaviour shows up here, not just a
change of outcome. Regenerate only for an intended rule change:

    python tests/test_golden.py
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import fields
from enum import Enum
from pathlib import Path

from danish_wist import Card, Deal, Pass, Phase, PlayerView
from danish_wist.actions import Action, decode, encode
from danish_wist.bots import RandomBot, RuleBot
from danish_wist.record import replay, to_record

GOLDEN = Path(__file__).parent / "data" / "golden_deals.jsonl"
SEED = 20260925
STREAM = 150  # deals dealt in order from Random(SEED); more are added for coverage
VIEW_FIELDS = [
    "seat",
    "phase",
    "to_act",
    "legal_actions",
    "dealer",
    "hand",
    "auction",
    "declarer",
    "bid",
    "called_suit",
    "partner",
    "trumps",
    "turned_cat",
    "discards",
    "fucdic_declared",
    "fucdic",
    "trick",
    "tricks",
    "tricks_won",
    "scores",
]


def canonical(value):
    """Plain JSON for a view field. Containers must stay tuples."""
    match value:
        case None | bool() | int():
            return value
        case Enum():
            return f"{type(value).__name__}.{value.name}"
        case Card():
            return str(value)
        case tuple():
            return [canonical(item) for item in value]
    if isinstance(value, Action):
        return encode(value)
    raise TypeError(f"unexpected {type(value).__name__} in a view: {value!r}")


def step_state(deal: Deal) -> list:
    views = [deal.view(seat) for seat in range(4)]
    return [
        deal.to_act,
        deal.is_over,
        [encode(action) for action in deal.legal_actions()],
        [[canonical(getattr(view, name)) for name in VIEW_FIELDS] for view in views],
    ]


def trace_digest(record: dict) -> str:
    """Replay `record` step by step, hashing the state before and after every action."""
    deal = Deal(
        record["dealer"],
        [[Card.parse(c) for c in hand.split()] for hand in record["hands"]],
        [Card.parse(c) for c in record["cat"].split()],
    )
    steps = [step_state(deal)]
    for seat, text in record["actions"]:
        assert seat == deal.to_act
        deal.apply(decode(text))
        steps.append(step_state(deal))
    text = json.dumps(steps, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def load() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines()]


def test_view_fields_are_unchanged():
    assert [f.name for f in fields(PlayerView)] == VIEW_FIELDS


def test_seeded_dealing_is_unchanged():
    rng = random.Random(SEED)
    for i, record in enumerate(load()[:STREAM]):
        deal = Deal.new(i % 4, rng)
        assert [" ".join(map(str, hand)) for hand in deal.initial_hands] == record["hands"]
        assert " ".join(map(str, deal.cat)) == record["cat"]


def test_golden_deals_replay_step_by_step():
    records = load()
    assert len(records) >= STREAM
    for i, record in enumerate(records):
        assert to_record(replay(record), **record["meta"]) == record, f"deal {i}"
        assert trace_digest(record) == record["meta"]["trace"], f"deal {i}"


# --- Generating the golden file ------------------------------------------------


class FirstChoice:
    """Always the first legal action, so every auction is passed out."""

    def choose(self, view: PlayerView):
        return view.legal_actions[0]


def agents_for(i: int, rng: random.Random) -> list:
    match i % 8:
        case 0 | 3 | 6:
            return [RandomBot(rng)] * 4
        case 1 | 4:
            return [RuleBot()] * 4
        case 7:
            return [FirstChoice()] * 4
        case _:
            return [RuleBot(), RandomBot(rng)] * 2 if i % 2 else [RandomBot(rng), RuleBot()] * 2


def play(deal: Deal, agents: list) -> tuple[Deal, set[str]]:
    """Play to the end, noting the situations met on the way."""
    tags = set()
    while not deal.is_over:
        seat = deal.to_act
        view = deal.view(seat)
        action = agents[seat].choose(view)
        words = encode(action).split()
        if words[0] in ("play", "discard", "fucdic"):
            words = words[:1]
        elif words[0] == "bid":
            words = ["bid", *words[2:]]
        tags.add(" ".join(words))
        tags.add(view.phase.name)
        if view.phase is Phase.PLAY and view.trick:
            if len(view.legal_actions) == 1:
                tags.add("forced play")
            if view.trick[0][1].is_joker:
                tags.add("joker led")
        deal.apply(action)
    if deal.redeal:
        tags.add("all passed" if isinstance(deal.history[-1][1], Pass) else "iron-hand redeal")
    else:
        tags.add("alone" if deal.alone else "partner")
        tags.add(f"turned {deal.turned}")
        tags.add(f"trumps {deal.trumps}")
        tags.add("fucdic placed" if deal.fucdic else "no fucdic")
        tags.add("made" if deal.scores[deal.declarer] > 0 else "failed")
    return deal, tags


def generate(pool: int = 20000) -> list[dict]:
    deck_rng, bot_rng = random.Random(SEED), random.Random(SEED + 1)
    chosen, covered = [], set()
    for i in range(STREAM + pool):
        deal, tags = play(Deal.new(i % 4, deck_rng), agents_for(i, bot_rng))
        if i < STREAM or not tags <= covered:
            chosen.append(to_record(deal))
            covered |= tags
    for record in chosen:
        record["meta"] = {"trace": trace_digest(record)}
    return chosen


if __name__ == "__main__":
    records = generate()
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))
    print(f"wrote {len(records)} deals to {GOLDEN}")
