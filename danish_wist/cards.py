"""Cards and the 55-card deck: a standard 52 plus 3 Jokers (RULES.md §2)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"

    def __str__(self) -> str:
        return self.value


FUCDIC_RANK = 0  # the face-down fucdic ranks below the 2 of its suit
JACK, QUEEN, KING, ACE = 11, 12, 13, 14
_RANK_NAMES = {JACK: "J", QUEEN: "Q", KING: "K", ACE: "A"}
_RANKS_BY_NAME = {name: rank for rank, name in _RANK_NAMES.items()}


@dataclass(frozen=True)
class Card:
    """A playing card. Jokers have no suit; a fucdic stand-in has rank 0."""

    rank: int
    suit: Suit | None

    @property
    def is_joker(self) -> bool:
        return self.suit is None

    def __str__(self) -> str:
        if self.is_joker:
            return "JK"
        return f"{_RANK_NAMES.get(self.rank, self.rank)}{self.suit}"

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def parse(cls, text: str) -> Card:
        """Parse short notation such as 'AS', '10H', '2C', 'JK' (Joker) or '0D' (fucdic)."""
        text = text.strip().upper()
        if text == "JK":
            return JOKER
        rank_text, suit_text = text[:-1], text[-1]
        rank = _RANKS_BY_NAME[rank_text] if rank_text in _RANKS_BY_NAME else int(rank_text)
        if not (2 <= rank <= ACE or rank == FUCDIC_RANK):
            raise ValueError(f"bad card: {text!r}")
        return cls(rank, Suit(suit_text))


JOKER = Card(0, None)


def ace_of(suit: Suit) -> Card:
    return Card(ACE, suit)


def parse_cards(text: str) -> list[Card]:
    """Parse a space-separated list of cards, e.g. 'AS KS JK'."""
    return [Card.parse(part) for part in text.split()]


def full_deck() -> list[Card]:
    return [Card(rank, suit) for suit in Suit for rank in range(2, ACE + 1)] + [JOKER] * 3


def shuffled_deck(rng: random.Random) -> list[Card]:
    deck = full_deck()
    rng.shuffle(deck)
    return deck


def is_iron_hand(hand: list[Card]) -> bool:
    """No court cards, no aces and no Jokers (RULES.md §3)."""
    return all(not card.is_joker and card.rank <= 10 for card in hand)


def sort_key(card: Card) -> tuple[int, int]:
    """Group by suit, low to high, with Jokers last."""
    if card.is_joker:
        return (len(Suit), 0)
    return (list(Suit).index(card.suit), card.rank)
