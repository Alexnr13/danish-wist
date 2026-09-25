"""Following suit and winning tricks (RULES.md §7)."""

from __future__ import annotations

from .cards import Card, Suit

# A trick is the cards played so far, in order, with the seat that played each.
Trick = list[tuple[int, Card]]


def legal_plays(hand: list[Card], trick: Trick, called_ace: Card | None) -> list[Card]:
    options = list(dict.fromkeys(hand))  # Jokers are interchangeable
    if not trick:
        return options
    led = trick[0][1]
    if led.is_joker:
        return options
    following = [card for card in options if card.suit == led.suit]
    if not following:
        return options
    # Someone else led the called suit: its ace must be played now.
    if called_ace in following:
        return [called_ace]
    return following


def trick_winner(trick: Trick, trumps: Suit | None) -> int:
    leader, led = trick[0]
    if led.is_joker:
        return leader
    seat, _ = max(trick, key=lambda played: _strength(played[1], led.suit, trumps))
    return seat


def winning_cards(cards: list[Card], trick: Trick, trumps: Suit | None) -> list[Card]:
    """Those of `cards` that would win `trick` so far if played next. `trick` has been led."""
    led = trick[0][1]
    if led.is_joker:
        return []  # a led Joker always wins
    # On a tie the earlier card wins, so the next card must be strictly stronger.
    best = max(_strength(card, led.suit, trumps) for _, card in trick)
    return [card for card in cards if _strength(card, led.suit, trumps) > best]


def _strength(card: Card, led_suit: Suit, trumps: Suit | None) -> tuple[int, int]:
    if card.suit is not None and card.suit == trumps:
        return (2, card.rank)
    if card.suit == led_suit:
        return (1, card.rank)
    return (0, 0)  # off-suit cards and Jokers that were not led
