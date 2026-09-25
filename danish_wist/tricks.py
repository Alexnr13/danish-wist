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

    def strength(card: Card) -> tuple[int, int]:
        if card.suit is not None and card.suit == trumps:
            return (2, card.rank)
        if card.suit == led.suit:
            return (1, card.rank)
        return (0, 0)  # off-suit cards and Jokers that were not led

    seat, _ = max(trick, key=lambda played: strength(played[1]))
    return seat
