"""Computer players. A bot sees only a `PlayerView`, exactly like a human."""

from __future__ import annotations

import random
from collections import Counter

from .actions import Action, CallAce, Discard, FlipChoice, NameTrumps, Pass, Play, TakeCat
from .bidding import NUM_PLAYERS, Attachment, Bid
from .cards import ACE, KING, Card, Suit
from .game import Phase, PlayerView
from .tricks import trick_winner

# Tricks a partner is assumed to contribute when judging how high to bid.
PARTNER_TRICKS = 4


class RandomBot:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose(self, view: PlayerView) -> Action:
        return self.rng.choice(view.legal_actions)


class RuleBot:
    """Plays by a few rules of thumb. A baseline opponent, not a strong one."""

    def choose(self, view: PlayerView) -> Action:
        legal = view.legal_actions
        match view.phase:
            case Phase.IRON_HAND:
                return legal[0]  # always take the redeal
            case Phase.AUCTION:
                return self._bid(view)
            case Phase.CALL_ACE:
                return self._call_ace(view)
            case Phase.NAME_TRUMPS:
                allowed = [a.suit for a in legal]
                return NameTrumps(best_suit(view.hand, allowed))
            case Phase.FLIP:
                return FlipChoice(self._likes_flip(view))
            case Phase.EXCHANGE:
                return TakeCat(True)
            case Phase.DISCARD:
                return Discard(min(view.hand, key=lambda c: keep_value(c, view.trumps)))
            case Phase.PLAY:
                return Play(self._play(view))
        raise ValueError(f"no move in phase {view.phase}")

    def _bid(self, view: PlayerView) -> Action:
        target = int(estimate_tricks(view.hand)) + PARTNER_TRICKS
        attachments = {None}
        if best_suit(view.hand, list(Suit)) is Suit.CLUBS:
            attachments.add(Attachment.CLUBS)
        bids = [
            a
            for a in view.legal_actions
            if isinstance(a, Bid) and a.level <= target and a.attachment in attachments
        ]
        if not bids:
            return Pass()
        return min(bids, key=lambda b: (b.level, b.attachment is not None))

    def _call_ace(self, view: PlayerView) -> Action:
        planned_trumps = best_suit(view.hand, list(Suit))
        if view.bid.attachment is Attachment.CLUBS:
            planned_trumps = Suit.CLUBS
        suits = [a.suit for a in view.legal_actions if a.suit is not planned_trumps]
        missing = [s for s in suits if Card(ACE, s) not in view.hand] or suits
        # Prefer an ace that sits over our king, then a suit we hold more of.
        best = max(missing, key=lambda s: (Card(KING, s) in view.hand, suit_length(view.hand, s)))
        return CallAce(best)

    def _likes_flip(self, view: PlayerView) -> bool:
        turned = view.turned_cat[-1]
        if turned.is_joker:
            return sum(c.rank == ACE for c in view.hand) >= 3
        return suit_length(view.hand, turned.suit) >= 4

    def _play(self, view: PlayerView) -> Card:
        cards = [a.card for a in view.legal_actions]
        trick = list(view.trick)
        by_cheapness = sorted(cards, key=lambda c: keep_value(c, view.trumps))
        if not trick:
            return self._lead(view, cards, by_cheapness)
        if trick_winner(trick, view.trumps) in allies(view):
            return by_cheapness[0]
        winning = [
            c
            for c in by_cheapness
            if trick_winner(trick + [(view.seat, c)], view.trumps) == view.seat
        ]
        return winning[0] if winning else by_cheapness[0]

    def _lead(self, view: PlayerView, cards: list[Card], by_cheapness: list[Card]) -> Card:
        jokers = [c for c in cards if c.is_joker]
        if jokers:
            return jokers[0]  # a led Joker always wins
        side_aces = [c for c in cards if c.rank == ACE and c.suit is not view.trumps]
        if side_aces:
            return side_aces[0]
        trumps = [c for c in cards if c.suit is view.trumps]
        if view.seat in (view.declarer, view.partner) and len(trumps) >= 3:
            return max(trumps, key=lambda c: c.rank)
        non_trumps = [c for c in by_cheapness if c.suit is not view.trumps]
        return (non_trumps or by_cheapness)[0]


def allies(view: PlayerView) -> set[int]:
    """Seats this player knows to be on their side (always including themselves)."""
    if view.partner is None:
        return {view.seat}
    declaring_side = {view.declarer, view.partner}
    if view.seat in declaring_side:
        return declaring_side
    return set(range(NUM_PLAYERS)) - declaring_side


def suit_length(hand: tuple[Card, ...] | list[Card], suit: Suit) -> int:
    return sum(card.suit is suit for card in hand)


def best_suit(hand, allowed: list[Suit]) -> Suit:
    """The longest allowed suit, ties broken by high cards."""
    return max(
        allowed, key=lambda s: (suit_length(hand, s), sum(c.rank for c in hand if c.suit is s))
    )


def estimate_tricks(hand) -> float:
    """A rough count of the tricks this hand takes by itself."""
    tricks = float(sum(card.is_joker for card in hand))
    counts = Counter(card.suit for card in hand if not card.is_joker)
    for card in hand:
        if card.rank == ACE:
            tricks += 1
        elif card.rank == KING and counts[card.suit] >= 2:
            tricks += 0.5
    longest = max(counts.values(), default=0)
    return tricks + max(0, longest - 3)


def keep_value(card: Card, trumps: Suit | None) -> int:
    """How much a card is worth keeping: Jokers, then trumps, then by rank."""
    if card.is_joker:
        return 100
    if card.suit is trumps:
        return 50 + card.rank
    return card.rank
