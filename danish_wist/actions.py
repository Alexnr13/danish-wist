"""Every decision a player can make, one small immutable type per kind."""

from __future__ import annotations

from dataclasses import dataclass

from .bidding import Bid
from .cards import Card, Suit


@dataclass(frozen=True)
class DeclareIronHand:
    declare: bool


@dataclass(frozen=True)
class Pass:
    pass


@dataclass(frozen=True)
class CallAce:
    suit: Suit


@dataclass(frozen=True)
class NameTrumps:
    suit: Suit


@dataclass(frozen=True)
class FlipChoice:
    """In Flip: accept the last turned card as trumps, or turn the next one."""

    accept: bool


@dataclass(frozen=True)
class TakeCat:
    take: bool


@dataclass(frozen=True)
class Discard:
    card: Card


@dataclass(frozen=True)
class Play:
    card: Card


Action = DeclareIronHand | Pass | Bid | CallAce | NameTrumps | FlipChoice | TakeCat | Discard | Play
