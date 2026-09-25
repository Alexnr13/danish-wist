"""Every decision a player can make, one small immutable type per kind."""

from __future__ import annotations

from dataclasses import dataclass

from .bidding import Attachment, Bid
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
class DeclareFucdic:
    """Place `card` face down as the fucdic, or decline when `card` is None."""

    card: Card | None


@dataclass(frozen=True)
class Play:
    card: Card


Action = (
    DeclareIronHand
    | Pass
    | Bid
    | CallAce
    | NameTrumps
    | FlipChoice
    | TakeCat
    | Discard
    | DeclareFucdic
    | Play
)


def encode(action: Action) -> str:
    """Short, human-readable text for an action, e.g. 'bid 9 flip' or 'play AS'."""
    match action:
        case DeclareIronHand(declare):
            return "iron-hand" if declare else "no-iron-hand"
        case Pass():
            return "pass"
        case Bid(level, attachment):
            return f"bid {level} {attachment}" if attachment else f"bid {level}"
        case CallAce(suit):
            return f"call {suit}"
        case NameTrumps(suit):
            return f"trumps {suit}"
        case FlipChoice(accept):
            return "flip accept" if accept else "flip next"
        case TakeCat(take):
            return "take-cat" if take else "keep-hand"
        case Discard(card):
            return f"discard {card}"
        case DeclareFucdic(card):
            return f"fucdic {card}" if card else "no-fucdic"
        case Play(card):
            return f"play {card}"
    raise ValueError(f"unknown action: {action!r}")


def decode(text: str) -> Action:
    """The inverse of `encode`."""
    word, *args = text.split()
    match word, args:
        case "iron-hand", []:
            return DeclareIronHand(True)
        case "no-iron-hand", []:
            return DeclareIronHand(False)
        case "pass", []:
            return Pass()
        case "bid", [level]:
            return Bid(int(level))
        case "bid", [level, attachment]:
            return Bid(int(level), Attachment(attachment))
        case "call", [suit]:
            return CallAce(Suit(suit))
        case "trumps", [suit]:
            return NameTrumps(Suit(suit))
        case "flip", ["accept"]:
            return FlipChoice(True)
        case "flip", ["next"]:
            return FlipChoice(False)
        case "take-cat", []:
            return TakeCat(True)
        case "keep-hand", []:
            return TakeCat(False)
        case "discard", [card]:
            return Discard(Card.parse(card))
        case "fucdic", [card]:
            return DeclareFucdic(Card.parse(card))
        case "no-fucdic", []:
            return DeclareFucdic(None)
        case "play", [card]:
            return Play(Card.parse(card))
    raise ValueError(f"unknown action: {text!r}")
