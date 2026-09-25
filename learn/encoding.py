"""Turn a player's view into tokens, and actions into indices (LEARNING.md §3).

The network sees a deal as a short sequence of tokens: the cards in hand plus
one token per public event, all from the viewer's point of view (seats are
relative: 0 is the viewer, 1 the next player clockwise, and so on). Every
token is five small integers, which the network embeds and sums:

    (kind, card, seat, value, position)

Only a `PlayerView` goes into `encode`, so nothing hidden can leak into what
a policy sees. `encode_oracle` (for training critics only) adds the hidden cards.

Actions use one fixed index space for all phases; `legal_mask` marks the ones
allowed now. Pure Python: no NumPy or PyTorch needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from danish_wist.actions import (
    Action,
    CallAce,
    DeclareFucdic,
    DeclareIronHand,
    Discard,
    FlipChoice,
    NameTrumps,
    Pass,
    Play,
    TakeCat,
)
from danish_wist.bidding import ALL_BIDS, NUM_PLAYERS, Bid
from danish_wist.cards import ACE, FUCDIC_RANK, JOKER, Card, Suit
from danish_wist.game import Deal, Phase, PlayerView

SUITS = list(Suit)
PHASES = list(Phase)

# --- Cards -------------------------------------------------------------------

NO_CARD = 0
REAL_CARDS = [Card(rank, suit) for suit in SUITS for rank in range(2, ACE + 1)] + [JOKER]
FUCDIC_CARDS = [Card(FUCDIC_RANK, suit) for suit in SUITS]
ALL_CARDS = REAL_CARDS + FUCDIC_CARDS
_CARD_INDEX = {card: i + 1 for i, card in enumerate(ALL_CARDS)}
NUM_CARD_IDS = len(ALL_CARDS) + 1  # including NO_CARD


def card_id(card: Card | None) -> int:
    return NO_CARD if card is None else _CARD_INDEX[card]


# --- Tokens ------------------------------------------------------------------


class Kind(IntEnum):
    PAD = 0
    PHASE = 1  # value: phase
    DEALER = 2  # seat
    HAND = 3  # card
    BID = 4  # seat, value: bid id (0 = pass), position: order in the auction
    CONTRACT = 5  # seat: declarer, value: bid id
    CALLED = 6  # value: suit
    PARTNER = 7  # seat, once known to the viewer
    TRUMPS = 8  # value: suit, or NO_TRUMPS
    TURNED = 9  # card turned in Flip, position: order
    DISCARD = 10  # card the viewer discarded (declarer only)
    FUCDIC = 11  # seat: declarer, card: the real card if the viewer placed it
    PLAY = 12  # seat, card, value: place in the trick, position: trick number


NO_SEAT = NUM_PLAYERS
NO_TRUMPS = len(SUITS)
_BID_ID = {bid: i + 1 for i, bid in enumerate(ALL_BIDS)}  # 0 is pass
NUM_BID_IDS = len(ALL_BIDS) + 1
NUM_VALUES = max(NUM_BID_IDS, len(PHASES), NO_TRUMPS + 1)  # largest `value` + 1
MAX_TOKENS = 160  # a player's own view: hand, auction, contract, cat, 52 plays, and room
MAX_ORACLE_TOKENS = MAX_TOKENS + 48  # plus the hidden cards

Token = tuple[int, int, int, int, int]


def bid_id(bid: Bid | None) -> int:
    return 0 if bid is None else _BID_ID[bid]


def encode(view: PlayerView) -> list[Token]:
    """The view as tokens, seats relative to the viewer."""

    def rel(seat: int | None) -> int:
        return NO_SEAT if seat is None else (seat - view.seat) % NUM_PLAYERS

    tokens: list[Token] = [
        (Kind.PHASE, NO_CARD, NO_SEAT, PHASES.index(view.phase), 0),
        (Kind.DEALER, NO_CARD, rel(view.dealer), 0, 0),
    ]
    tokens += [(Kind.HAND, card_id(card), NO_SEAT, 0, 0) for card in view.hand]
    tokens += [
        (Kind.BID, NO_CARD, rel(seat), bid_id(bid), i) for i, (seat, bid) in enumerate(view.auction)
    ]
    if view.declarer is not None:
        tokens.append((Kind.CONTRACT, NO_CARD, rel(view.declarer), bid_id(view.bid), 0))
    if view.called_suit is not None:
        tokens.append((Kind.CALLED, NO_CARD, NO_SEAT, SUITS.index(view.called_suit), 0))
    if view.partner is not None:
        tokens.append((Kind.PARTNER, NO_CARD, rel(view.partner), 0, 0))
    if trumps_decided(view):
        trumps = NO_TRUMPS if view.trumps is None else SUITS.index(view.trumps)
        tokens.append((Kind.TRUMPS, NO_CARD, NO_SEAT, trumps, 0))
    tokens += [(Kind.TURNED, card_id(c), NO_SEAT, 0, i) for i, c in enumerate(view.turned_cat)]
    tokens += [(Kind.DISCARD, card_id(card), NO_SEAT, 0, 0) for card in view.discards]
    if view.fucdic_declared:
        tokens.append((Kind.FUCDIC, card_id(view.fucdic), rel(view.declarer), 0, 0))
    for number, trick in enumerate([*view.tricks, view.trick]):
        tokens += [
            (Kind.PLAY, card_id(card), rel(seat), place, number)
            for place, (seat, card) in enumerate(trick)
        ]
    assert len(tokens) <= MAX_TOKENS
    return tokens


def encode_oracle(deal: Deal, seat: int) -> list[Token]:
    """The seat's own tokens plus every hidden card: **for training critics only**.

    A critic that sees the hidden cards judges positions far more accurately,
    which makes learning much less noisy. The policy never sees these tokens.
    The extra tokens reuse the existing kinds, marked by `seat` and `value`:
    another player's hand (HAND, their seat), the untaken cat (HAND, no seat,
    value 1), the declarer's discards (DISCARD, declarer's seat) and the real
    fucdic card (FUCDIC, value 1).
    """
    tokens = encode(deal.view(seat))

    def rel(other: int) -> int:
        return (other - seat) % NUM_PLAYERS

    for other in range(NUM_PLAYERS):
        if other != seat:
            tokens += [(Kind.HAND, card_id(c), rel(other), 0, 0) for c in deal.hands[other]]
    if not deal.took_cat:
        tokens += [(Kind.HAND, card_id(c), NO_SEAT, 1, 0) for c in deal.cat]
    if seat != deal.declarer:
        tokens += [(Kind.DISCARD, card_id(c), rel(deal.declarer), 0, 0) for c in deal.discards]
        if deal.fucdic is not None:
            tokens.append((Kind.FUCDIC, card_id(deal.fucdic), rel(deal.declarer), 1, 0))
    return tokens


def trumps_decided(view: PlayerView) -> bool:
    return PHASES.index(view.phase) >= PHASES.index(Phase.EXCHANGE)


# --- Actions -----------------------------------------------------------------

ACTIONS: list[Action] = [
    DeclareIronHand(True),
    DeclareIronHand(False),
    Pass(),
    *ALL_BIDS,
    *(CallAce(s) for s in SUITS),
    *(NameTrumps(s) for s in SUITS),
    FlipChoice(True),
    FlipChoice(False),
    TakeCat(True),
    TakeCat(False),
    *(Discard(c) for c in REAL_CARDS),
    DeclareFucdic(None),
    *(DeclareFucdic(c) for c in REAL_CARDS),
    *(Play(c) for c in ALL_CARDS),
]
NUM_ACTIONS = len(ACTIONS)
_ACTION_INDEX = {action: i for i, action in enumerate(ACTIONS)}


def action_index(action: Action) -> int:
    return _ACTION_INDEX[action]


def legal_mask(view: PlayerView) -> list[bool]:
    mask = [False] * NUM_ACTIONS
    for action in view.legal_actions:
        mask[_ACTION_INDEX[action]] = True
    return mask


@dataclass(frozen=True)
class Observation:
    """Everything a network needs for one decision."""

    tokens: list[Token]
    legal: list[int]  # indices into ACTIONS


def observe(view: PlayerView) -> Observation:
    return Observation(encode(view), [_ACTION_INDEX[a] for a in view.legal_actions])
