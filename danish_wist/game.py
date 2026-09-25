"""A single deal, from the iron-hand check to settlement (RULES.md §3–§9).

A `Deal` is a state machine. Callers ask whose turn it is (`to_act`), what
they may do (`legal_actions`), and then `apply` one of those actions. What
each player is allowed to know is given by `view(seat)`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto

from .actions import (
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
from .bidding import NUM_PLAYERS, Attachment, Auction, Bid
from .cards import FUCDIC_RANK, Card, Suit, ace_of, is_iron_hand, shuffled_deck, sort_key
from .scoring import settle
from .tricks import Trick, legal_plays, trick_winner

HAND_SIZE = 13
CAT_SIZE = 3


class Phase(Enum):
    IRON_HAND = auto()
    AUCTION = auto()
    CALL_ACE = auto()
    NAME_TRUMPS = auto()
    FLIP = auto()
    EXCHANGE = auto()
    DISCARD = auto()
    FUCDIC = auto()
    PLAY = auto()
    DONE = auto()


class IllegalActionError(ValueError):
    pass


@dataclass(frozen=True)
class PlayerView:
    """Everything one seat may know. `partner` is None until known to this seat."""

    seat: int
    phase: Phase
    to_act: int | None
    legal_actions: tuple[Action, ...]
    dealer: int
    hand: tuple[Card, ...]
    auction: tuple[tuple[int, Bid | None], ...]
    declarer: int | None
    bid: Bid | None
    called_suit: Suit | None
    partner: int | None
    trumps: Suit | None
    turned_cat: tuple[Card, ...]
    discards: tuple[Card, ...]
    fucdic_declared: bool
    fucdic: Card | None  # the real face-down card, shown to the declarer only
    trick: tuple[tuple[int, Card], ...]
    tricks: tuple[tuple[tuple[int, Card], ...], ...]
    tricks_won: tuple[int, ...]
    scores: tuple[int, ...] | None


class Deal:
    def __init__(self, dealer: int, hands: list[list[Card]], cat: list[Card]) -> None:
        assert len(hands) == NUM_PLAYERS and all(len(h) == HAND_SIZE for h in hands)
        assert len(cat) == CAT_SIZE
        self.dealer = dealer
        self.hands = [sorted(hand, key=sort_key) for hand in hands]
        self.initial_hands = [tuple(hand) for hand in self.hands]
        self.cat = list(cat)

        self.auction = Auction(self.forehand)
        self.declarer: int | None = None
        self.bid: Bid | None = None
        self.called_suit: Suit | None = None
        self.partner: int | None = None  # equals the declarer when playing alone
        self.partner_revealed = False
        self.trumps: Suit | None = None  # None also means no trumps once play starts
        self.turned = 0  # cat cards turned face up in Flip
        self.took_cat = False
        self.discards: list[Card] = []
        self.fucdic: Card | None = None  # the real card placed face down
        self.trick: Trick = []
        self.tricks: list[Trick] = []
        self.tricks_won = [0] * NUM_PLAYERS
        self.leader: int | None = None
        self.scores: list[int] | None = None
        self.redeal = False
        self.history: list[tuple[int, Action]] = []  # every action, in order

        seats = self._seats_from(self.forehand)
        self._iron_hands = [s for s in seats if is_iron_hand(self.hands[s])]
        self.phase = Phase.IRON_HAND if self._iron_hands else Phase.AUCTION

    @classmethod
    def new(cls, dealer: int, rng: random.Random) -> Deal:
        deck = shuffled_deck(rng)
        hands = [deck[i * HAND_SIZE : (i + 1) * HAND_SIZE] for i in range(NUM_PLAYERS)]
        return cls(dealer, hands, deck[NUM_PLAYERS * HAND_SIZE :])

    # --- Queries -------------------------------------------------------------

    @property
    def forehand(self) -> int:
        return (self.dealer + 1) % NUM_PLAYERS

    @property
    def is_over(self) -> bool:
        return self.phase is Phase.DONE

    @property
    def alone(self) -> bool:
        return self.partner == self.declarer

    @property
    def called_ace(self) -> Card | None:
        return ace_of(self.called_suit) if self.called_suit else None

    @property
    def to_act(self) -> int | None:
        match self.phase:
            case Phase.IRON_HAND:
                return self._iron_hands[0]
            case Phase.AUCTION:
                return self.auction.to_act
            case Phase.NAME_TRUMPS:
                return self.partner if self.bid.attachment is Attachment.HALVES else self.declarer
            case Phase.CALL_ACE | Phase.FLIP | Phase.EXCHANGE | Phase.DISCARD | Phase.FUCDIC:
                return self.declarer
            case Phase.PLAY:
                return (self.leader + len(self.trick)) % NUM_PLAYERS
            case Phase.DONE:
                return None

    def legal_actions(self) -> list[Action]:
        match self.phase:
            case Phase.IRON_HAND:
                return [DeclareIronHand(True), DeclareIronHand(False)]
            case Phase.AUCTION:
                return [Pass(), *self.auction.legal_bids()]
            case Phase.CALL_ACE:
                clubs_trumps = self.bid.attachment is Attachment.CLUBS
                return [CallAce(s) for s in Suit if not (clubs_trumps and s is Suit.CLUBS)]
            case Phase.NAME_TRUMPS:
                return [NameTrumps(s) for s in Suit if s is not self.called_suit]
            case Phase.FLIP:
                return [FlipChoice(True), FlipChoice(False)]
            case Phase.EXCHANGE:
                return [TakeCat(True), TakeCat(False)]
            case Phase.DISCARD:
                return [Discard(c) for c in dict.fromkeys(self.hands[self.declarer])]
            case Phase.FUCDIC:
                in_suit = [c for c in self.hands[self.declarer] if c.suit is self.called_suit]
                candidates = in_suit or list(dict.fromkeys(self.hands[self.declarer]))
                return [DeclareFucdic(None), *(DeclareFucdic(c) for c in candidates)]
            case Phase.PLAY:
                cards = legal_plays(self.hands[self.to_act], self.trick, self.called_ace)
                return [Play(c) for c in cards]
            case Phase.DONE:
                return []

    def view(self, seat: int) -> PlayerView:
        return PlayerView(
            seat=seat,
            phase=self.phase,
            to_act=self.to_act,
            legal_actions=tuple(self.legal_actions()) if seat == self.to_act else (),
            dealer=self.dealer,
            hand=tuple(self.hands[seat]),
            auction=tuple(self.auction.history),
            declarer=self.declarer,
            bid=self.bid,
            called_suit=self.called_suit,
            partner=self._partner_known_to(seat),
            trumps=self.trumps,
            turned_cat=tuple(self.cat[: self.turned]),
            discards=tuple(self.discards) if seat == self.declarer else (),
            fucdic_declared=self.fucdic is not None,
            fucdic=self.fucdic if seat == self.declarer else None,
            trick=tuple(self.trick),
            tricks=tuple(tuple(t) for t in self.tricks),
            tricks_won=tuple(self.tricks_won),
            scores=tuple(self.scores) if self.scores else None,
        )

    # --- Actions -------------------------------------------------------------

    def apply(self, action: Action) -> None:
        if action not in self.legal_actions():
            raise IllegalActionError(f"{action} is not legal in {self.phase.name}")
        self.history.append((self.to_act, action))

        match action:
            case DeclareIronHand(declare=True):
                self._finish_with_redeal()
            case DeclareIronHand(declare=False):
                self._iron_hands.pop(0)
                if not self._iron_hands:
                    self.phase = Phase.AUCTION
            case Pass():
                self._auction_act(None)
            case Bid():
                self._auction_act(action)
            case CallAce(suit):
                self._call_ace(suit)
            case NameTrumps(suit):
                self.trumps = suit
                if self.bid.attachment is Attachment.HALVES:
                    self.partner_revealed = True
                self.phase = Phase.EXCHANGE
            case FlipChoice(accept):
                self._flip(accept)
            case TakeCat(take=True):
                self._take_cat()
            case TakeCat(take=False):
                self._offer_fucdic()
            case Discard(card):
                self.hands[self.declarer].remove(card)
                self.discards.append(card)
                if len(self.discards) == CAT_SIZE:
                    self._offer_fucdic()
            case DeclareFucdic(card):
                if card is not None:
                    self._place_fucdic(card)
                self._start_play()
            case Play(card):
                self._play(card)

    def _auction_act(self, bid: Bid | None) -> None:
        self.auction.act(bid)
        if not self.auction.finished:
            return
        if self.auction.holder is None:
            self._finish_with_redeal()
            return
        self.declarer, self.bid = self.auction.holder, self.auction.high_bid
        self.phase = Phase.CALL_ACE

    def _call_ace(self, suit: Suit) -> None:
        self.called_suit = suit
        holders = [s for s in range(NUM_PLAYERS) if self.called_ace in self.hands[s]]
        self.partner = holders[0] if holders else self.declarer  # in the cat: alone

        match self.bid.attachment:
            case None | Attachment.HALVES:
                self.phase = Phase.NAME_TRUMPS
            case Attachment.CLUBS:
                self.trumps = Suit.CLUBS
                self.phase = Phase.EXCHANGE
            case Attachment.FLIP:
                self.phase = Phase.FLIP
                self._turn_cat_card()

    def _turn_cat_card(self) -> None:
        self.turned += 1
        if self.cat[self.turned - 1] == self.called_ace:
            self.partner_revealed = True  # everyone now sees the declarer is alone
        if self.turned == CAT_SIZE:
            self._accept_turned_card()

    def _flip(self, accept: bool) -> None:
        if accept:
            self._accept_turned_card()
        else:
            self._turn_cat_card()

    def _accept_turned_card(self) -> None:
        self.trumps = self.cat[self.turned - 1].suit  # a Joker gives no trumps
        self.phase = Phase.EXCHANGE

    def _take_cat(self) -> None:
        self.took_cat = True
        self.hands[self.declarer] = sorted(self.hands[self.declarer] + self.cat, key=sort_key)
        self.phase = Phase.DISCARD

    def _offer_fucdic(self) -> None:
        hand = self.hands[self.declarer]
        if sum(card.suit is self.called_suit for card in hand) <= 1:
            self.phase = Phase.FUCDIC
        else:
            self._start_play()

    def _place_fucdic(self, card: Card) -> None:
        """The real card leaves the hand; a face-down 'zero' of the called suit takes its place."""
        hand = self.hands[self.declarer]
        hand.remove(card)
        hand.insert(0, Card(FUCDIC_RANK, self.called_suit))
        self.hands[self.declarer] = sorted(hand, key=sort_key)
        self.fucdic = card

    def _start_play(self) -> None:
        self.leader = self.declarer
        self.phase = Phase.PLAY

    def _play(self, card: Card) -> None:
        seat = self.to_act
        self.hands[seat].remove(card)
        self.trick.append((seat, card))
        if card == self.called_ace:
            self.partner_revealed = True
        if len(self.trick) < NUM_PLAYERS:
            return

        winner = trick_winner(self.trick, self.trumps)
        self.tricks_won[winner] += 1
        self.tricks.append(self.trick)
        self.trick = []
        self.leader = winner
        if len(self.tricks) == HAND_SIZE:
            self._finish_with_settlement()

    def _finish_with_settlement(self) -> None:
        taken = self.tricks_won[self.declarer]
        if not self.alone:
            taken += self.tricks_won[self.partner]
        self.scores = settle(self.bid, taken, self.declarer, self.partner)
        self.partner_revealed = True
        self.phase = Phase.DONE

    def _finish_with_redeal(self) -> None:
        self.redeal = True
        self.scores = [0] * NUM_PLAYERS
        self.phase = Phase.DONE

    # --- Helpers -------------------------------------------------------------

    def _partner_known_to(self, seat: int) -> int | None:
        if self.partner is None:
            return None
        if self.partner_revealed:
            return self.partner
        if seat == self.partner and not self.alone:
            return self.partner  # they hold the called ace
        if seat == self.declarer and self.alone:
            # Alone by calling their own ace: known at once. Ace in the cat:
            # known only once the declarer has picked it up.
            ace_was_in_cat = self.called_ace in self.cat
            if not ace_was_in_cat or self.took_cat:
                return self.partner
        return None

    @staticmethod
    def _seats_from(first: int) -> list[int]:
        return [(first + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)]
