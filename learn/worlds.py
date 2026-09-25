"""Deal out the cards a player cannot see, consistently with what they have seen.

Search at play time (LEARNING.md §3, step 5) needs *worlds*: complete deals
that agree with everything one seat knows. `sample_worlds(view, n, rng)`
builds them from the `PlayerView` alone, so nothing hidden leaks in:

1. Work out which cards are unseen and where they could be: each other
   player's hand, the cat or discards, and a declared fucdic's real card.
   Each place's size is known, and so are some constraints: suits a player
   has shown out of, the declarer having no other card of the called suit
   after fucdic, and a revealed partner holding the called ace.
2. Deal the unseen cards into those places at random, respecting the constraints.
3. Rebuild the deal from the start and replay the public history through the
   engine, filling in the hidden choices (the discards, the fucdic card). The
   engine rejects anything the rules forbid, such as a player who held the
   called ace but did not play it when its suit was led.
4. Keep the world only if it shows the seat *exactly* the view it had.
"""

from __future__ import annotations

import random
from collections import Counter

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
from danish_wist.bidding import NUM_PLAYERS
from danish_wist.cards import FUCDIC_RANK, Card, ace_of, full_deck
from danish_wist.game import CAT_SIZE, HAND_SIZE, Deal, IllegalActionError, Phase, PlayerView

OUT = "out"  # the untaken cat, or the discards
FUCDIC = "fucdic"  # the real card under a declared fucdic


def sample_worlds(
    view: PlayerView, count: int, rng: random.Random, max_tries: int = 50
) -> list[Deal]:
    """Up to `count` deals consistent with `view`, each at the moment of `view`."""
    worlds: list[Deal] = []
    for _ in range(count * max_tries):
        world = _try_world(view, rng)
        if world is not None:
            worlds.append(world)
            if len(worlds) == count:
                break
    return worlds


def _is_stand_in(card: Card) -> bool:
    return card.rank == FUCDIC_RANK and not card.is_joker


def _try_world(view: PlayerView, rng: random.Random) -> Deal | None:
    places = _deal_unseen(view, rng)
    if places is None:
        return None
    try:
        deal = _rebuild(view, places, rng)
    except (IllegalActionError, AssertionError, ValueError):
        return None
    if deal is None or deal.view(view.seat) != view:
        return None
    return deal


# --- 1 and 2: dealing the unseen cards ---------------------------------------


def _played(view: PlayerView) -> list[tuple[int, Card]]:
    return [played for trick in (*view.tricks, view.trick) for played in trick]


def _voids(view: PlayerView) -> dict[int, set]:
    """Suits each player has shown out of: they cannot hold any now."""
    voids: dict[int, set] = {seat: set() for seat in range(NUM_PLAYERS)}
    for trick in (*view.tricks, view.trick):
        if not trick or trick[0][1].is_joker:
            continue
        led = trick[0][1]
        for seat, card in trick[1:]:
            if card.suit is not led.suit:
                voids[seat].add(led.suit)
    return voids


def _deal_unseen(view: PlayerView, rng: random.Random) -> dict | None:
    """Randomly place every unseen card; None if the constraints could not be met."""
    me, declarer = view.seat, view.declarer
    played = _played(view)
    seen = Counter(c for c in view.hand if not _is_stand_in(c))
    seen.update(c for _, c in played if not _is_stand_in(c))
    seen.update(view.discards)
    known: dict = {}  # cards known to be in a place
    turned = Counter(view.turned_cat)
    if not view.took_cat:
        seen.update(turned)  # still in the cat
    elif declarer != me:
        # Turned cards the declarer picked up and has not played are known to be in their hand.
        kept = turned - Counter(c for s, c in played if s == declarer)
        known[declarer] = list(kept.elements())
        seen.update(kept)
    if view.fucdic is not None:
        seen[view.fucdic] += 1
    unseen = list((Counter(full_deck()) - seen).elements())

    # How many unseen cards each place holds.
    capacity: dict = {}
    for seat in range(NUM_PLAYERS):
        if seat == me:
            continue
        held = HAND_SIZE - sum(s == seat for s, _ in played)
        if seat == declarer and view.fucdic_declared:
            stand_in_played = any(s == seat and _is_stand_in(c) for s, c in played)
            held -= 0 if stand_in_played else 1
        held -= len(known.get(seat, []))
        capacity[seat] = held
    if view.phase is Phase.DISCARD and declarer != me:
        return None  # only the declarer decides while discarding
    if view.took_cat:
        capacity[OUT] = 0 if declarer == me else CAT_SIZE
    else:
        capacity[OUT] = CAT_SIZE - len(view.turned_cat)
    if view.fucdic_declared and declarer != me:
        capacity[FUCDIC] = 1
    if sum(capacity.values()) != len(unseen):
        return None

    voids = _voids(view)
    if view.fucdic_declared:
        voids[declarer].add(view.called_suit)  # no real card of the called suit is left

    def allowed(card: Card) -> list:
        return [
            place
            for place, room in capacity.items()
            if room > 0 and not (isinstance(place, int) and card.suit in voids[place])
        ]

    placed: dict = {place: list(known.get(place, [])) for place in capacity}
    ace = ace_of(view.called_suit) if view.called_suit else None
    partner = view.partner
    if ace in unseen and partner is not None and partner not in (me, declarer):
        unseen.remove(ace)  # a revealed partner who has not played it still holds it
        if capacity.get(partner, 0) == 0:
            return None
        placed[partner].append(ace)
        capacity[partner] -= 1

    rng.shuffle(unseen)
    unseen.sort(key=lambda c: len(allowed(c)))  # most constrained first
    for card in unseen:
        options = allowed(card)
        if not options:
            return None
        place = rng.choices(options, weights=[capacity[p] for p in options])[0]
        placed[place].append(card)
        capacity[place] -= 1
    return placed


# --- 3: rebuilding the deal ----------------------------------------------------


def _rebuild(view: PlayerView, places: dict, rng: random.Random) -> Deal | None:
    me, declarer = view.seat, view.declarer
    played = _played(view)
    fucdic = view.fucdic if declarer == me else (places.get(FUCDIC) or [None])[0]

    # Each seat's hand now, as real cards (a fucdic stand-in becomes its real card).
    now = {seat: list(places.get(seat, [])) for seat in range(NUM_PLAYERS) if seat != me}
    now[me] = [c for c in view.hand if not _is_stand_in(c)]
    if view.fucdic_declared:
        now[declarer].append(fucdic)
    hands = {s: now[s] + [c for p, c in played if p == s and not _is_stand_in(c)] for s in now}

    # The cat, in order: turned cards first. Undo the exchange for the declarer.
    if view.took_cat:
        # Which of the declarer's cards came from the cat never matters later,
        # so any of them will do for the unturned ones.
        rest = _pick(hands[declarer], view.turned_cat, rng)
        cat = list(view.turned_cat) + rest
        for card in cat:
            hands[declarer].remove(card)
        discards = list(view.discards) if declarer == me else places[OUT]
        hands[declarer] += discards
    else:
        discards = list(view.discards)  # a declarer part-way through putting cards down
        if discards:
            hands[declarer] += discards
        rest = list(places.get(OUT, []))
        rng.shuffle(rest)
        cat = list(view.turned_cat) + rest
    if len(cat) != CAT_SIZE or any(len(h) != HAND_SIZE for h in hands.values()):
        return None

    deal = Deal(view.dealer, [hands[s] for s in range(NUM_PLAYERS)], cat)
    script = _Script(view, discards, fucdic)
    while not script.reached(deal):
        deal.apply(script.next_action(deal))
    return deal


def _pick(hand: list[Card], turned: tuple[Card, ...], rng: random.Random) -> list[Card]:
    """The cat's unturned cards: any of the declarer's other cards."""
    others = list(hand)
    for card in turned:
        others.remove(card)
    return rng.sample(others, CAT_SIZE - len(turned))


class _Script:
    """The actions that lead from the deal to the moment of the view."""

    def __init__(self, view: PlayerView, discards: list[Card], fucdic: Card | None) -> None:
        self.view = view
        self.bids = [bid for _, bid in view.auction]
        self.discards = list(discards)
        self.fucdic = fucdic
        self.plays = [card for _, card in _played(view)]
        if view.seat == view.declarer or not view.took_cat:
            self.discard_count = len(view.discards)
        else:
            self.discard_count = CAT_SIZE

    def reached(self, deal: Deal) -> bool:
        view = self.view
        return (
            deal.phase is view.phase
            and deal.to_act == view.to_act
            and len(deal.auction.history) == len(view.auction)
            and deal.turned == len(view.turned_cat)
            and len(deal.discards) == self.discard_count
            and sum(map(len, deal.tricks)) + len(deal.trick) == len(self.plays)
        )

    def next_action(self, deal: Deal) -> Action:
        view = self.view
        match deal.phase:
            case Phase.IRON_HAND:
                return DeclareIronHand(False)  # they played on, so any iron hand was kept
            case Phase.AUCTION:
                bid = self.bids[len(deal.auction.history)]
                return Pass() if bid is None else bid
            case Phase.CALL_ACE:
                return CallAce(view.called_suit)
            case Phase.NAME_TRUMPS:
                return NameTrumps(view.trumps)
            case Phase.FLIP:
                return FlipChoice(deal.turned == len(view.turned_cat))
            case Phase.EXCHANGE:  # still putting cards down means the exchange was chosen
                return TakeCat(view.took_cat or view.phase is Phase.DISCARD)
            case Phase.DISCARD:
                return Discard(self.discards[len(deal.discards)])
            case Phase.FUCDIC:
                return DeclareFucdic(self.fucdic if view.fucdic_declared else None)
            case Phase.PLAY:
                return Play(self.plays[sum(map(len, deal.tricks)) + len(deal.trick)])
        raise ValueError(f"nothing to replay in {deal.phase}")
