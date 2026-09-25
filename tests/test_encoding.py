import random

from helpers import auction_won_by, deal_with

from danish_wist import Bid, CallAce, Card, Deal, DeclareFucdic, NameTrumps, Pass, Suit, TakeCat
from danish_wist.cards import full_deck
from learn.encoding import (
    ACTIONS,
    MAX_TOKENS,
    NO_CARD,
    NO_SEAT,
    NUM_ACTIONS,
    NUM_BID_IDS,
    NUM_CARD_IDS,
    Kind,
    action_index,
    card_id,
    encode,
    legal_mask,
    observe,
)


def random_views(deals: int, seed: int):
    """Every view of the player to act, through `deals` random deals."""
    rng = random.Random(seed)
    for _ in range(deals):
        deal = Deal.new(rng.randrange(4), rng)
        while not deal.is_over:
            view = deal.view(deal.to_act)
            yield view
            deal.apply(rng.choice(view.legal_actions))


def test_action_indices_are_a_bijection():
    assert len(set(ACTIONS)) == NUM_ACTIONS
    assert all(action_index(action) == i for i, action in enumerate(ACTIONS))


def test_every_card_has_a_distinct_id():
    ids = {card_id(card) for card in set(full_deck())}
    assert len(ids) == 53 and NO_CARD not in ids


def test_every_legal_action_is_indexed_and_masked():
    for view in random_views(200, seed=1):
        mask = legal_mask(view)
        assert sum(mask) == len(view.legal_actions)
        assert [ACTIONS[i] for i in observe(view).legal] == list(view.legal_actions)


def test_tokens_stay_within_their_ranges():
    for view in random_views(200, seed=2):
        tokens = encode(view)
        assert len(tokens) <= MAX_TOKENS
        for kind, card, seat, value, position in tokens:
            assert kind in list(Kind) and kind != Kind.PAD
            assert 0 <= card < NUM_CARD_IDS
            assert 0 <= seat <= NO_SEAT
            assert 0 <= value < NUM_BID_IDS and 0 <= position < MAX_TOKENS


def test_the_viewer_is_always_seat_zero():
    for view in random_views(50, seed=3):
        tokens = encode(view)
        plays = [p for trick in (*view.tricks, view.trick) for p in trick]
        for kind, count in [
            (Kind.PLAY, sum(seat == view.seat for seat, _ in plays)),
            (Kind.BID, sum(seat == view.seat for seat, _ in view.auction)),
        ]:
            assert sum(t[0] == kind and t[2] == 0 for t in tokens) == count


def test_hidden_cards_do_not_change_the_encoding():
    # Seat 1 holds the same hand in both deals; everyone else's cards differ.
    mine = "AS KS QS JS 10S AH KH QH AC KC QC JC 10C"
    first = deal_with({1: mine}, cat="2D 3D 4D")
    second = deal_with({1: mine}, cat="JK JK JK")
    for deal in (first, second):
        auction_won_by(deal, 1, Bid(9))
        deal.apply(CallAce(Suit.DIAMONDS))
    assert first.hands[2] != second.hands[2]
    assert encode(first.view(1)) == encode(second.view(1))
    assert encode(first.view(3)) != encode(first.view(1))


def test_only_the_declarer_sees_the_real_fucdic_card():
    deal = deal_with({1: "AS KS QS JS 10S AH KH QH AC KC QC JC 10C", 3: "AD 5D"})
    auction_won_by(deal, 1, Bid(8))
    deal.apply(CallAce(Suit.DIAMONDS))
    deal.apply(NameTrumps(Suit.SPADES))
    deal.apply(TakeCat(False))
    deal.apply(DeclareFucdic(Card.parse("10C")))
    fucdic = {
        seat: [t for t in encode(deal.view(seat)) if t[0] == Kind.FUCDIC] for seat in range(4)
    }
    assert fucdic[1] == [(Kind.FUCDIC, card_id(Card.parse("10C")), 0, 0, 0)]
    assert fucdic[2] == [(Kind.FUCDIC, NO_CARD, 3, 0, 0)]  # declarer is 3 seats on from 2


def test_passes_are_encoded():
    deal = deal_with({0: "AC", 1: "AS", 2: "AD", 3: "AH"})
    deal.apply(Pass())
    assert (Kind.BID, NO_CARD, 3, 0, 0) in encode(deal.view(2))
