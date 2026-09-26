import random

from helpers import auction_won_by, deal_with

from danish_wist import Bid, CallAce, Card, Deal, DeclareFucdic, NameTrumps, Pass, Suit, TakeCat
from danish_wist.bots import RuleBot
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


def replayed(deal: Deal, hands, cat, actions: int) -> Deal | None:
    """`deal`'s first actions dealt from other cards, if the same seats can make them."""
    other = Deal(deal.dealer, [list(hand) for hand in hands], list(cat))
    for seat, action in deal.history[:actions]:
        if other.to_act != seat or action not in other.legal_actions():
            return None
        other.apply(action)
    return other


def test_swapping_cards_a_player_cannot_see_never_changes_what_they_see():
    # At random points of finished deals, swap two cards the player to act has
    # never seen (in other hands or the cat) and replay the same actions.
    rng, bot, compared = random.Random(7), RuleBot(), 0
    for number in range(300):
        deal = Deal.new(number % 4, rng)
        while not deal.is_over:
            view = deal.view(deal.to_act)
            deal.apply(bot.choose(view) if number % 2 else rng.choice(view.legal_actions))
        places = {
            c: ("hand", s, i) for s, h in enumerate(deal.initial_hands) for i, c in enumerate(h)
        }
        places |= {c: ("cat", 0, i) for i, c in enumerate(deal.cat)}
        for _ in range(4):
            actions = rng.randrange(len(deal.history))
            here = replayed(deal, deal.initial_hands, deal.cat, actions)
            seat, view = here.to_act, here.view(here.to_act)
            played = {card for trick in [*view.tricks, view.trick] for _, card in trick}
            seen = {*view.hand, *view.turned_cat, *view.discards, *played, view.fucdic}
            unseen = [
                card
                for card, (kind, holder, _) in places.items()
                if card not in seen and (kind, holder) != ("hand", seat) and not card.is_joker
            ]
            if len(unseen) < 2:
                continue
            a, b = rng.sample(unseen, 2)
            if places[a][:2] == places[b][:2]:
                continue  # the same hand: swapping changes nothing
            hands, cat = [list(h) for h in deal.initial_hands], list(deal.cat)
            for card, other in [(a, b), (b, a)]:
                kind, holder, index = places[card]
                (hands[holder] if kind == "hand" else cat)[index] = other
            there = replayed(deal, hands, cat, actions)
            if there is None or there.to_act != seat:
                continue  # the swap made a past action impossible
            compared += 1
            assert encode(there.view(seat)) == encode(view)
            assert there.view(seat).legal_actions == view.legal_actions
    assert compared > 300


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


def test_the_exchange_is_encoded_for_everyone():
    from danish_wist import Discard

    deal = deal_with({1: "AS KS QS JS 10S AH KH QH AC KC QC JC 10C", 3: "AD 5D"})
    auction_won_by(deal, 1, Bid(8))
    deal.apply(CallAce(Suit.DIAMONDS))
    deal.apply(NameTrumps(Suit.SPADES))
    deal.apply(TakeCat(True))
    for card in ["QC", "JC", "10C"]:
        deal.apply(Discard(Card.parse(card)))
    assert (Kind.EXCHANGED, NO_CARD, 3, 0, 0) in encode(deal.view(2))
