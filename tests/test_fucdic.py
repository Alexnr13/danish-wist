"""Fucdic (RULES.md §6)."""

from helpers import auction_won_by, deal_with

from danish_wist import (
    Bid,
    CallAce,
    Card,
    DeclareFucdic,
    Discard,
    NameTrumps,
    Phase,
    Play,
    Suit,
    TakeCat,
)
from danish_wist.tricks import trick_winner

ZERO_OF_DIAMONDS = Card.parse("0D")


def declarer_calls_diamonds(hand: str, partner_hand: str = "AD 5D", cat: str = "") -> object:
    deal = deal_with({1: hand, 3: partner_hand}, cat=cat)
    auction_won_by(deal, 1, Bid(8))
    deal.apply(CallAce(Suit.DIAMONDS))
    deal.apply(NameTrumps(Suit.SPADES))
    return deal


VOID_IN_DIAMONDS = "AS KS QS JS 10S AH KH QH AC KC QC JC 10C"


def test_offered_when_void_and_any_card_may_be_chosen():
    deal = declarer_calls_diamonds(VOID_IN_DIAMONDS)
    deal.apply(TakeCat(False))
    assert deal.phase is Phase.FUCDIC
    choices = {a.card for a in deal.legal_actions()}
    assert choices == {None, *deal.hands[1]}


def test_with_one_card_of_the_suit_that_card_must_be_the_fucdic():
    deal = declarer_calls_diamonds("9D KS QS JS 10S AH KH QH AC KC QC JC 10C")
    deal.apply(TakeCat(False))
    assert deal.legal_actions() == [DeclareFucdic(None), DeclareFucdic(Card.parse("9D"))]


def test_not_offered_with_two_cards_of_the_suit():
    deal = declarer_calls_diamonds("9D 8D QS JS 10S AH KH QH AC KC QC JC 10C")
    deal.apply(TakeCat(False))
    assert deal.phase is Phase.PLAY


def test_judged_on_the_hand_after_the_exchange():
    deal = declarer_calls_diamonds(VOID_IN_DIAMONDS, cat="9D 8D 7D")
    deal.apply(TakeCat(True))
    for card in ["QC", "JC", "10C"]:
        deal.apply(Discard(Card.parse(card)))
    assert deal.phase is Phase.PLAY  # now holds three diamonds, so no fucdic


def test_fucdic_is_a_hidden_zero_that_brings_out_the_called_ace():
    deal = declarer_calls_diamonds(VOID_IN_DIAMONDS)
    deal.apply(TakeCat(False))
    deal.apply(DeclareFucdic(Card.parse("10C")))

    assert ZERO_OF_DIAMONDS in deal.hands[1] and Card.parse("10C") not in deal.hands[1]
    assert deal.view(1).fucdic == Card.parse("10C")
    assert deal.view(2).fucdic is None and deal.view(2).fucdic_declared

    deal.apply(Play(ZERO_OF_DIAMONDS))  # the declarer leads it
    deal.apply(deal.legal_actions()[0])  # seat 2
    assert deal.legal_actions() == [Play(Card.parse("AD"))]  # partner must play the ace
    assert deal.view(0).trick[0] == (1, ZERO_OF_DIAMONDS)  # never revealed


def test_fucdic_is_the_lowest_card_of_its_suit():
    def trick(*cards):
        return [(seat, Card.parse(c)) for seat, c in enumerate(cards)]

    assert trick_winner(trick("0D", "2D", "3H", "4H"), None) == 1
    assert trick_winner(trick("0D", "2H", "3H", "4H"), None) == 0  # led and unopposed: it wins
