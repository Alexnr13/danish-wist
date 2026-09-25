import pytest
from helpers import auction_won_by, deal_with

from danish_wist import (
    Attachment,
    Bid,
    CallAce,
    Card,
    DeclareIronHand,
    Discard,
    FlipChoice,
    IllegalActionError,
    NameTrumps,
    Pass,
    Phase,
    Play,
    Suit,
    TakeCat,
)

AH = Card.parse("AH")


def test_iron_hand_redeals():
    deal = deal_with({1: "2C 3C 4C 5C 6D 7D 8D 9H 10H 2S 3S 4S 5S"})
    assert deal.phase is Phase.IRON_HAND and deal.to_act == 1
    deal.apply(DeclareIronHand(True))
    assert deal.is_over and deal.redeal and deal.scores == [0, 0, 0, 0]


def test_all_pass_redeals():
    deal = deal_with({0: "AC", 1: "AS", 2: "AD", 3: "AH"})  # no iron hands
    for _ in range(4):
        deal.apply(Pass())
    assert deal.is_over and deal.redeal


def test_plain_contract_partner_and_first_lead():
    deal = deal_with({1: "KH QH AS KS QS JS", 3: "AH 5H"})
    auction_won_by(deal, 1, Bid(8))
    assert deal.declarer == 1 and deal.phase is Phase.CALL_ACE
    deal.apply(CallAce(Suit.HEARTS))
    assert deal.partner == 3 and not deal.alone

    assert NameTrumps(Suit.HEARTS) not in deal.legal_actions()  # not the called suit
    deal.apply(NameTrumps(Suit.SPADES))
    deal.apply(TakeCat(False))

    assert deal.phase is Phase.PLAY and deal.to_act == 1  # the declarer leads
    deal.apply(Play(Card.parse("KH")))
    deal.apply(deal.legal_actions()[0])  # seat 2
    assert deal.to_act == 3 and deal.legal_actions() == [Play(AH)]


def test_clubs_contract_cannot_call_the_ace_of_clubs():
    deal = deal_with({1: "AS AH"})
    auction_won_by(deal, 1, Bid(9, Attachment.CLUBS))
    assert CallAce(Suit.CLUBS) not in deal.legal_actions()
    deal.apply(CallAce(Suit.DIAMONDS))
    assert deal.trumps is Suit.CLUBS and deal.phase is Phase.EXCHANGE


def test_halves_partner_names_trumps_and_is_revealed():
    deal = deal_with({1: "KS", 2: "AH"})
    auction_won_by(deal, 1, Bid(9, Attachment.HALVES))
    deal.apply(CallAce(Suit.HEARTS))
    assert deal.phase is Phase.NAME_TRUMPS and deal.to_act == 2
    assert NameTrumps(Suit.HEARTS) not in deal.legal_actions()
    assert deal.view(0).partner is None
    deal.apply(NameTrumps(Suit.DIAMONDS))
    assert deal.view(0).partner == 2


def test_halves_with_called_ace_in_cat_declarer_names_trumps_alone():
    deal = deal_with({1: "KS"}, cat="AH")
    auction_won_by(deal, 1, Bid(9, Attachment.HALVES))
    deal.apply(CallAce(Suit.HEARTS))
    assert deal.alone and deal.to_act == 1


def test_calling_your_own_ace_plays_alone():
    deal = deal_with({1: "AH KS"})
    auction_won_by(deal, 1, Bid(8))
    deal.apply(CallAce(Suit.HEARTS))
    assert deal.alone
    assert deal.view(1).partner == 1  # the declarer knows
    assert deal.view(2).partner is None  # nobody else does yet


def test_flip_accepts_first_card_then_exchange_is_optional():
    deal = deal_with({1: "KS"}, cat="QD 5C JK")
    auction_won_by(deal, 1, Bid(9, Attachment.FLIP))
    deal.apply(CallAce(Suit.HEARTS))
    assert deal.phase is Phase.FLIP and deal.view(3).turned_cat == (Card.parse("QD"),)
    deal.apply(FlipChoice(True))
    assert deal.trumps is Suit.DIAMONDS
    assert deal.phase is Phase.EXCHANGE
    deal.apply(TakeCat(False))
    assert deal.phase is Phase.PLAY and len(deal.hands[1]) == 13


def test_flip_third_card_is_forced_and_joker_means_no_trumps():
    deal = deal_with({1: "KS"}, cat="QD 5C JK")
    auction_won_by(deal, 1, Bid(9, Attachment.FLIP))
    deal.apply(CallAce(Suit.HEARTS))
    deal.apply(FlipChoice(False))
    deal.apply(FlipChoice(False))
    assert deal.trumps is None and deal.phase is Phase.EXCHANGE


def test_flip_may_make_the_called_suit_trumps():
    deal = deal_with({1: "KS", 2: "AH"}, cat="QH 5C 2D")
    auction_won_by(deal, 1, Bid(9, Attachment.FLIP))
    deal.apply(CallAce(Suit.HEARTS))
    deal.apply(FlipChoice(True))
    assert deal.trumps is Suit.HEARTS


def test_exchange_takes_all_three_and_discards_three():
    deal = deal_with({1: "KS"}, cat="AD AC AS")
    auction_won_by(deal, 1, Bid(8))
    deal.apply(CallAce(Suit.HEARTS))
    deal.apply(NameTrumps(Suit.SPADES))
    deal.apply(TakeCat(True))
    for _ in range(3):
        deal.apply(Discard(deal.hands[1][0]))
    assert len(deal.hands[1]) == 13 and len(deal.view(1).discards) == 3
    assert deal.view(2).discards == ()
    assert deal.phase is Phase.PLAY


def test_illegal_action_is_rejected():
    deal = deal_with({1: "AS"})
    with pytest.raises(IllegalActionError):
        deal.apply(Play(Card.parse("AS")))


def test_views_hide_other_hands():
    deal = deal_with({1: "AS"})
    view = deal.view(2)
    assert Card.parse("AS") not in view.hand and view.legal_actions == ()


def test_turning_up_the_called_ace_in_flip_reveals_the_declarer_is_alone():
    deal = deal_with({1: "KS"}, cat="AH 5C 2D")
    auction_won_by(deal, 1, Bid(9, Attachment.FLIP))
    deal.apply(CallAce(Suit.HEARTS))
    assert deal.view(0).partner == 1


def test_called_ace_in_cat_is_unknown_to_declarer_until_taken():
    deal = deal_with({1: "KS"}, cat="AH 5C 2D")
    auction_won_by(deal, 1, Bid(8))
    deal.apply(CallAce(Suit.HEARTS))
    deal.apply(NameTrumps(Suit.SPADES))
    assert deal.view(1).partner is None
    deal.apply(TakeCat(True))
    assert deal.view(1).partner == 1 and deal.view(0).partner is None


def test_partner_is_known_to_themselves_and_revealed_when_the_ace_is_played():
    deal = deal_with({1: "KH QH AS KS QS JS", 3: "AH 5H"})
    auction_won_by(deal, 1, Bid(8))
    deal.apply(CallAce(Suit.HEARTS))
    deal.apply(NameTrumps(Suit.SPADES))
    deal.apply(TakeCat(False))
    assert [deal.view(s).partner for s in range(4)] == [None, None, None, 3]
    deal.apply(Play(Card.parse("KH")))
    deal.apply(deal.legal_actions()[0])
    deal.apply(Play(AH))
    assert [deal.view(s).partner for s in range(4)] == [3, 3, 3, 3]
