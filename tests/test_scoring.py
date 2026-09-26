import random

from helpers import auction_won_by, deal_with

from danish_wist import CallAce, NameTrumps, Suit, TakeCat
from danish_wist.bidding import Attachment, Bid
from danish_wist.scoring import contract_amount, settle, trick_value


def test_trick_values_double_per_level_and_with_attachment():
    assert [trick_value(Bid(level), Suit.HEARTS) for level in range(7, 14)] == [
        10,
        20,
        40,
        80,
        160,
        320,
        640,
    ]
    assert trick_value(Bid(8, Attachment.CLUBS), Suit.CLUBS) == 40
    assert trick_value(Bid(13, Attachment.FLIP), None) == 1280


def test_made_contract_scores_every_trick():
    assert contract_amount(Bid(8), Suit.HEARTS, 10) == 200
    assert contract_amount(Bid(8), Suit.HEARTS, 8) == 160


def test_failed_contract_pays_missing_tricks_at_double():
    assert contract_amount(Bid(8), Suit.HEARTS, 6) == -80


def test_two_against_two():
    assert settle(Bid(8), Suit.HEARTS, 10, declarer=0, partner=2) == [200, -200, 200, -200]
    assert settle(Bid(8), Suit.HEARTS, 6, declarer=0, partner=2) == [-80, 80, -80, 80]


def test_alone_collects_from_all_three():
    clubs = settle(Bid(8, Attachment.CLUBS), Suit.CLUBS, 8, declarer=1, partner=1)
    assert clubs == [-320, 960, -320, -320]
    assert settle(Bid(8), Suit.HEARTS, 7, declarer=1, partner=1) == [40, -120, 40, 40]


def play_out(deal, rng):
    while not deal.is_over:
        deal.apply(rng.choice(deal.legal_actions()))
    return deal


def test_a_finished_deal_pays_for_the_tricks_of_declarer_and_partner():
    for seed in range(8):
        deal = deal_with({1: "KH QH AS KS QS JS", 3: "AH 5H"})
        auction_won_by(deal, 1, Bid(8))
        deal.apply(CallAce(Suit.HEARTS))
        play_out(deal, random.Random(seed))
        amount = contract_amount(Bid(8), deal.trumps, deal.tricks_won[1] + deal.tricks_won[3])
        assert deal.scores == [-amount, amount, -amount, amount]


def test_calling_the_ace_in_the_cat_plays_alone_for_triple_even_if_the_cat_is_declined():
    for seed in range(8):
        deal = deal_with({1: "KS QS JS"}, cat="AH")
        auction_won_by(deal, 1, Bid(8))
        deal.apply(CallAce(Suit.HEARTS))
        deal.apply(NameTrumps(Suit.SPADES))
        deal.apply(TakeCat(False))
        play_out(deal, random.Random(seed))
        amount = contract_amount(Bid(8), deal.trumps, deal.tricks_won[1])
        assert deal.alone and deal.scores == [-amount, 3 * amount, -amount, -amount]


def test_clubs_trumps_double_a_plain_contract_as_if_it_were_clubs():
    assert trick_value(Bid(9), Suit.CLUBS) == 80
    assert trick_value(Bid(9, Attachment.CLUBS), Suit.CLUBS) == 80
    assert trick_value(Bid(9), Suit.HEARTS) == 40


def test_clubs_trumps_never_double_an_attachment_twice():
    assert trick_value(Bid(9, Attachment.HALVES), Suit.CLUBS) == 80
    assert trick_value(Bid(9, Attachment.FLIP), Suit.CLUBS) == 80


def test_a_plain_contract_played_in_clubs_settles_at_the_clubs_trick_value():
    for seed in range(8):
        deal = deal_with({1: "AC KC QC JC", 3: "AH"})
        auction_won_by(deal, 1, Bid(8))
        deal.apply(CallAce(Suit.HEARTS))
        deal.apply(NameTrumps(Suit.CLUBS))
        play_out(deal, random.Random(seed))
        taken = deal.tricks_won[1] + deal.tricks_won[3]
        amount = taken * 40 if taken >= 8 else -(8 - taken) * 2 * 40
        assert deal.scores == [-amount, amount, -amount, amount]
