from danish_wist.bidding import Attachment, Bid
from danish_wist.scoring import contract_amount, settle, trick_value


def test_trick_values_double_per_level_and_with_attachment():
    assert [trick_value(Bid(level)) for level in range(7, 14)] == [
        10,
        20,
        40,
        80,
        160,
        320,
        640,
    ]
    assert trick_value(Bid(8, Attachment.CLUBS)) == 40
    assert trick_value(Bid(13, Attachment.FLIP)) == 1280


def test_made_contract_scores_every_trick():
    assert contract_amount(Bid(8), 10) == 200
    assert contract_amount(Bid(8), 8) == 160


def test_failed_contract_pays_missing_tricks_at_double():
    assert contract_amount(Bid(8), 6) == -80


def test_two_against_two():
    assert settle(Bid(8), 10, declarer=0, partner=2) == [200, -200, 200, -200]
    assert settle(Bid(8), 6, declarer=0, partner=2) == [-80, 80, -80, 80]


def test_alone_collects_from_all_three():
    assert settle(Bid(8, Attachment.CLUBS), 8, declarer=1, partner=1) == [-320, 960, -320, -320]
    assert settle(Bid(8), 7, declarer=1, partner=1) == [40, -120, 40, 40]
