from danish_wist.bidding import Attachment, Auction, Bid

FLIP, CLUBS, HALVES = Attachment.FLIP, Attachment.CLUBS, Attachment.HALVES


def test_higher_level_beats_any_attachment():
    auction = Auction(forehand=1)
    auction.act(Bid(9, FLIP))
    assert Bid(10) in auction.legal_bids()
    assert Bid(9) not in auction.legal_bids()


def test_each_attachment_once_per_level_and_later_wins():
    # The example from RULES.md: 9 Flip, 9 Clubs, then only 9 Halves or 10+.
    auction = Auction(forehand=1)
    auction.act(Bid(9, FLIP))
    auction.act(Bid(9, CLUBS))
    legal = auction.legal_bids()
    assert Bid(9, HALVES) in legal
    assert Bid(9, FLIP) not in legal
    assert Bid(9, CLUBS) not in legal
    assert min(b.level for b in legal if b.attachment is None) == 10


def test_attachments_reset_at_a_new_level():
    auction = Auction(forehand=1)
    auction.act(Bid(8, FLIP))
    auction.act(Bid(9))
    assert Bid(9, FLIP) in auction.legal_bids()


def test_duels_until_a_pass_then_next_player_enters():
    auction = Auction(forehand=1)
    auction.act(Bid(8))  # seat 1
    assert auction.to_act == 2
    auction.act(Bid(9))  # seat 2 overcalls
    assert auction.to_act == 1  # seat 1 may answer at once
    auction.act(None)  # seat 1 passes and is out
    assert auction.to_act == 3
    auction.act(None)
    assert auction.to_act == 0
    auction.act(Bid(10))
    assert auction.to_act == 2
    auction.act(None)
    assert auction.finished
    assert (auction.holder, auction.high_bid) == (0, Bid(10))


def test_forehand_pass_lets_the_next_player_open():
    auction = Auction(forehand=1)
    auction.act(None)
    assert auction.to_act == 2
    auction.act(Bid(7))
    assert auction.to_act == 3


def test_all_pass_has_no_winner():
    auction = Auction(forehand=1)
    for _ in range(4):
        auction.act(None)
    assert auction.finished and auction.holder is None
