"""Play thousands of random deals and check invariants that must always hold."""

import random
from collections import Counter

from danish_wist import Match, Phase
from danish_wist.cards import full_deck


def play_random_deal(match: Match, rng: random.Random):
    deal = match.new_deal()
    while not deal.is_over:
        seat = deal.to_act
        view = deal.view(seat)
        assert list(view.legal_actions) == deal.legal_actions()
        deal.apply(rng.choice(view.legal_actions))
    match.record(deal)
    return deal


def test_random_deals_keep_invariants():
    rng = random.Random(2026)
    match = Match(rng)
    for _ in range(3000):
        deal = play_random_deal(match, rng)
        assert sum(deal.scores) == 0
        if deal.redeal:
            continue
        assert deal.phase is Phase.DONE
        assert len(deal.tricks) == 13 and sum(deal.tricks_won) == 13
        assert all(not hand for hand in deal.hands)
        played = [card for trick in deal.tricks for _, card in trick]
        if deal.fucdic:  # swap the face-down stand-in for the real card
            played = [c for c in played if c.rank != 0 or c.is_joker] + [deal.fucdic]
        unplayed = deal.discards if deal.took_cat else deal.cat
        assert Counter(played + unplayed) == Counter(full_deck())
    assert sum(match.scores) == 0
    assert match.deals_played > 2500
