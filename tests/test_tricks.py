import random

from danish_wist.cards import JOKER, Card, Suit, full_deck, parse_cards
from danish_wist.tricks import legal_plays, trick_winner, winning_cards

AH = Card.parse("AH")


def led(*cards: str) -> list:
    return [(seat, Card.parse(text)) for seat, text in enumerate(cards)]


def test_must_follow_suit():
    hand = parse_cards("2H 9H 3C JK")
    assert legal_plays(hand, led("KH"), None) == parse_cards("2H 9H")


def test_void_player_may_play_anything_including_a_joker():
    hand = parse_cards("3C 4S JK JK")
    assert legal_plays(hand, led("KH"), None) == parse_cards("3C 4S JK")


def test_anything_may_follow_a_led_joker():
    hand = parse_cards("2H 3C")
    assert legal_plays(hand, led("JK"), None) == hand


def test_any_card_may_be_led_including_a_joker_on_the_first_trick():
    hand = parse_cards("2H 3C JK")
    assert legal_plays(hand, [], None) == hand


def test_called_ace_must_be_played_when_another_player_leads_its_suit():
    hand = parse_cards("5H AH 3C")
    assert legal_plays(hand, led("KH"), AH) == [AH]


def test_partner_need_not_lead_the_called_ace():
    hand = parse_cards("5H AH 3C")
    assert legal_plays(hand, [], AH) == hand


def test_called_ace_not_forced_by_other_suits_or_a_joker_lead():
    hand = parse_cards("5H AH 3C")
    assert legal_plays(hand, led("KC"), AH) == parse_cards("3C")
    assert legal_plays(hand, led("JK"), AH) == hand


def test_highest_of_led_suit_wins_without_trumps():
    assert trick_winner(led("5H", "KH", "AS", "2H"), Suit.CLUBS) == 1


def test_highest_trump_wins():
    assert trick_winner(led("AH", "2C", "5C", "KH"), Suit.CLUBS) == 2


def test_no_trumps():
    assert trick_winner(led("5H", "AS", "6H", "2C"), None) == 2


def test_led_joker_wins_even_against_another_joker():
    assert (
        trick_winner(
            [(2, JOKER), (3, Card.parse("AC")), (0, JOKER), (1, Card.parse("2H"))], Suit.CLUBS
        )
        == 2
    )


def test_joker_that_follows_never_wins():
    assert trick_winner(led("2H", "JK", "3H", "4S"), None) == 2


def test_winning_cards_are_those_that_would_take_the_trick():
    rng = random.Random(9)
    deck = full_deck()
    for _ in range(3000):
        rng.shuffle(deck)
        size = rng.randrange(1, 4)
        trick = [(seat, card) for seat, card in enumerate(deck[:size])]
        cards, trumps = deck[size : size + 8], rng.choice([*Suit, None])
        expected = [c for c in cards if trick_winner(trick + [(size, c)], trumps) == size]
        assert winning_cards(cards, trick, trumps) == expected
