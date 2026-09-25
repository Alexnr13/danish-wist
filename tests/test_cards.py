import copy
import pickle
from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from danish_wist.cards import JOKER, Card, Suit, full_deck, is_iron_hand, parse_cards


def test_deck_is_52_plus_three_jokers():
    deck = full_deck()
    assert len(deck) == 55
    assert Counter(deck)[JOKER] == 3
    assert len(set(deck)) == 53


def test_parse_and_print_round_trip():
    for text in ["AS", "10H", "2C", "QD", "JK"]:
        assert str(Card.parse(text)) == text
    assert Card.parse("KH") == Card(13, Suit.HEARTS)


def test_iron_hand_has_no_courts_aces_or_jokers():
    assert is_iron_hand(parse_cards("2C 3C 4C 5D 6D 7D 8H 9H 10H 2S 3S 4S 10S"))
    assert not is_iron_hand(parse_cards("2C 3C 4C 5D 6D 7D 8H 9H 10H 2S 3S 4S JS"))
    assert not is_iron_hand(parse_cards("2C 3C 4C 5D 6D 7D 8H 9H 10H 2S 3S 4S JK"))


def test_each_card_is_one_object_even_after_copying_or_pickling():
    ace = Card.parse("AS")
    assert ace is Card(14, Suit.SPADES) and Card(0, None) is JOKER
    assert copy.copy(ace) is ace and copy.deepcopy(ace) is ace
    assert pickle.loads(pickle.dumps([ace, JOKER])) == [ace, JOKER]
    assert pickle.loads(pickle.dumps(ace)) is ace


def test_cards_are_immutable():
    with pytest.raises(FrozenInstanceError):
        Card.parse("AS").rank = 2
