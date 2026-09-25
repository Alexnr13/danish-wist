"""Build deals with chosen cards so tests can set up exact situations."""

from danish_wist import Deal, Pass, Phase
from danish_wist.cards import full_deck, parse_cards


def deal_with(hands: dict[int, str] | None = None, cat: str = "", dealer: int = 0) -> Deal:
    """A deal where the given seats hold (at least) the given cards.

    Unspecified cards are filled in from the rest of the deck, in deck order,
    with low spot cards first so they don't upset the chosen situation.
    """
    hands = hands or {}
    remaining = full_deck()
    fixed = {seat: parse_cards(text) for seat, text in hands.items()}
    fixed_cat = parse_cards(cat)
    for card in [c for cards in fixed.values() for c in cards] + fixed_cat:
        remaining.remove(card)
    remaining.sort(key=lambda c: (c.is_joker, c.rank))

    def fill(cards, size):
        cards = list(cards)
        while len(cards) < size:
            cards.append(remaining.pop(0))
        return cards

    full_hands = [fill(fixed.get(seat, []), 13) for seat in range(4)]
    return Deal(dealer, full_hands, fill(fixed_cat, 3))


def auction_won_by(deal: Deal, seat: int, bid) -> None:
    """Everyone passes except `seat`, who makes `bid`."""
    while deal.phase is Phase.IRON_HAND:
        deal.apply(deal.legal_actions()[1])  # decline
    while deal.phase is Phase.AUCTION:
        deal.apply(bid if deal.to_act == seat and bid in deal.legal_actions() else Pass())
