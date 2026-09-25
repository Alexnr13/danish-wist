"""Trick values and zero-sum settlement (RULES.md §9)."""

from __future__ import annotations

from .bidding import MIN_LEVEL, NUM_PLAYERS, Bid


def trick_value(bid: Bid) -> int:
    value = 10 * 2 ** (bid.level - MIN_LEVEL)
    return value * 2 if bid.attachment else value


def contract_amount(bid: Bid, tricks_taken: int) -> int:
    """What each defender pays: positive if the contract was made, negative if failed."""
    value = trick_value(bid)
    if tricks_taken >= bid.level:
        return tricks_taken * value
    return -(bid.level - tricks_taken) * 2 * value


def settle(bid: Bid, tricks_taken: int, declarer: int, partner: int) -> list[int]:
    """Score change for each seat. `partner == declarer` means the declarer played alone."""
    amount = contract_amount(bid, tricks_taken)
    declaring_side = {declarer, partner}
    defenders = NUM_PLAYERS - len(declaring_side)
    share = amount * defenders // len(declaring_side)
    return [share if seat in declaring_side else -amount for seat in range(NUM_PLAYERS)]
