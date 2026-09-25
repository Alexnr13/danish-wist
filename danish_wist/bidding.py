"""Bids and the one-on-one auction (RULES.md §4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

MIN_LEVEL, MAX_LEVEL = 7, 13
NUM_PLAYERS = 4


class Attachment(Enum):
    FLIP = "flip"
    CLUBS = "clubs"
    HALVES = "halves"

    __hash__ = object.__hash__  # members are unique; quicker than Enum's hash

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Bid:
    level: int
    attachment: Attachment | None = None

    def __str__(self) -> str:
        return f"{self.level} {self.attachment}" if self.attachment else str(self.level)


ALL_BIDS = [
    Bid(level, attachment)
    for level in range(MIN_LEVEL, MAX_LEVEL + 1)
    for attachment in (None, *Attachment)
]


class Auction:
    """A series of duels between the holder of the high bid and a challenger.

    The challenger (`to_act`) must beat the high bid or pass. A bid swaps the
    roles; a pass eliminates the challenger and brings in the next player.
    """

    def __init__(self, forehand: int) -> None:
        self.to_act: int | None = forehand
        self.holder: int | None = None
        self.high_bid: Bid | None = None
        self.history: list[tuple[int, Bid | None]] = []
        self._waiting = [(forehand + i) % NUM_PLAYERS for i in range(1, NUM_PLAYERS)]
        self._used_at_level: set[Attachment] = set()

    @property
    def finished(self) -> bool:
        return self.to_act is None

    def beats_high_bid(self, bid: Bid) -> bool:
        if self.high_bid is None:
            return True
        if bid.level != self.high_bid.level:
            return bid.level > self.high_bid.level
        # Same level: only an attachment not yet used at this level.
        return bid.attachment is not None and bid.attachment not in self._used_at_level

    def legal_bids(self) -> list[Bid]:
        return [bid for bid in ALL_BIDS if self.beats_high_bid(bid)]

    def act(self, bid: Bid | None) -> None:
        """Apply a bid, or a pass when `bid` is None. Assumes the bid is legal."""
        player = self.to_act
        self.history.append((player, bid))
        if bid is None:
            self._bring_in_next_player()
            return

        if self.high_bid is None or bid.level != self.high_bid.level:
            self._used_at_level = set()
        if bid.attachment is not None:
            self._used_at_level.add(bid.attachment)

        previous_holder = self.holder
        self.high_bid, self.holder = bid, player
        if previous_holder is None:
            self._bring_in_next_player()
        else:
            self.to_act = previous_holder

    def _bring_in_next_player(self) -> None:
        self.to_act = self._waiting.pop(0) if self._waiting else None
