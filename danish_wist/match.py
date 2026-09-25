"""A session of consecutive deals with running scores (RULES.md §3)."""

from __future__ import annotations

import random

from .bidding import NUM_PLAYERS
from .game import Deal


class Match:
    def __init__(self, rng: random.Random | None = None, dealer: int = 0) -> None:
        self.rng = rng or random.Random()
        self.dealer = dealer
        self.scores = [0] * NUM_PLAYERS
        self.deals_played = 0

    def new_deal(self) -> Deal:
        return Deal.new(self.dealer, self.rng)

    def record(self, deal: Deal) -> None:
        """Add a finished deal's scores. A redeal keeps the same dealer."""
        assert deal.is_over
        if deal.redeal:
            return
        self.scores = [
            total + change for total, change in zip(self.scores, deal.scores, strict=True)
        ]
        self.deals_played += 1
        self.dealer = (self.dealer + 1) % NUM_PLAYERS

    @property
    def sets_completed(self) -> int:
        """Deals are played in sets of four so that everyone deals once."""
        return self.deals_played // NUM_PLAYERS
