"""Agents made of parts: one decides the auction, one sets up the contract, one plays.

Against a RuleBot field, the arena's score mixes bidding and card play: a
policy that bids differently changes who declares, the contract and its stake,
and then plays it. Swapping parts with RuleBot takes the score apart
(results/review-2026-09/REVIEW.md §2.2):

    python -m learn.arena --candidate play:runs/rl/policy.npz --field rule
    python -m learn.arena --candidate hybrid:runs/rl/policy.npz,rule,rule --field rule

**`play:<agent>` is the fixed-contract card-play test.** RuleBot bids and sets
up the contract; the agent only plays the cards. The baseline plays the same
deal all-RuleBot, so the contract and its stake are the same in both and the
difference is card play alone. It resolves to about ±3.5 per deal over 2000
deals, against about ±15 for the full arena. `play:search:<.npz>` measures
search the same way.

`hybrid:<auction>,<contract>,<play>` names the three parts: the auction (the
iron hand and bids), the contract's set-up (the call, trumps, Flip, the
exchange, discards, fucdic) and card play.
"""

from __future__ import annotations

from collections.abc import Sequence

from danish_wist.actions import Action
from danish_wist.game import Phase, PlayerView

AUCTION, CONTRACT, PLAY = "auction", "contract", "play"
PARTS = (AUCTION, CONTRACT, PLAY)


def part(phase: Phase) -> str:
    """Which part decides in `phase`."""
    if phase in (Phase.IRON_HAND, Phase.AUCTION):
        return AUCTION
    return PLAY if phase is Phase.PLAY else CONTRACT


class Hybrid:
    """Plays each part with its own agent; each agent gets its decisions in one batch."""

    def __init__(self, auction, contract, play) -> None:
        self.parts = {AUCTION: auction, CONTRACT: contract, PLAY: play}

    def choose(self, view: PlayerView) -> Action:
        return self.choose_batch([view])[0]

    def choose_batch(self, views: Sequence[PlayerView]) -> list[Action]:
        chosen: list[Action | None] = [None] * len(views)
        for name, agent in self.parts.items():
            index = [i for i, view in enumerate(views) if part(view.phase) == name]
            if not index:
                continue
            mine = [views[i] for i in index]
            if hasattr(agent, "choose_batch"):
                actions = agent.choose_batch(mine)
            else:
                actions = [agent.choose(view) for view in mine]
            for i, action in zip(index, actions, strict=True):
                chosen[i] = action
        return chosen
