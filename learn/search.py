"""Search at play time: think ahead in deals sampled from what a player knows.

At each decision with a real choice, `SearchAgent` samples `worlds` complete
deals consistent with its view (`learn.worlds`), plays every legal action to
the end of the deal in each of them with a fast `rollout` agent in all four
seats, and picks the action with the best average score for its seat. Every
action is tried in the same worlds, so the comparison is fair.

Because the rollouts are played by a policy that sees only its own views,
this avoids the "strategy fusion" error of solving each sampled deal with
every card face up. More worlds make a stronger (and slower) player, which is
one way to get bots of different strengths.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence

from danish_wist.actions import Action
from danish_wist.bots import Agent
from danish_wist.game import Deal, Phase, PlayerView

from .worlds import sample_worlds


def finish(deals: Sequence[Deal], agent) -> None:
    """Play every deal to the end with `agent` in all seats, in batches."""
    active = [deal for deal in deals if not deal.is_over]
    while active:
        views = [deal.view(deal.to_act) for deal in active]
        if hasattr(agent, "choose_batch"):
            actions = agent.choose_batch(views)
        else:
            actions = [agent.choose(view) for view in views]
        for deal, action in zip(active, actions, strict=True):
            deal.apply(action)
        active = [deal for deal in active if not deal.is_over]


class SearchAgent:
    """Searches in `phases` (card play by default); elsewhere plays as `rollout`.

    With a `belief` model (anything with `beliefs(view)`, such as `NetAgent` or
    `learn.inference.NumpyAgent`), worlds are sampled in proportion to where it
    thinks the unseen cards are.

    Bidding is left to the policy by default: with a couple of dozen bids and a
    handful of noisy worlds, the bid that looks best is usually the one whose
    estimate was luckiest, so search overbids ("winner's curse").
    """

    def __init__(
        self,
        rollout: Agent,
        worlds: int = 8,
        rng: random.Random | None = None,
        phases: frozenset[Phase] = frozenset({Phase.PLAY}),
        belief=None,
    ):
        self.rollout = rollout
        self.worlds = worlds
        self.rng = rng or random.Random()
        self.phases = phases
        self.belief = belief

    def choose(self, view: PlayerView) -> Action:
        legal = view.legal_actions
        if len(legal) == 1:
            return legal[0]
        if view.phase not in self.phases:
            return self.rollout.choose(view)
        belief = self.belief.beliefs(view) if self.belief is not None else None
        worlds = sample_worlds(view, self.worlds, self.rng, belief=belief)
        if not worlds:
            return self.rollout.choose(view)  # nothing consistent found: fall back
        games, tried = [], []
        for action in legal:
            for world in worlds:
                game = copy.deepcopy(world)
                game.apply(action)
                games.append(game)
                tried.append(action)
        finish(games, self.rollout)
        totals = dict.fromkeys(legal, 0)
        for game, action in zip(games, tried, strict=True):
            totals[action] += game.scores[view.seat]
        return max(legal, key=totals.__getitem__)

    def choose_batch(self, views: Sequence[PlayerView]) -> list[Action]:
        return [self.choose(view) for view in views]
