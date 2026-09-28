"""Card play by one-ply search in sampled worlds, judged by the oracle critic (REVIEW.md T3.1).

At each card play with a choice, `CriticSearch` samples deals consistent
with its view (`learn.worlds`), plays each legal card in each of them, and
asks the self-play critic, which sees every card, what each result is worth
to its seat. It plays the card with the best mean over the worlds. The critic
values a position for play by the policy from there on, by every player on
their own information, so this is a one-step policy improvement rather than
"perfect-information Monte Carlo" with its strategy fusion.

The critic learned only positions where its own seat is about to act. With
`reply=True` the other seats first answer the card with the policy, and the
critic judges the position at the seat's next decision (or the final score);
without it, the position straight after the card.

    python -m learn.arena --candidate play:critic:runs/rl/checkpoints/policy-0100.pt \\
        --field rule --worlds 100

The critic is found beside the policy: `critic-0100.pt` for `policy-0100.pt`,
`critic.pt` for `policy.pt`. Everything the networks do for all the views,
worlds and cards of a batch goes to the GPU together.
"""

from __future__ import annotations

import pickle
import random
from collections.abc import Sequence
from pathlib import Path

import torch

from danish_wist.actions import Action
from danish_wist.game import Deal, Phase, PlayerView

from .encoding import encode_oracle
from .model import Net, NetAgent, load
from .worlds import sample_worlds

GROUP = 8  # views searched together: about 8 x 7 cards x 100 worlds deals at once
CHUNK = 2048  # rows per forward pass


def critic_beside(policy: str) -> str:
    """The critic saved with a policy: critic-0100.pt beside policy-0100.pt."""
    path = Path(policy)
    return str(path.with_name(path.name.replace("policy", "critic", 1)).with_suffix(".pt"))


class CriticSearch:
    def __init__(
        self,
        policy: Net,
        critic: Net,
        worlds: int = 100,
        rng: random.Random | None = None,
        reply: bool = False,
    ) -> None:
        self.policy = NetAgent(policy)
        self.critic = critic.eval()
        self.worlds = worlds
        self.rng = rng or random.Random()
        self.reply = reply

    @classmethod
    def load(cls, policy: str, device: str, **options) -> CriticSearch:
        return cls(load(policy).to(device), load(critic_beside(policy)).to(device), **options)

    def choose(self, view: PlayerView) -> Action:
        return self.choose_batch([view])[0]

    def choose_batch(self, views: Sequence[PlayerView]) -> list[Action]:
        searching = [
            i
            for i, view in enumerate(views)
            if view.phase is Phase.PLAY and len(view.legal_actions) > 1
        ]
        chosen: list[Action | None] = [None] * len(views)
        for start in range(0, len(searching), GROUP):
            self._search(views, searching[start : start + GROUP], chosen)
        rest = [i for i, action in enumerate(chosen) if action is None]
        if rest:  # no search, or no world found
            actions = self.policy.choose_batch([views[i] for i in rest])
            for i, action in zip(rest, actions, strict=True):
                chosen[i] = action
        return chosen

    def _search(self, views: Sequence[PlayerView], group: list[int], chosen: list) -> None:
        """Search the views at `group`'s indices together; put their cards in `chosen`."""
        searches = []  # (the view's index, the view, each card tried in each world)
        for i in group:
            worlds = sample_worlds(views[i], self.worlds, self.rng)
            if worlds:
                searches.append((i, views[i], self._tried(views[i], worlds)))
        if not searches:
            return
        deals = [deal for _, _, tried in searches for deal in tried]
        seats = [view.seat for _, view, tried in searches for _ in tried]
        if self.reply:
            self._reply(deals, seats)
        values = self._values(deals, seats)
        start = 0
        for i, view, tried in searches:
            per_card = values[start : start + len(tried)].view(len(view.legal_actions), -1)
            chosen[i] = view.legal_actions[int(per_card.mean(1).argmax())]
            start += len(tried)

    @staticmethod
    def _tried(view: PlayerView, worlds: list[Deal]) -> list[Deal]:
        """Each legal card played in each world, card by card.

        Each world is copied by unpickling it: several times quicker than a deep copy.
        """
        pickled = [pickle.dumps(world, pickle.HIGHEST_PROTOCOL) for world in worlds]
        tried = []
        for action in view.legal_actions:
            for world in pickled:
                deal = pickle.loads(world)
                deal.apply(action)
                tried.append(deal)
        return tried

    def _reply(self, deals: list[Deal], seats: list[int]) -> None:
        """Play the other seats with the policy until each deal is back to its seat, or over."""
        pairs = list(zip(deals, seats, strict=True))
        while waiting := [d for d, seat in pairs if not d.is_over and d.to_act != seat]:
            for start in range(0, len(waiting), CHUNK):
                part = waiting[start : start + CHUNK]
                actions = self.policy.choose_batch([deal.view(deal.to_act) for deal in part])
                for deal, action in zip(part, actions, strict=True):
                    deal.apply(action)

    @torch.no_grad()
    def _values(self, deals: list[Deal], seats: list[int]) -> torch.Tensor:
        """What each deal is worth to its seat: the score if it is over, else the critic's view."""
        from .selfplay import _critic_values  # training code: only imported when searching

        pairs = zip(deals, seats, strict=True)
        values = torch.tensor([float(d.scores[s]) if d.is_over else 0.0 for d, s in pairs])
        going = [i for i, deal in enumerate(deals) if not deal.is_over]
        if going:
            oracles = [encode_oracle(deals[i], seats[i]) for i in going]
            values[going] = _critic_values(self.critic, oracles, chunk=CHUNK)
        return values
