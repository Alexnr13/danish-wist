"""How much does a policy's belief head know about the unseen cards? By phase.

    python -m learn.beliefs runs/rl-004d/checkpoints/policy-0010.pt
    python -m learn.beliefs runs/rl-004d/checkpoints/policy-0010.pt --field rule

Plays deals with the policy greedily (in every seat, or in one seat among
`--field`) and, at each of its real decisions (forced moves left out, as in
training), scores its belief head: the cross-entropy of where each unseen
suited card really is (`encoding.belief_targets`), per card, grouped by
phase. Against it stands the **prior**, what `learn.worlds` samples from
without beliefs: each place in proportion to the unseen cards it holds (at
the auction about 1.28 nats: 13, 13 and 13 cards in the other hands, 3 in the
cat). A head that has learned something scores well below the prior, and
should do so most late in play, when voids and the cards played say most
(REVIEW.md T1.5; training logs the same cross-entropy as `belief_loss`).
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from functools import partial

import torch

from danish_wist.bidding import NUM_PLAYERS
from danish_wist.game import Phase, PlayerView

from .arena import add_device_argument, parse_with_device, random_positions
from .encoding import ACTIONS, NOT_HIDDEN, belief_targets, observe
from .model import device_of, inputs, load
from .runner import Decision, Runner

GROUPS = ("auction", "contract set-up", "tricks 1-4", "tricks 5-8", "tricks 9-13")


def group(view: PlayerView) -> str:
    if view.phase in (Phase.IRON_HAND, Phase.AUCTION):
        return GROUPS[0]
    if view.phase is not Phase.PLAY:
        return GROUPS[1]
    trick = len(view.tricks) + 1
    return GROUPS[2] if trick <= 4 else GROUPS[3] if trick <= 8 else GROUPS[4]


def prior_log_odds(targets: torch.Tensor) -> torch.Tensor:
    """Log-probabilities (B, places) of the prior: each place by how many unseen cards it holds.

    `targets` is (B, 52), as `belief_targets` gives: a place, or NOT_HIDDEN.
    """
    hidden = targets != NOT_HIDDEN
    places = torch.nn.functional.one_hot(targets.clamp(min=0), NUM_PLAYERS) * hidden[..., None]
    counts = places.sum(1).float()
    return (counts / counts.sum(-1, keepdim=True)).log()


class Probe:
    """Plays greedily with the network, and adds up its beliefs' and the prior's losses."""

    def __init__(self, path: str, device: str) -> None:
        self.net = load(path).to(device).eval()
        # group -> [decisions, unseen cards, belief cross-entropy, prior cross-entropy]
        self.totals: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0, 0.0])

    @torch.no_grad()
    def choose_decisions(self, decisions: list[Decision]) -> list:
        tokens, padding, legal = inputs([observe(d.view) for d in decisions], device_of(self.net))
        summary = self.net.summarise(tokens, padding)
        choices = self.net.heads(summary, legal)[0].argmax(-1).tolist()
        beliefs = torch.log_softmax(self.net.beliefs(summary), -1).cpu()  # (B, 52, 4)
        targets = torch.tensor([belief_targets(d.deal, d.seat) for d in decisions])
        prior = prior_log_odds(targets)
        for i, decision in enumerate(decisions):
            hidden = targets[i] != NOT_HIDDEN
            if len(decision.view.legal_actions) == 1 or not hidden.any():
                continue
            places = targets[i][hidden]
            found = beliefs[i][hidden].gather(1, places[:, None]).sum().item()
            totals = self.totals[group(decision.view)]
            totals[0] += 1
            totals[1] += len(places)
            totals[2] -= found
            totals[3] -= prior[i][places].sum().item()
        return [ACTIONS[choice] for choice in choices]

    def report(self) -> dict[str, list[float]]:
        return dict(self.totals)


def _probe_agents(worker: int, path: str, field: str | None, device: str) -> dict:
    from .arena import make_agent

    agents = {"probe": Probe(path, device)}
    if field is not None:
        agents["field"] = make_agent(field, random.Random(worker), device=device)
    return agents


def _nothing(game, deal, agents) -> None:
    return None


def measure(
    path: str, deals: int, field: str | None = None, seed: int = 0, workers: int = 1, device="cpu"
) -> dict[str, list[float]]:
    """Totals by group: decisions, unseen cards, and the two cross-entropies summed over cards."""
    games = []
    for position in random_positions(deals, random.Random(seed)):
        if field is None:
            games.append((position, ["probe"] * NUM_PLAYERS))
        else:
            for seat in range(NUM_PLAYERS):
                lineup = ["field"] * NUM_PLAYERS
                lineup[seat] = "probe"
                games.append((position, lineup))
    make = partial(_probe_agents, path=path, field=field, device=device)
    with Runner(make, workers=workers, games_in_flight=256) as runner:
        for _ in runner.play(games, _nothing):
            pass
        reports = runner.broadcast("probe", "report")
    totals: dict[str, list[float]] = {}
    for report in reports:
        for name, values in report.items():
            before = totals.get(name, [0, 0, 0.0, 0.0])
            totals[name] = [a + b for a, b in zip(before, values, strict=True)]
    return totals


def table(totals: dict[str, list[float]]) -> str:
    lines = [f"{'phase':<16}{'decisions':>10}{'unseen':>8}{'beliefs':>9}{'prior':>8}{'gain':>8}"]
    rows = [(name, totals[name]) for name in GROUPS if name in totals]
    rows.append(("all", [sum(v[k] for _, v in rows) for k in range(4)]))
    for name, (count, cards, belief, prior) in rows:
        gain = 1 - belief / prior if prior else math.nan
        lines.append(
            f"{name:<16}{count:>10,.0f}{cards / count:>8.1f}"
            f"{belief / cards:>9.3f}{prior / cards:>8.3f}{gain:>8.1%}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="The belief head's loss by phase.")
    parser.add_argument("policy", help="a network (.pt or .npz)")
    parser.add_argument("--field", help="play in one seat among this agent, not self-play")
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    add_device_argument(parser)
    args = parse_with_device(parser)
    device = args.device or "cpu"
    totals = measure(args.policy, args.deals, args.field, args.seed, args.workers, device)
    where = f"among {args.field}" if args.field else "in self-play"
    print(f"{args.policy} {where}, {args.deals} deals: cross-entropy per unseen card (nats)")
    print("(gain: how much less than the prior's)")
    print(table(totals))


if __name__ == "__main__":
    main()
