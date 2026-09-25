"""The policy and value network (LEARNING.md §3). Requires PyTorch.

A small transformer reads the tokens from `learn.encoding`. Each of a token's
five fields has its own embedding table and the embeddings are summed. A
learned summary token is prepended; its output feeds a policy head over the
fixed action space (masked to the legal actions) and a value head.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from danish_wist.actions import Action
from danish_wist.bidding import NUM_PLAYERS
from danish_wist.game import PlayerView

from .encoding import (
    ACTIONS,
    BELIEF_CARDS,
    MAX_TOKENS,
    NO_SEAT,
    NUM_ACTIONS,
    NUM_CARD_IDS,
    NUM_VALUES,
    Kind,
    Observation,
    observe,
)


@dataclass(frozen=True)
class NetConfig:
    width: int = 128
    layers: int = 4
    heads: int = 4


class Net(nn.Module):
    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or NetConfig()
        width = config.width
        self.embed = nn.ModuleList(
            nn.Embedding(size, width)
            for size in (len(Kind), NUM_CARD_IDS, NO_SEAT + 1, NUM_VALUES, MAX_TOKENS)
        )
        self.summary = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width, config.heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
        )
        self.policy = nn.Linear(width, NUM_ACTIONS)
        self.value = nn.Linear(width, 1)
        # Where each unseen suited card is (see encoding.belief_targets): trained
        # alongside the policy as an auxiliary task, and later used to sample deals.
        self.belief = nn.Linear(width, BELIEF_CARDS * NUM_PLAYERS)

    def forward(
        self, tokens: torch.Tensor, padding: torch.Tensor, legal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens (B, T, 5), padding (B, T) true where padded, legal (B, A).

        Returns masked policy logits (B, A) and values (B,).
        """
        return self.heads(self.summarise(tokens, padding), legal)

    def summarise(self, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        """The summary token's output (B, width), which every head reads."""
        x = sum(embed(tokens[..., i]) for i, embed in enumerate(self.embed))
        x = torch.cat([self.summary.expand(len(x), -1, -1), x], dim=1)
        padding = torch.cat([padding.new_zeros(len(x), 1), padding], dim=1)
        return self.encoder(x, src_key_padding_mask=padding)[:, 0]

    def heads(
        self, summary: torch.Tensor, legal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy(summary).masked_fill(~legal, float("-inf"))
        return logits, self.value(summary).squeeze(-1)

    def beliefs(self, summary: torch.Tensor) -> torch.Tensor:
        """Logits (B, 52, 4) over where each suited card is."""
        return self.belief(summary).view(len(summary), BELIEF_CARDS, NUM_PLAYERS)


def collate(observations: Sequence[Observation]) -> tuple[torch.Tensor, ...]:
    """Pad a batch of observations into tensors: tokens, padding mask, legal mask."""
    length = max(len(o.tokens) for o in observations)
    tokens = torch.zeros(len(observations), length, 5, dtype=torch.long)
    padding = torch.ones(len(observations), length, dtype=torch.bool)
    legal = torch.zeros(len(observations), NUM_ACTIONS, dtype=torch.bool)
    for i, observation in enumerate(observations):
        n = len(observation.tokens)
        tokens[i, :n] = torch.as_tensor(observation.tokens)
        padding[i, :n] = False
        legal[i, observation.legal] = True
    return tokens, padding, legal


class NetAgent:
    """Plays with a network: the most likely legal action, or a sample if `temperature` > 0."""

    def __init__(self, net: Net, temperature: float = 0.0, rng: random.Random | None = None):
        self.net = net.eval()
        self.temperature = temperature
        self.generator = torch.Generator().manual_seed((rng or random.Random()).getrandbits(63))

    def choose(self, view: PlayerView) -> Action:
        return self.choose_batch([view])[0]

    @torch.no_grad()
    def beliefs(self, view: PlayerView) -> list[list[float]]:
        """For each suited card, the probability of each place (see `encoding.belief_targets`)."""
        tokens, padding, _ = collate([observe(view)])
        return torch.softmax(self.net.beliefs(self.net.summarise(tokens, padding)), -1)[0].tolist()

    @torch.no_grad()
    def choose_batch(self, views: Sequence[PlayerView]) -> list[Action]:
        logits, _ = self.net(*collate([observe(view) for view in views]))
        if self.temperature == 0:
            choices = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            choices = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
        return [ACTIONS[i] for i in choices.tolist()]


def save(net: Net, path: str) -> None:
    torch.save({"config": asdict(net.config), "state": net.state_dict()}, path)


def load(path: str) -> Net:
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    net = Net(NetConfig(**checkpoint["config"]))
    missing, unexpected = net.load_state_dict(checkpoint["state"], strict=False)
    # Checkpoints from before the belief head load with a fresh one.
    assert not unexpected and all(k.startswith("belief.") for k in missing), (missing, unexpected)
    return net


def export(net: Net, path: str) -> None:
    """Write the weights as plain arrays for `learn.inference` (no PyTorch needed to play)."""
    arrays = {name: t.detach().cpu().numpy() for name, t in net.state_dict().items()}
    np.savez(path, config=json.dumps(asdict(net.config)), **arrays)
