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
from itertools import accumulate

import numpy as np
import torch
from torch import nn

from danish_wist.actions import Action
from danish_wist.bidding import NUM_PLAYERS
from danish_wist.game import PlayerView

from . import inference
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
    value_bins: int = 1  # 1: one value; more: logits over value bins (the self-play critic)


class Net(nn.Module):
    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or NetConfig()
        width = config.width
        sizes = (len(Kind), NUM_CARD_IDS, NO_SEAT + 1, NUM_VALUES, MAX_TOKENS)
        self.embed = nn.ModuleList(nn.Embedding(size, width) for size in sizes)
        self.offsets = list(accumulate(sizes[:-1], initial=0))  # each field's first row
        self.summary = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width, config.heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
        )
        self.policy = nn.Linear(width, NUM_ACTIONS)
        self.value = nn.Linear(width, config.value_bins)
        if config.value_bins > 1:  # start from even odds on every bin
            nn.init.zeros_(self.value.weight)
            nn.init.zeros_(self.value.bias)
        # Where each unseen suited card is (see encoding.belief_targets): trained
        # alongside the policy as an auxiliary task, and later used to sample deals.
        self.belief = nn.Linear(width, BELIEF_CARDS * NUM_PLAYERS)

    def forward(
        self, tokens: torch.Tensor, padding: torch.Tensor, legal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens (B, T, 5), padding (B, T) true where padded, legal (B, A).

        Returns masked policy logits (B, A) and values (B,), or value logits
        (B, value_bins) when there are several bins.
        """
        return self.heads(self.summarise(tokens, padding), legal)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """The sum of each field's embedding (B, T, width)."""
        if tokens.device.type == "mps":
            return self.embed_multi_hot(tokens)
        return sum(embed(tokens[..., i]) for i, embed in enumerate(self.embed))

    def embed_multi_hot(self, tokens: torch.Tensor) -> torch.Tensor:
        """The same sum as one multi-hot product, for Apple's GPU.

        There the lookups' backward pass is about 10x slower, because a batch
        uses only a few rows of each table over and over. Elsewhere the product
        costs more than the lookups: its one-hot rows are twice the width.
        """
        weight = torch.cat([embed.weight for embed in self.embed])
        rows = tokens + tokens.new_tensor(self.offsets)
        hot = weight.new_zeros(*tokens.shape[:2], len(weight)).scatter_(-1, rows, 1.0)
        return hot @ weight

    def summarise(self, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        """The summary token's output (B, width), which every head reads."""
        x = self.embed_tokens(tokens)
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
    return tuple(torch.from_numpy(a) for a in inference.collate(observations))


def device_of(net: nn.Module) -> torch.device:
    return next(net.parameters()).device


def inputs(observations: Sequence[Observation], device: torch.device) -> list[torch.Tensor]:
    """`collate`d and on `device`."""
    return [t.to(device, non_blocking=True) for t in collate(observations)]


class NetAgent:
    """Plays with a network: the most likely legal action, or a sample if `temperature` > 0.

    The network runs wherever it is (move it to a GPU first to use one); sampling
    stays on the CPU, so a seeded agent draws the same moves on any device.
    """

    def __init__(self, net: Net, temperature: float = 0.0, rng: random.Random | None = None):
        self.net = net.eval()
        self.temperature = temperature
        self.generator = torch.Generator().manual_seed((rng or random.Random()).getrandbits(63))

    def choose(self, view: PlayerView) -> Action:
        return self.choose_batch([view])[0]

    @torch.no_grad()
    def beliefs(self, view: PlayerView) -> list[list[float]]:
        """For each suited card, the probability of each place (see `encoding.belief_targets`)."""
        tokens, padding, _ = inputs([observe(view)], device_of(self.net))
        return torch.softmax(self.net.beliefs(self.net.summarise(tokens, padding)), -1)[0].tolist()

    @torch.no_grad()
    def choose_batch(self, views: Sequence[PlayerView]) -> list[Action]:
        logits, _ = self.net(*inputs([observe(view) for view in views], device_of(self.net)))
        logits = logits.cpu()
        if self.temperature == 0:
            choices = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            choices = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
        return [ACTIONS[i] for i in choices.tolist()]


def save(net: Net, path: str) -> None:
    torch.save({"config": asdict(net.config), "state": net.state_dict()}, path)


def load(path: str) -> Net:
    """A network from a checkpoint (`save`) or from its exported arrays (`export`, `.npz`)."""
    if str(path).endswith(".npz"):
        with np.load(path) as data:
            config = json.loads(str(data["config"]))
            state = {name: torch.from_numpy(data[name]) for name in data.files if name != "config"}
    else:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        config, state = checkpoint["config"], checkpoint["state"]
    net = Net(NetConfig(**config))
    missing, unexpected = net.load_state_dict(state, strict=False)
    # Checkpoints from before the belief head load with a fresh one.
    assert not unexpected and all(k.startswith("belief.") for k in missing), (missing, unexpected)
    return net


def export(net: Net, path: str) -> None:
    """Write the weights as plain arrays for `learn.inference` (no PyTorch needed to play)."""
    arrays = {name: t.detach().cpu().numpy() for name, t in net.state_dict().items()}
    np.savez(path, config=json.dumps(asdict(net.config)), **arrays)
