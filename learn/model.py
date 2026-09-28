"""The policy and value network (LEARNING.md §3). Requires PyTorch.

A small transformer reads the tokens from `learn.encoding`. Each of a token's
five fields has its own embedding table and the embeddings are summed. A
learned summary token is prepended; its output feeds a policy head over the
fixed action space (masked to the legal actions) and a value head.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import accumulate, chain

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
        """The summary token's output (B, width), which every head reads.

        The encoder's output is needed for the summary token alone, so its last
        layer works out only that: the summary attends to every token as before,
        but the other tokens' attention and feed-forward, which nothing reads,
        are skipped. The same numbers as the whole encoder, for a fifth less work.
        """
        x = self.embed_tokens(tokens)
        x = torch.cat([self.summary.expand(len(x), -1, -1), x], dim=1)
        padding = torch.cat([padding.new_zeros(len(x), 1), padding], dim=1)
        *layers, last = self.encoder.layers
        for layer in layers:
            x = layer(x, src_key_padding_mask=padding)
        normed = last.norm1(x)  # the layers are pre-norm (norm_first)
        attended = last.self_attn(
            normed[:, :1], normed, normed, key_padding_mask=padding, need_weights=False
        )[0]
        x = x[:, 0] + attended[:, 0]
        x = x + last.linear2(last.activation(last.linear1(last.norm2(x))))
        return self.encoder.norm(x)

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


@dataclass
class Question:
    """Decisions for a network, as arrays: what a served agent asks (`learn.runner`).

    The answer (`Networks`) is the logits of each row's legal actions, in the
    order of `legal`.
    """

    tokens: np.ndarray  # (B, T, 5)
    lengths: np.ndarray  # (B,) each row's tokens
    legal: np.ndarray  # every row's legal actions (indices into ACTIONS), row after row
    counts: np.ndarray  # (B,) how many legal actions each row has

    @classmethod
    def of(cls, observations: Sequence[Observation]) -> Question:
        tokens, lengths = inference.pad([o.tokens for o in observations], dtype=np.int16)
        counts = np.fromiter(map(len, (o.legal for o in observations)), np.int64, len(lengths))
        legal = np.fromiter(chain.from_iterable(o.legal for o in observations), np.int16)
        return cls(tokens, lengths, legal, counts)

    def observations(self) -> list[Observation]:
        """Each row's observation, as views of these arrays."""
        ends = np.cumsum(self.counts)
        rows = zip(self.tokens, self.lengths, self.counts, ends, strict=True)
        return [
            Observation(tokens[:length], self.legal[end - count : end])
            for tokens, length, count, end in rows
        ]

    def rows(self) -> np.ndarray:
        """The row of each legal action in `legal`."""
        return np.repeat(np.arange(len(self.counts)), self.counts)

    def logits(self, answer: np.ndarray) -> torch.Tensor:
        """The answer as logits (B, NUM_ACTIONS), -inf for the actions that are not legal."""
        logits = np.full((len(self.counts), NUM_ACTIONS), -np.inf, dtype=np.float32)
        logits[self.rows(), self.legal] = answer
        return torch.from_numpy(logits)


class Networks:
    """Networks by name, run in one process for the agents of every runner worker.

    Each call answers a round's questions (`Question`) from all the workers:
    the questions to each network go through it as one batch, and one copy
    back brings every answer. With one process owning the GPU, a network sees
    a batch per round for all the workers instead of one per worker and agent,
    and the GPU is not switched between the workers' processes (PERFORMANCE.md).
    """

    def __init__(self, nets: dict[str, Net], device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.nets = {name: net.to(self.device).eval() for name, net in nets.items()}
        graphed = self.device.type == "cuda"
        self._forward = _Graphed(next(iter(self.nets.values()))) if graphed else _logits

    def load(self, name: str, state: dict) -> None:
        self.nets[name].load_state_dict(state)

    @torch.no_grad()
    def __call__(self, entries: list[tuple[str, Question]]) -> list[np.ndarray]:
        order = sorted(range(len(entries)), key=lambda i: list(self.nets).index(entries[i][0]))
        questions = [entries[i][1] for i in order]
        sizes = [len(q.counts) for q in questions]
        tokens = np.zeros((sum(sizes), max(q.tokens.shape[1] for q in questions), 5), np.int16)
        start = 0
        for question, size in zip(questions, sizes, strict=True):
            tokens[start : start + size, : question.tokens.shape[1]] = question.tokens
            start += size
        lengths = np.concatenate([q.lengths for q in questions])
        starts = np.cumsum([0, *sizes])
        rows = np.concatenate([q.rows() + s for q, s in zip(questions, starts, strict=False)])
        legal = np.concatenate([q.legal for q in questions]).astype(np.int64)
        tokens, lengths, rows, legal = (
            torch.from_numpy(a).to(self.device) for a in (tokens, lengths, rows, legal)
        )
        padding = torch.arange(tokens.shape[1], device=self.device) >= lengths[:, None]
        tokens = tokens.int()
        logits = torch.empty(len(lengths), NUM_ACTIONS, device=self.device)
        first = 0
        for name, count in _runs([entries[i][0] for i in order]):
            batch = slice(starts[first], starts[first + count])
            logits[batch] = self._forward(self.nets[name], tokens[batch], padding[batch])
            first += count
        found = logits[rows, legal].cpu().numpy()
        answers: list[np.ndarray] = [np.empty(0)] * len(entries)
        split = np.cumsum([len(q.legal) for q in questions])[:-1]
        for i, answer in zip(order, np.split(found, split), strict=True):
            answers[i] = answer
        return answers


def _logits(net: Net, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    """The policy's logits, all of them; on an NVIDIA GPU with the trunk in bfloat16 (as the
    update has it: `learn.selfplay._summarise`)."""
    if tokens.device.type != "cuda":
        return net.policy(net.summarise(tokens, padding))
    with torch.autocast("cuda", torch.bfloat16):
        summary = net.summarise(tokens, padding)
    return net.policy(summary.float())


class _Graphed:
    """`_logits` replayed from CUDA graphs, for any network shaped like `net`.

    A graph is captured for each shape of batch (rows and tokens, rounded up)
    on a copy of `net`, and each call copies the network's weights into that
    copy first: a few launches instead of the hundred or so kernels of a
    forward pass, which is most of what a round's small batches cost. The
    graphs share one pool of memory, so each call's result must be used (as
    `Networks` does, copying it) before the next call.
    """

    MOST_ROWS = 2048  # larger batches keep the GPU busy anyway, and run as they are

    def __init__(self, net: Net) -> None:
        self.net = copy.deepcopy(net).eval()
        self.weights = list(self.net.parameters())
        self.graphs: dict[tuple[int, int], tuple] = {}
        self.pool = None

    def __call__(self, net: Net, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        rows, length = padding.shape
        if rows > self.MOST_ROWS:
            return _logits(net, tokens, padding)
        shape = (_bucket(rows), -(-length // 16) * 16)
        if shape not in self.graphs:
            self.graphs[shape] = self._capture(*shape)
        graph, static_tokens, static_padding, logits = self.graphs[shape]
        torch._foreach_copy_(self.weights, list(net.parameters()))
        static_tokens[:rows, :length] = tokens
        static_padding[:rows] = True  # beyond `length` too: those tokens are not there
        static_padding[:rows, :length] = padding
        graph.replay()
        return logits[:rows]

    def _capture(self, rows: int, length: int) -> tuple:
        device = self.weights[0].device
        tokens = torch.zeros(rows, length, 5, dtype=torch.int32, device=device)
        padding = torch.ones(rows, length, dtype=torch.bool, device=device)
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):  # warm up (libraries' workspaces) before capturing
            for _ in range(2):
                _logits(self.net, tokens, padding)
        torch.cuda.current_stream(device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            logits = _logits(self.net, tokens, padding)
        self.pool = graph.pool()
        return graph, tokens, padding, logits


def _bucket(rows: int) -> int:
    """Rows rounded up to a power of two from 8 to 64, then to a multiple of 64: few shapes to
    capture, and a small batch costs about the same at 8 rows as at 1."""
    return max(8, 1 << (rows - 1).bit_length()) if rows <= 64 else -(-rows // 64) * 64


def _runs(names: list[str]) -> list[tuple[str, int]]:
    """Each name in `names` with how many times it comes in a row."""
    runs: list[tuple[str, int]] = []
    for name in names:
        if runs and runs[-1][0] == name:
            runs[-1] = (name, runs[-1][1] + 1)
        else:
            runs.append((name, 1))
    return runs


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
