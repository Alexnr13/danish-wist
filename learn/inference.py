"""Run a trained network with NumPy alone: no PyTorch needed to play.

Training saves a PyTorch checkpoint; `learn.model.export` turns it into a
`.npz` file of plain arrays that this module reads. The forward pass here
mirrors `learn.model.Net` exactly (a pre-norm transformer encoder), and a test
checks the two agree.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence

import numpy as np

from danish_wist.actions import Action
from danish_wist.bidding import NUM_PLAYERS
from danish_wist.game import PlayerView

from .encoding import ACTIONS, NUM_ACTIONS, Observation, observe


class NumpyNet:
    def __init__(self, path: str) -> None:
        with np.load(path) as data:
            self.weights = {name: data[name] for name in data.files if name != "config"}
            config = json.loads(str(data["config"]))
        self.heads, self.layers = config["heads"], config["layers"]

    def __call__(self, tokens, padding, legal) -> tuple[np.ndarray, np.ndarray]:
        """tokens (B, T, 5) ints, padding (B, T) bools, legal (B, A) bools -> logits, values."""
        w = self.weights
        summary = self.summarise(tokens, padding)
        logits = np.where(legal, _linear(summary, w, "policy"), -np.inf)
        return logits, _linear(summary, w, "value")[:, 0]

    def beliefs(self, tokens, padding) -> np.ndarray:
        """Probabilities (B, 52, 4) of where each suited card is."""
        logits = _linear(self.summarise(tokens, padding), self.weights, "belief")
        logits = logits.reshape(len(logits), -1, NUM_PLAYERS)
        scaled = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return scaled / scaled.sum(axis=-1, keepdims=True)

    def summarise(self, tokens, padding) -> np.ndarray:
        w = self.weights
        x = sum(w[f"embed.{i}.weight"][tokens[..., i]] for i in range(5))
        x = np.concatenate([np.broadcast_to(w["summary"], (len(x), 1, x.shape[-1])), x], axis=1)
        padding = np.concatenate([np.zeros((len(x), 1), dtype=bool), padding], axis=1)
        for i in range(self.layers):
            p = f"encoder.layers.{i}."
            x = x + self._attention(_layer_norm(x, w, p + "norm1"), padding, p + "self_attn.")
            hidden = np.maximum(_linear(_layer_norm(x, w, p + "norm2"), w, p + "linear1"), 0)
            x = x + _linear(hidden, w, p + "linear2")
        return _layer_norm(x, w, "encoder.norm")[:, 0]

    def _attention(self, x, padding, prefix):
        w = self.weights
        batch, length, width = x.shape
        size = width // self.heads
        qkv = x @ w[prefix + "in_proj_weight"].T + w[prefix + "in_proj_bias"]
        q, k, v = (
            part.reshape(batch, length, self.heads, size).transpose(0, 2, 1, 3)
            for part in np.split(qkv, 3, axis=-1)
        )
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(size)
        scores = np.where(padding[:, None, None, :], -np.inf, scores)
        scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attended = (scores / scores.sum(axis=-1, keepdims=True)) @ v
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
        return _linear(attended, w, prefix + "out_proj")


def _linear(x, w, name):
    return x @ w[name + ".weight"].T + w[name + ".bias"]


def _layer_norm(x, w, name, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * w[name + ".weight"] + w[name + ".bias"]


def collate(observations: Sequence[Observation]) -> tuple[np.ndarray, ...]:
    """Pad a batch of observations into arrays: tokens (B, T, 5), padding mask, legal mask.

    Tokens may be lists of tuples (from `observe`) or arrays (as training
    stores them). Everything is placed by a few whole-array operations: copied
    row by row, the batch cost about 100 times more than this, and more than
    the network itself on a GPU.
    """
    tokens, lengths = pad([o.tokens for o in observations])
    padding = np.arange(tokens.shape[1]) >= lengths[:, None]
    return tokens, padding, legal_mask([o.legal for o in observations])


def pad(sequences: Sequence, dtype: type = np.int64) -> tuple[np.ndarray, np.ndarray]:
    """Token sequences (lists of tuples, or arrays) as one array (B, T, 5), T the longest
    one's length, and their lengths."""
    batch = len(sequences)
    # np.repeat takes counts as np.intp: int64 here, int32 in the browser (Pyodide).
    lengths = np.fromiter(map(len, sequences), dtype=np.intp, count=batch)
    if isinstance(sequences[0], np.ndarray):
        flat = np.concatenate(sequences)
    else:
        flat = np.array([t for tokens in sequences for t in tokens], dtype=dtype)
    rows = np.repeat(np.arange(batch), lengths)
    places = np.arange(len(flat)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    tokens = np.zeros((batch, lengths.max(), 5), dtype=dtype)
    tokens[rows, places] = flat
    return tokens, lengths


def legal_mask(legal: Sequence) -> np.ndarray:
    """(B, NUM_ACTIONS), true at each row's legal action indices."""
    mask = np.zeros((len(legal), NUM_ACTIONS), dtype=bool)
    counts = np.fromiter(map(len, legal), dtype=np.intp, count=len(legal))
    choices = np.concatenate([np.asarray(indices, dtype=np.int64) for indices in legal])
    mask[np.repeat(np.arange(len(legal)), counts), choices] = True
    return mask


class NumpyAgent:
    """Plays with an exported network: the most likely legal action, or a sample."""

    def __init__(self, path: str, temperature: float = 0.0, rng: random.Random | None = None):
        self.net = NumpyNet(path)
        self.temperature = temperature
        self.rng = np.random.default_rng((rng or random.Random()).getrandbits(63))

    def choose(self, view: PlayerView) -> Action:
        return self.choose_batch([view])[0]

    def beliefs(self, view: PlayerView) -> list[list[float]]:
        """For each suited card, the probability of each place (see `encoding.belief_targets`)."""
        tokens, padding, _ = collate([observe(view)])
        return self.net.beliefs(tokens, padding)[0].tolist()

    def choose_batch(self, views: Sequence[PlayerView]) -> list[Action]:
        logits, _ = self.net(*collate([observe(view) for view in views]))
        if self.temperature == 0:
            return [ACTIONS[i] for i in logits.argmax(axis=-1)]
        scaled = np.exp((logits - logits.max(axis=-1, keepdims=True)) / self.temperature)
        probs = scaled / scaled.sum(axis=-1, keepdims=True)
        return [ACTIONS[self.rng.choice(NUM_ACTIONS, p=p)] for p in probs]
