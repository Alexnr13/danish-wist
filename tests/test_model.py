import random

import pytest

torch = pytest.importorskip("torch")

from danish_wist import Deal  # noqa: E402
from danish_wist.bots import RuleBot  # noqa: E402
from learn.arena import play, random_positions  # noqa: E402
from learn.encoding import observe  # noqa: E402
from learn.imitate import accuracy, teacher_samples, train  # noqa: E402
from learn.model import Net, NetAgent, NetConfig, collate, load, save  # noqa: E402

SMALL = NetConfig(width=32, layers=2, heads=2)


def some_views(count: int):
    rng = random.Random(0)
    views = []
    while len(views) < count:
        deal = Deal.new(0, rng)
        while not deal.is_over:
            views.append(deal.view(deal.to_act))
            deal.apply(rng.choice(deal.legal_actions()))
    return views[:count]


def test_padding_does_not_change_the_output():
    torch.manual_seed(0)
    net = Net(SMALL).eval()
    short, long = some_views(1)[0], some_views(40)[-1]
    alone, _ = net(*collate([observe(short)]))
    padded, _ = net(*collate([observe(short), observe(long)]))
    legal = torch.isfinite(alone[0])
    assert torch.allclose(alone[0][legal], padded[0][legal], atol=1e-5)


def test_collate_pads_each_observation_and_marks_its_legal_actions():
    import numpy as np

    from learn.encoding import NUM_ACTIONS, Observation

    observations = [observe(view) for view in some_views(50)[::7]]
    stored = [
        Observation(np.asarray(o.tokens, np.int16), np.asarray(o.legal, np.int16))
        for o in observations
    ]
    for batch in (observations, stored):  # as `observe` makes them, and as training keeps them
        tokens, padding, legal = collate(batch)
        assert tokens.dtype == torch.long and tokens.shape[1] == max(len(o.tokens) for o in batch)
        for i, observation in enumerate(observations):
            n = len(observation.tokens)
            assert tokens[i, :n].tolist() == [list(t) for t in observation.tokens]
            assert not tokens[i, n:].any() and not padding[i, :n].any() and padding[i, n:].all()
            assert legal[i].nonzero().flatten().tolist() == sorted(observation.legal)
        assert legal.shape == (len(batch), NUM_ACTIONS)


def test_the_multi_hot_embedding_is_the_sum_of_lookups():
    torch.manual_seed(2)
    net = Net(SMALL).double()  # in float32 the heavily reused rows' sums differ by platform
    tokens, _, _ = collate([observe(view) for view in some_views(30)])
    weights = [embed.weight for embed in net.embed]
    lookups = sum(embed(tokens[..., i]) for i, embed in enumerate(net.embed))
    assert torch.allclose(net.embed_multi_hot(tokens), lookups, atol=1e-6)
    assert torch.equal(net.embed_tokens(tokens), lookups)  # off Apple's GPU
    by_hot = torch.autograd.grad(net.embed_multi_hot(tokens).square().sum(), weights)
    by_lookup = torch.autograd.grad(lookups.square().sum(), weights)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(by_hot, by_lookup, strict=True))


def test_an_untrained_network_only_plays_legal_moves():
    torch.manual_seed(1)
    agent = NetAgent(Net(SMALL), temperature=1.0, rng=random.Random(1))
    for position in random_positions(5, random.Random(2)):
        assert play(position, [agent] * 4).is_over  # apply() rejects illegal moves


def test_save_and_load_round_trip(tmp_path):
    torch.manual_seed(2)
    net = Net(SMALL).eval()
    save(net, str(tmp_path / "net.pt"))
    inputs = collate([observe(v) for v in some_views(3)])
    assert torch.equal(net(*inputs)[0], load(str(tmp_path / "net.pt")).eval()(*inputs)[0])


def test_the_network_can_learn_to_copy_its_teacher():
    torch.manual_seed(3)
    samples = teacher_samples(RuleBot(), 5, random.Random(3))
    net = Net(SMALL)
    before = accuracy(net, samples)
    train(net, samples, epochs=40, rng=random.Random(3), lr=3e-3, batch_size=32)
    assert accuracy(net, samples) > max(0.9, before)


def test_numpy_inference_matches_pytorch(tmp_path):
    from learn.inference import NumpyAgent, NumpyNet
    from learn.inference import collate as numpy_collate
    from learn.model import export

    torch.manual_seed(4)
    net = Net(SMALL).eval()
    export(net, str(tmp_path / "net.npz"))
    observations = [observe(v) for v in some_views(60)[::7]]  # varied lengths, so padding
    expected_logits, expected_values = net(*collate(observations))
    logits, values = NumpyNet(str(tmp_path / "net.npz"))(*numpy_collate(observations))
    legal = torch.isfinite(expected_logits).numpy()
    assert (legal == (logits > -float("inf"))).all()
    assert abs(logits[legal] - expected_logits.detach().numpy()[legal]).max() < 1e-4
    assert abs(values - expected_values.detach().numpy()).max() < 1e-4

    agent = NumpyAgent(str(tmp_path / "net.npz"), temperature=1.0, rng=random.Random(4))
    assert play(random_positions(1, random.Random(5))[0], [agent] * 4).is_over


def test_numpy_and_pytorch_beliefs_agree(tmp_path):
    from learn.inference import NumpyAgent
    from learn.model import export

    torch.manual_seed(13)
    net = Net(SMALL).eval()
    export(net, str(tmp_path / "net.npz"))
    view = some_views(30)[-1]
    by_torch = NetAgent(net).beliefs(view)
    by_numpy = NumpyAgent(str(tmp_path / "net.npz")).beliefs(view)
    assert len(by_torch) == 52 and all(abs(sum(row) - 1) < 1e-5 for row in by_torch)
    assert (
        max(
            abs(a - b)
            for x, y in zip(by_torch, by_numpy, strict=True)
            for a, b in zip(x, y, strict=True)
        )
        < 1e-5
    )


def test_a_network_loads_from_its_exported_arrays(tmp_path):
    from learn.model import export

    torch.manual_seed(5)
    net = Net(SMALL)
    export(net, str(tmp_path / "net.npz"))
    loaded = load(str(tmp_path / "net.npz"))
    assert loaded.config == net.config
    assert all(
        torch.equal(a, b)
        for a, b in zip(net.state_dict().values(), loaded.state_dict().values(), strict=True)
    )


def test_a_network_plays_the_same_with_numpy_and_with_pytorch(tmp_path):
    from learn.arena import duplicate, make_agent
    from learn.model import export

    torch.manual_seed(6)
    export(Net(SMALL), str(tmp_path / "net.npz"))
    path, rng = str(tmp_path / "net.npz"), random.Random(6)
    by_numpy, by_torch = make_agent(path, rng), make_agent(path, rng, device="cpu")
    assert isinstance(by_torch, NetAgent)
    positions = random_positions(3, random.Random(7))
    field = RuleBot()
    assert (
        duplicate(by_numpy, field, positions).per_deal
        == duplicate(by_torch, field, positions).per_deal
    )


SERVED_GRAPHS_AFTER_CLEARS = """
import torch
from learn.model import Net, NetConfig, _Graphed

torch.manual_seed(0)
net = Net(NetConfig()).cuda().eval()
graphed = _Graphed(net)
shapes = [(rows, length) for length in (96, 112) for rows in range(64, 2049, 64)]
with torch.no_grad():
    for rows, length in shapes:
        tokens = torch.zeros(rows, length, 5, dtype=torch.int32, device="cuda")
        graphed(net, tokens, torch.zeros(rows, length, dtype=torch.bool, device="cuda"))
        torch._C._cuda_clearCublasWorkspaces()  # as each capture of the update's graphs does
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # as `torch.cuda.graph` does before each capture
    for rows, length in shapes:
        tokens = torch.zeros(rows, length, 5, dtype=torch.int32, device="cuda")
        graphed(net, tokens, torch.zeros(rows, length, dtype=torch.bool, device="cuda"))
        torch.cuda.synchronize()
print("replayed", len(graphed.graphs))
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
def test_the_served_graphs_keep_their_workspaces_when_cublas_s_are_cleared():
    """The served networks' graphs (`model._Graphed`) replay after cuBLAS's workspaces are
    cleared and PyTorch's cache emptied, as the graphed update's captures do. One of these 64
    captures warms up on the stream it is captured on (every 32nd new stream from PyTorch's pool
    is `torch.cuda.graph`'s), which once left a graph a workspace outside its pool, and the GPU
    faulted replaying it (TRAINING.md, "The cooldown check"). In a process of its own, since a
    fault ends the process's use of the GPU."""
    import subprocess
    import sys
    from pathlib import Path

    found = subprocess.run(
        [sys.executable, "-c", SERVED_GRAPHS_AFTER_CLEARS],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert found.returncode == 0, found.stderr[-2000:]
    assert found.stdout.split() == ["replayed", "64"]
