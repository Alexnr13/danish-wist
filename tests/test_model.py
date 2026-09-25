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
