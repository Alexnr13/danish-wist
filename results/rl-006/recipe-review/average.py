"""Weight averages of rl-006's saved checkpoints, and the magnets of its resume states at 4000 and
6000, as policies in runs/avg/ (TRAINING.md, "The recipe for a long run").

    python results/rl-006/recipe-review/average.py
"""

import torch

from learn.model import Net, NetConfig, export, save

RUN = "runs/rl-006/checkpoints"


def write(state: dict, config: dict, out: str) -> None:
    net = Net(NetConfig(**config))
    net.load_state_dict(state)
    save(net, out + ".pt")
    export(net, out + ".npz")


def average(paths: list[str], out: str) -> None:
    first = torch.load(paths[0], weights_only=True, map_location="cpu")
    total = {k: v.double() for k, v in first["state"].items()}
    for path in paths[1:]:
        for k, v in torch.load(path, weights_only=True, map_location="cpu")["state"].items():
            total[k] += v.double()
    write({k: (v / len(paths)).float() for k, v in total.items()}, first["config"], out)


config = torch.load(f"{RUN}/policy-6000.pt", weights_only=True)["config"]
for last, count in [(6000, 10), (6000, 20), (4000, 20), (5400, 20)]:
    paths = [f"{RUN}/policy-{last - 10 * i:04d}.pt" for i in range(count)]
    average(paths, f"runs/avg/avg-{last - 10 * (count - 1)}-{last}")
for state, name in [
    ("runs/rl-006-state-6000/state.pt", "magnet-6000"),
    ("runs/rl-006-state-4000/state.pt", "magnet-4000"),
]:
    write(torch.load(state, map_location="cpu")["magnet"], config, f"runs/avg/{name}")
