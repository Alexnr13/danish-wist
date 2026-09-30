"""Do the served networks' graphs survive cuBLAS's workspaces being cleared and the cache emptied
(as each capture of the update's graphs does), once a served capture has warmed up on the capture
stream itself (every 32nd new stream from PyTorch's pool is that stream)? Before commit a5be326
the 33rd graph's replay faulted the GPU (Xid 31); since, all 40 replay (fault.txt here).

    python results/rl-007-cool/served_repro.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from learn.model import Net, NetConfig, _Graphed  # noqa: E402

torch.manual_seed(0)
net = Net(NetConfig()).cuda().eval()
graphed = _Graphed(net)
shapes = [
    (rows, length)
    for length in (32, 48, 64, 80, 96)
    for rows in (64, 256, 512, 768, 1024, 1536, 1984, 2048)
]


def call(rows, length):
    tokens = torch.zeros(rows, length, 5, device="cuda", dtype=torch.int32)
    padding = torch.zeros(rows, length, dtype=torch.bool, device="cuda")
    return graphed(net, tokens, padding).sum().item()


with torch.no_grad():
    for shape in shapes:
        call(*shape)
        torch._C._cuda_clearCublasWorkspaces()  # as `learn.selfplay._capture` does
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # as `torch.cuda.graph` does
    print("captured", len(graphed.graphs), flush=True)
    for i, shape in enumerate(shapes):
        call(*shape)
        torch.cuda.synchronize()
        print("replayed", i, shape, flush=True)
print("replayed every graph without a fault")
