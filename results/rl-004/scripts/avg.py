"""Average the weights of several policy checkpoints (same architecture) and export an .npz.

usage: python results/rl-004/scripts/avg.py <out.npz> <policy-a.pt> <policy-b.pt> ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repository

import torch  # noqa: E402

from learn.model import export, load  # noqa: E402

if __name__ == "__main__":
    out, paths = sys.argv[1], sys.argv[2:]
    nets = [load(p) for p in paths]
    states = [n.state_dict() for n in nets]
    avg = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            avg[k] = torch.stack([s[k] for s in states]).mean(0)
        else:
            avg[k] = v
    nets[0].load_state_dict(avg)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    export(nets[0], out)
    print(f"averaged {len(paths)} checkpoints into {out}")
