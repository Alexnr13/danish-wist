#!/bin/sh
# After the stop rule fired at 23,600 (two in-run exploiters in a row with a significant margin):
# each of the last three in-run exploiters (23,200, 23,400, 23,600) played again against the
# policy it was trained against, on learn.exploit's fresh deals (seeds 101 and 102, 4000 deals;
# in the run each was measured on 2000 fixed deals), and against the magnet of that iteration,
# the kind of checkpoint that is chosen. The league keeps bare weights, so each is first written
# as a checkpoint with the policy's config (runs/rl-007-exploiters/).
# usage: sh results/rl-007/stop-check.sh > results/rl-007/stop-check.txt
mkdir -p runs/rl-007-exploiters
for n in 23200 23400 23600; do
    python -c "
import torch
config = torch.load('runs/rl-007/checkpoints/policy-$n.pt', weights_only=True)['config']
state = torch.load('runs/rl-007/league/exploiter-$n.pt', weights_only=True)
torch.save({'config': config, 'state': state}, 'runs/rl-007-exploiters/exploiter-$n.pt')"
    for target in policy magnet; do
        python -m learn.arena --field runs/rl-007/checkpoints/$target-$n.npz --seeds 101 102 \
            --deals 2000 --candidate runs/rl-007-exploiters/exploiter-$n.pt
    done
done
