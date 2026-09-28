#!/bin/sh
# The review's measures for the handoff checkpoints, on the reporting seeds (TRAINING.md §6).
# usage: sh results/workstation-2026-09/reference.sh > results/workstation-2026-09/reference.txt
C="runs/rl-004d/checkpoints/policy-0010.npz runs/bc-explore.npz runs/rl-003/checkpoints/policy-0110.npz runs/rl-004d/checkpoints/policy-0100.npz runs/x-004b/policy.npz"
echo "== card play on fixed contracts (play:), against RuleBot"
python -m learn.arena --field rule --seeds 0 41 42 --deals 2000 --candidate $(for c in $C; do echo play:$c; done)
for field in rule runs/rl-003/checkpoints/policy-0110.npz runs/rl-004d/checkpoints/policy-0010.npz runs/x-004d-0010/policy.pt; do
    echo; echo "== the full game against a field of $field"
    python -m learn.arena --field $field --seeds 0 41 42 --deals 2000 --candidate $C
done
