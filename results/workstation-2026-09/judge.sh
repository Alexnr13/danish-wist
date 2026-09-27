#!/bin/sh
# Judge the sweep's runs as TRAINING.md §6 says, on the choosing seeds (31, 32): card play on
# fixed contracts, and the full game against each reference field, all paired with the start.
# usage: sh results/workstation-2026-09/judge.sh [iteration] > results/workstation-2026-09/judge.txt
AT=${1:-0200}
C="runs/rl-004d/checkpoints/policy-0010.npz"
for run in runs/sw-*/; do
    [ -e ${run}checkpoints/policy-$AT.npz ] && C="$C ${run}checkpoints/policy-$AT.npz"
done
echo "== card play on fixed contracts (play:), against RuleBot"
python -m learn.arena --field rule --seeds 31 32 --deals 2000 --candidate $(for c in $C; do echo play:$c; done)
for field in rule runs/rl-003/checkpoints/policy-0110.npz runs/rl-004d/checkpoints/policy-0010.npz runs/x-004d-0010/policy.pt; do
    echo; echo "== the full game against a field of $field"
    python -m learn.arena --field $field --seeds 31 32 --deals 2000 --candidate $C
done
echo; echo "== the runs' own exploiters (margin against the learner when each was made)"
for run in runs/sw-*/; do
    echo "$run: $(grep -o '"exploiter_margin": [-0-9.]*' ${run}log.jsonl | cut -d' ' -f2 | tr '\n' ' ')"
done
