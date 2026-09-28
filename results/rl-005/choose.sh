#!/bin/sh
# Choose among rl-005's checkpoints as TRAINING.md §6 says: every 200 iterations and the last,
# paired with the start (rl-004d's 10), on the choosing seeds 31 and 32: card play on fixed
# contracts, and the full game against each reference field.
# usage: sh results/rl-005/choose.sh > results/rl-005/choose.txt
# (SEEDS="0 41 42" C="<policies>" to report chosen ones on the reporting seeds)
SEEDS=${SEEDS:-31 32}
START=runs/rl-004d/checkpoints/policy-0010.npz
LAST=$(ls runs/rl-005/checkpoints/policy-*.npz | tail -1)
C=${C:-"$(ls runs/rl-005/checkpoints/policy-*[02468]00.npz) $LAST"}
C="$START $(echo $C | tr ' ' '\n' | awk '!seen[$0]++' | tr '\n' ' ')"
echo "== card play on fixed contracts (play:), against RuleBot"
python -m learn.arena --field rule --seeds $SEEDS --deals 2000 --candidate $(for c in $C; do echo play:$c; done)
for field in rule runs/rl-003/checkpoints/policy-0110.npz $START runs/x-004d-0010/policy.pt; do
    echo; echo "== the full game against a field of $field"
    python -m learn.arena --field $field --seeds $SEEDS --deals 2000 --candidate $C
done
