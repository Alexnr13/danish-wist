#!/bin/sh
# Choose among rl-005's checkpoints as TRAINING.md §6 says: every 200 iterations from FROM and the
# last, each paired with START, on the choosing seeds: card play on fixed contracts, and the full
# game against each reference field.
# usage: sh results/rl-005/choose.sh > results/rl-005/choose.txt      (as run on 28 September)
# The defaults are that first choice's; override SEEDS, START, FROM, FIELDS, DEALS, or C (the
# candidates: SEEDS="0 41 42" C="<policies>" reports chosen ones on the reporting seeds).
SEEDS=${SEEDS:-31 32}
START=${START:-runs/rl-004d/checkpoints/policy-0010.npz}
FROM=${FROM:-0}
FIELDS=${FIELDS:-rule runs/rl-003/checkpoints/policy-0110.npz runs/rl-004d/checkpoints/policy-0010.npz runs/x-004d-0010/policy.pt}
DEALS=${DEALS:-2000}
if [ -z "$C" ]; then
    LAST=$(ls runs/rl-005/checkpoints/policy-*.npz | tail -1)
    for c in runs/rl-005/checkpoints/policy-*[02468]00.npz $LAST; do
        n=$(basename $c .npz | sed 's/^policy-0*//')
        [ "${n:-0}" -ge "$FROM" ] && C="$C $c"
    done
fi
C="$START $(echo $C | tr ' ' '\n' | awk 'NF && !seen[$0]++' | tr '\n' ' ')"
echo "== card play on fixed contracts (play:), against RuleBot"
python -m learn.arena --field rule --seeds $SEEDS --deals $DEALS --candidate $(for c in $C; do echo play:$c; done)
for field in $FIELDS; do
    echo; echo "== the full game against a field of $field"
    python -m learn.arena --field $field --seeds $SEEDS --deals $DEALS --candidate $C
done
