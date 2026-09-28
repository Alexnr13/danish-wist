#!/bin/sh
# Judge the sweep's runs as TRAINING.md §6 says, on the choosing seeds (31, 32): card play on
# fixed contracts, and the full game against each reference field. Every candidate is paired
# with the first, sw-control, so the "paired" lines are each change's effect; the start
# (rl-004d's 10) is a candidate too, to show what the control's 200 iterations did.
# usage: sh results/workstation-2026-09/judge.sh [iteration] > results/workstation-2026-09/judge.txt
# (RUNS=runs/sw-test DEALS=100 for a quick check of the script itself; PREFIX, START and FIELDS
# judge another set of runs, <PREFIX>-control and <PREFIX>-*, as for rl-005's entropy check)
AT=${1:-0200}
RUNS=${RUNS:-runs}
DEALS=${DEALS:-2000}
PREFIX=${PREFIX:-sw}
START=${START:-runs/rl-004d/checkpoints/policy-0010.npz}
FIELDS=${FIELDS:-rule runs/rl-003/checkpoints/policy-0110.npz runs/rl-004d/checkpoints/policy-0010.npz runs/x-004d-0010/policy.pt}
C="$RUNS/$PREFIX-control/checkpoints/policy-$AT.npz $START"
for run in $RUNS/$PREFIX-*/; do
    case $run in */$PREFIX-control/) continue ;; esac
    [ -e ${run}checkpoints/policy-$AT.npz ] && C="$C ${run}checkpoints/policy-$AT.npz"
done
echo "== card play on fixed contracts (play:), against RuleBot"
python -m learn.arena --field rule --seeds 31 32 --deals $DEALS --candidate $(for c in $C; do echo play:$c; done)
for field in $FIELDS; do
    echo; echo "== the full game against a field of $field"
    python -m learn.arena --field $field --seeds 31 32 --deals $DEALS --candidate $C
done
echo; echo "== each run's own exploiters: margin against the learner when each was made"
for run in $RUNS/$PREFIX-*/; do
    echo "$run: $(grep -o '"exploiter_margin": [-0-9.]*' ${run}log.jsonl | cut -d' ' -f2 | tr '\n' ' ')"
done
