#!/bin/sh
# TRAINING.md "Next", step 4: a one-change check of the entropy bonus from rl-005's 2000. Two runs
# of 200 iterations with rl-005's settings, the sweep's exploiters (every 50 iterations for 25) and
# a league of the new reference policies: the control at rl-005's 0.03 and one at 0.01. About 20
# minutes per run. Judge them with judge.sh (PREFIX=ent; see "Next").
# usage (from the worktree, with .venv/bin on PATH): sh results/rl-005/entropy-check.sh
# (ITERATIONS=10 OUT=runs/ent-test for a quick check of the script itself)
set -e
ITERATIONS=${ITERATIONS:-200}
OUT=${OUT:-runs}
START="--init runs/rl-005/checkpoints/policy-2000.pt --init-critic runs/rl-005/checkpoints/critic-2000.pt --critic-warmup 2"
LEAGUE="--league-add runs/rl-004d/checkpoints/policy-0010.pt runs/rl-005/checkpoints/policy-1000.pt runs/x-004d-0010/policy.pt runs/x-005-2000/policy.pt --exploit-every 50 --exploit-iterations 25"
BASE="--deals 1024 --iterations $ITERATIONS --eval-every 10 --eval-deals 2000 --magnet 0.1 --magnet-ema 0.01 --explore-bids 0.15 --explore-levels 0.1"

mkdir -p $OUT
run() {
    name=$1; shift
    [ -e $OUT/$name/log.jsonl ] && { echo "$name already run"; return; }
    echo "== $name: $*"
    python -m learn.selfplay $START $LEAGUE $BASE "$@" --out $OUT/$name > $OUT/$name.out 2>&1
}

run ent-control --entropy 0.03
run ent-0.01 --entropy 0.01
