#!/bin/sh
# REVIEW.md T2.3 and T2.6: one change at a time from a common control, each run
# from rl-004d's 10 with the league and exploiters on. About 20 minutes per run.
# usage (from the worktree, with .venv/bin on PATH): sh results/workstation-2026-09/sweep.sh
set -e
START="--init runs/rl-004d/checkpoints/policy-0010.pt --init-critic runs/rl-004d/checkpoints/critic-0010.pt --critic-warmup 2"
LEAGUE="--league-add runs/rl-003/checkpoints/policy-0110.pt runs/rl-004d/checkpoints/policy-0100.pt runs/x-004b/policy.pt runs/x-004d-0010/policy.pt --exploit-every 50 --exploit-iterations 25"
BASE="--deals 1024 --iterations 200 --eval-every 10 --eval-deals 2000 --magnet 0.1 --magnet-ema 0.01"
EXPLORE="--explore-bids 0.15 --explore-levels 0.1"

run() {
    name=$1; shift
    [ -e runs/$name/log.jsonl ] && { echo "$name already run"; return; }
    echo "== $name: $*"
    python -m learn.selfplay $START $LEAGUE $BASE "$@" --out runs/$name > runs/$name.out 2>&1
}

run sw-control $EXPLORE --stake-scaling
run sw-entropy-0.03 $EXPLORE --stake-scaling --entropy 0.03
run sw-entropy-0.1 $EXPLORE --stake-scaling --entropy 0.1
run sw-lr-2.5e-4-2ep $EXPLORE --stake-scaling --policy-lr 2.5e-4 --ppo-epochs 2
run sw-no-explore-levels --explore-bids 0.15 --stake-scaling
run sw-no-stake-scaling $EXPLORE
