#!/bin/sh
# The workstation line's checkpoints on the same 6000 deals (the reporting seeds 0, 41 and 42):
# card play on fixed contracts and the full game, both against RuleBot, every 200th iteration of
# rl-005 (from rl-004d's 10) and of rl-006 (from rl-005's 3600). plot.py draws them.
# usage: sh results/progress/sweep.sh      (about 30 minutes on the workstation)
C="runs/rl-004d/checkpoints/policy-0010.npz $(ls runs/rl-005/checkpoints/policy-*[02468]00.npz runs/rl-006/checkpoints/policy-*[02468]00.npz)"
python -m learn.arena --field rule --seeds 0 41 42 --deals 2000 \
    --candidate $(for c in $C; do echo play:$c; done) > results/progress/card-play.txt
python -m learn.arena --field rule --seeds 0 41 42 --deals 2000 --candidate $C > results/progress/full-game.txt
