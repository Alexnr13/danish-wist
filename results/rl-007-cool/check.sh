#!/bin/sh
# The cooldown check (TRAINING.md, Results, "The cooldown check"), at 19,000 on the one-change
# seeds 31 to 36 (12,000 deals): card play on fixed contracts, and the full game against the
# reference fields (RuleBot, rl-006's 5400, the magnet at 18,000, x-006-magnet-18000-s1).
# First the cooldown's policy and magnet and the control's policy, each paired with the control's
# magnet; then the cooldown's policy paired with its own magnet.
# usage: sh results/rl-007-cool/check.sh      (writes check.txt and check-own-magnet.txt here)
HERE=results/rl-007-cool
export SEEDS="31 32 33 34 35 36"
export FIELDS="rule runs/rl-006/checkpoints/policy-5400.npz runs/rl-006/checkpoints/magnet-18000.npz runs/x-006-magnet-18000-s1/policy.pt"
START=runs/rl-007/checkpoints/magnet-19000.npz \
C="runs/rl-007-cool/checkpoints/policy-19000.npz runs/rl-007-cool/checkpoints/magnet-19000.npz runs/rl-007/checkpoints/policy-19000.npz" \
    sh results/rl-005/choose.sh > $HERE/check.txt
START=runs/rl-007-cool/checkpoints/magnet-19000.npz \
C="runs/rl-007-cool/checkpoints/policy-19000.npz" \
    sh results/rl-005/choose.sh > $HERE/check-own-magnet.txt
