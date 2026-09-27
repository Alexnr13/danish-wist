#!/bin/sh
# Pack the checkpoints the workstation needs (runs/ is not in git; see REVIEW.md, "Migration").
# Run from the repository root: sh results/review-2026-09/scripts/pack_runs.sh [out.tar.gz]
set -e
out=${1:-runs/handoff-2026-09.tar.gz}
tar czf "$out" \
  runs/bc-explore.pt runs/bc-explore.npz \
  runs/rl-003/checkpoints/policy-0110.pt runs/rl-003/checkpoints/policy-0110.npz runs/rl-003/checkpoints/critic-0110.pt \
  runs/rl-004d/checkpoints/policy-0010.pt runs/rl-004d/checkpoints/policy-0010.npz runs/rl-004d/checkpoints/critic-0010.pt \
  runs/rl-004d/checkpoints/policy-0100.pt runs/rl-004d/checkpoints/policy-0100.npz runs/rl-004d/checkpoints/critic-0100.pt \
  runs/rl-004d/state.pt runs/rl-004d/run.json runs/rl-004d/settings.json \
  runs/x-004b/policy.pt runs/x-004b/policy.npz \
  runs/arena/rl-004d-0010-vs-rule.jsonl
ls -lh "$out"
