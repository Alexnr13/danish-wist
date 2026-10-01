#!/bin/sh
# Card-play checks during rl-007's run to 32,000 (TRAINING.md "Next", Task C): every 1000
# iterations, the policy and its magnet on the choosing seeds 45 and 46 (4000 deals), paired with
# rl-006's magnet at 18,000, alongside the run with 4 workers. Waits for each checkpoint and the
# next iteration's log line (for the last, for the run to end), so the files are written.
# usage: nohup sh results/rl-007/play-checks.sh > runs/play-checks-rl-007.out 2>&1 &
BEST=runs/rl-006/checkpoints/magnet-18000.npz
LAST=${LAST:-32000}
for n in $(seq ${FIRST:-20000} 1000 $LAST); do
    out=results/rl-007/play-check-$n.txt
    [ -f $out ] && continue
    until [ -f runs/rl-007/checkpoints/magnet-$n.npz ] && \
        { grep -q "\"iteration\": $((n + 1))," runs/rl-007/log.jsonl || \
          { [ $n -eq $LAST ] && ! kill -0 $(cat runs/rl-007.pid) 2>/dev/null; }; }; do
        kill -0 $(cat runs/rl-007.pid) 2>/dev/null || [ -f runs/rl-007/checkpoints/magnet-$n.npz ] || \
            { echo "run stopped before $n"; exit 1; }
        sleep 30
    done
    python -m learn.arena --workers 4 --field rule --seeds 45 46 --deals 2000 \
        --candidate play:$BEST play:runs/rl-007/checkpoints/policy-$n.npz \
        play:runs/rl-007/checkpoints/magnet-$n.npz > $out.tmp 2>>runs/play-check-rl-007.err && mv $out.tmp $out
done
echo PLAY-CHECKS-DONE
