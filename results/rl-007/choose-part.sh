#!/bin/sh
# One part of the choice among rl-007's magnets (TRAINING.md "Next", Task C, and §6): each
# candidate paired with rl-006's magnet at 18,000 on the choosing seeds 45 and 46 (4000 deals),
# card play on fixed contracts, and the full game against the reference set (RuleBot, rl-006's
# 5400, the magnet at 18,000, x-006-magnet-18000-s1). Parts share the start, seeds and fields, so
# results/rl-006/choose_table.py reads them together. Waits for the part's last checkpoint when
# it runs alongside the run (4 workers).
# usage: FIRST=19500 LAST=24000 PART=a sh results/rl-007/choose-part.sh
#        C="<candidates>" PART=c WORKERS=22 sh results/rl-007/choose-part.sh
REF="rule runs/rl-006/checkpoints/policy-5400.npz runs/rl-006/checkpoints/magnet-18000.npz runs/x-006-magnet-18000-s1/policy.pt"
if [ -z "$C" ]; then
    until [ -f runs/rl-007/checkpoints/magnet-$LAST.npz ] && \
        { grep -q "\"iteration\": $((LAST + 1))," runs/rl-007/log.jsonl || \
          ! kill -0 $(cat runs/rl-007.pid) 2>/dev/null; }; do
        sleep 60
    done
    C=$(for n in $(seq $FIRST 500 $LAST); do echo runs/rl-007/checkpoints/magnet-$n.npz; done)
fi
SEEDS="45 46" START=runs/rl-006/checkpoints/magnet-18000.npz FIELDS="$REF" C="$C" WORKERS=${WORKERS:-4} \
    sh results/rl-005/choose.sh > results/rl-007/choose-$PART.txt.tmp 2>runs/choose-rl-007-$PART.err \
    && mv results/rl-007/choose-$PART.txt.tmp results/rl-007/choose-$PART.txt
echo CHOOSE-$PART-DONE
