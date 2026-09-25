"""How fast does self-play run? See PERFORMANCE.md.

Reports deals and decisions per second for `RandomBot` and `RuleBot` in every
seat: on one core with a plain loop and with the batched runner, and on all
cores with the runner, sending back whole deals or only a number. Then where
the time goes in each decision.

    python -m learn.bench                 # the full report
    python -m learn.bench --deals 1000    # quicker
    python -m learn.bench --workers 0     # one core only
"""

from __future__ import annotations

import argparse
import os
import platform
import random
import sys
from time import perf_counter

from danish_wist.bots import RandomBot, RuleBot
from danish_wist.game import Deal
from learn.arena import random_positions
from learn.runner import play_many


def random_agents(worker: int) -> list:
    return [RandomBot(random.Random(worker))] * 4


def rule_agents(worker: int) -> list:
    return [RuleBot()] * 4


AGENTS = {"random": random_agents, "rule": rule_agents}


def decisions(deal: Deal) -> int:
    return len(deal.history)


def play_deals(bot_name: str, deals: int, seed: int) -> int:
    """Play `deals` deals one at a time; return the number of decisions."""
    rng = random.Random(seed)
    agents = AGENTS[bot_name](seed)
    count = 0
    for i in range(deals):
        deal = Deal.new(i % 4, rng)
        while not deal.is_over:
            seat = deal.to_act
            deal.apply(agents[seat].choose(deal.view(seat)))
            count += 1
    return count


def plain_loop(bot_name: str, deals: int, repeat: int = 3) -> tuple[float, float]:
    """Deals and decisions per second in this process, the best of `repeat` runs."""
    play_deals(bot_name, max(deals // 10, 10), seed=1)  # warm up
    best = 0.0, 0.0
    for _ in range(repeat):
        start = perf_counter()
        count = play_deals(bot_name, deals, seed=0)
        elapsed = perf_counter() - start
        best = max(best, (deals / elapsed, count / elapsed))
    return best


def runner(bot_name: str, deals: int, workers: int, whole_deals: bool) -> tuple[float, float]:
    """Deals and decisions per second through `play_many`."""
    positions = random_positions(deals, random.Random(0))
    warm_up = random_positions(20 * max(workers, 1), random.Random(1))
    list(play_many(warm_up, AGENTS[bot_name], workers=workers))  # start the workers
    start = perf_counter()
    if whole_deals:
        count = sum(map(decisions, play_many(positions, AGENTS[bot_name], workers=workers)))
    else:
        count = sum(play_many(positions, AGENTS[bot_name], workers=workers, finish=decisions))
    elapsed = perf_counter() - start
    return deals / elapsed, count / elapsed


def cost_split(bot_name: str, deals: int) -> dict[str, float]:
    """Microseconds per decision spent in each engine call and in the bot.

    `legal_actions` is timed on its own for the split; a normal decision does
    not call it (the view already holds the legal actions).
    """
    rng = random.Random(2)
    agents = AGENTS[bot_name](2)
    totals = dict.fromkeys(("view", "legal_actions", "choose", "apply", "new deal"), 0.0)
    count = 0
    for i in range(deals):
        t0 = perf_counter()
        deal = Deal.new(i % 4, rng)
        totals["new deal"] += perf_counter() - t0
        while not deal.is_over:
            t0 = perf_counter()
            view = deal.view(deal.to_act)
            t1 = perf_counter()
            deal.legal_actions()
            t2 = perf_counter()
            action = agents[view.seat].choose(view)
            t3 = perf_counter()
            deal.apply(action)
            t4 = perf_counter()
            totals["view"] += t1 - t0
            totals["legal_actions"] += t2 - t1
            totals["choose"] += t3 - t2
            totals["apply"] += t4 - t3
            count += 1
    return {name: total / count * 1e6 for name, total in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark self-play speed.")
    parser.add_argument("--deals", type=int, default=3000, help="deals per core per measurement")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="0 skips the all-cores runs"
    )
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]} on {platform.machine()}, {os.cpu_count()} cores")
    print(f"{'':<32} {'bot':>6} {'deals/s':>9} {'decisions/s':>12}")

    def report(label: str, bot_name: str, rates: tuple[float, float]) -> None:
        print(f"{label:<32} {bot_name:>6} {rates[0]:>9,.0f} {rates[1]:>12,.0f}")

    for bot_name in AGENTS:
        report("one core, plain loop", bot_name, plain_loop(bot_name, args.deals))
    for bot_name in AGENTS:
        report("one core, runner", bot_name, runner(bot_name, args.deals, 1, True))
    if args.workers:
        total = args.deals * args.workers
        for whole_deals, sent in ((True, "whole deals"), (False, "a number")):
            for bot_name in AGENTS:
                rates = runner(bot_name, total, args.workers, whole_deals)
                report(f"{args.workers} workers, sending {sent}", bot_name, rates)
    print("\nµs per decision, one core:")
    for bot_name in AGENTS:
        split = cost_split(bot_name, max(args.deals // 3, 1))
        print(f"  {bot_name:>6}: " + ", ".join(f"{k} {v:.2f}" for k, v in split.items()))


if __name__ == "__main__":
    main()
