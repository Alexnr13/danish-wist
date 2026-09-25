"""How fast does self-play run? See PERFORMANCE.md.

Reports deals and decisions per second on one core and on all cores, for
`RandomBot` and `RuleBot` in every seat, and where the time goes per decision.

    python -m learn.bench                 # the full report
    python -m learn.bench --deals 1000    # quicker
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import platform
import random
import sys
from time import perf_counter

from danish_wist.bots import RandomBot, RuleBot
from danish_wist.game import Deal

BOTS = {"random": lambda rng: RandomBot(rng), "rule": lambda rng: RuleBot()}


def play_deals(bot_name: str, deals: int, seed: int) -> int:
    """Play `deals` deals with one bot in every seat; return the number of decisions."""
    rng = random.Random(seed)
    bot = BOTS[bot_name](rng)
    decisions = 0
    for i in range(deals):
        deal = Deal.new(i % 4, rng)
        while not deal.is_over:
            deal.apply(bot.choose(deal.view(deal.to_act)))
            decisions += 1
    return decisions


def one_core(bot_name: str, deals: int, repeat: int = 3) -> tuple[float, float]:
    """Deals and decisions per second in this process, the best of `repeat` runs."""
    play_deals(bot_name, max(deals // 10, 10), seed=1)  # warm up
    best = 0.0, 0.0
    for _ in range(repeat):
        start = perf_counter()
        decisions = play_deals(bot_name, deals, seed=0)
        elapsed = perf_counter() - start
        best = max(best, (deals / elapsed, decisions / elapsed))
    return best


def _worker(args: tuple[str, int, int]) -> int:
    return play_deals(*args)


def all_cores(bot_name: str, deals_per_worker: int, workers: int) -> tuple[float, float]:
    """Deals and decisions per second with `workers` processes, each playing its own deals."""
    context = multiprocessing.get_context("spawn")  # what macOS uses, and the slowest start
    with context.Pool(workers) as pool:
        pool.map(_worker, [(bot_name, 5, seed) for seed in range(workers)])  # start everyone
        start = perf_counter()
        jobs = [(bot_name, deals_per_worker, seed) for seed in range(workers)]
        decisions = sum(pool.map(_worker, jobs))
        elapsed = perf_counter() - start
    deals = deals_per_worker * workers
    return deals / elapsed, decisions / elapsed


def cost_split(bot_name: str, deals: int) -> dict[str, float]:
    """Microseconds per decision spent in each engine call and in the bot.

    `legal_actions` is timed on its own for the split; a normal decision does
    not call it directly (the view and `apply` use it internally).
    """
    rng = random.Random(2)
    bot = BOTS[bot_name](rng)
    totals = dict.fromkeys(("view", "legal_actions", "choose", "apply", "new deal"), 0.0)
    decisions = 0
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
            action = bot.choose(view)
            t3 = perf_counter()
            deal.apply(action)
            t4 = perf_counter()
            totals["view"] += t1 - t0
            totals["legal_actions"] += t2 - t1
            totals["choose"] += t3 - t2
            totals["apply"] += t4 - t3
            decisions += 1
    return {name: total / decisions * 1e6 for name, total in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark self-play speed.")
    parser.add_argument("--deals", type=int, default=3000, help="deals per measurement")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="0 skips the all-cores runs"
    )
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]} on {platform.machine()}, {os.cpu_count()} cores")
    print(f"{'':>12} {'bot':>6} {'deals/s':>9} {'decisions/s':>12}")
    for bot_name in BOTS:
        deals_s, decisions_s = one_core(bot_name, args.deals)
        print(f"{'one core':>12} {bot_name:>6} {deals_s:>9,.0f} {decisions_s:>12,.0f}")
    for bot_name in BOTS if args.workers else ():
        deals_s, decisions_s = all_cores(bot_name, args.deals, args.workers)
        label = f"{args.workers} workers"
        print(f"{label:>12} {bot_name:>6} {deals_s:>9,.0f} {decisions_s:>12,.0f}")
    print("\nµs per decision, one core:")
    for bot_name in BOTS:
        split = cost_split(bot_name, max(args.deals // 3, 1))
        print(f"  {bot_name:>6}: " + ", ".join(f"{k} {v:.2f}" for k, v in split.items()))


if __name__ == "__main__":
    main()
