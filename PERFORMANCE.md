# Performance — brief for the `performance` branch

**Goal:** make self-play for learned bots (see `LEARNING.md`) run as fast as
possible on an Apple MacBook Pro M1 Pro (8 performance + 2 efficiency cores,
16 GB unified memory, macOS). This branch is worked on by Claude Code running
locally on that laptop, so measure on the real hardware.

## Starting point

Pure Python, no dependencies. Measured in a Linux container (one core):
about 570 deals/s with `RandomBot`, 420 deals/s with `RuleBot`, about 64
decisions per deal. Re-measure on the laptop first.

## What learning needs from the environment

1. **Raw engine speed:** `Deal.view()`, `legal_actions()` and `apply()` are
   called once per decision. Profile first; expect wins from avoiding
   repeated `legal_actions()` work (`apply` recomputes it to validate),
   cheaper views, and fewer allocations.
2. **Parallel self-play on all cores** (`multiprocessing`; macOS uses
   *spawn*, so workers must be importable and cheap to start).
3. **Batched decisions.** Neural agents are fast only in batches, so the
   runner must advance many deals at once in each worker, collect every
   pending decision, ask the agent for all of them in one call, then apply.
   Provide this interface in `learn/runner.py` (refine it if needed, but keep
   it documented here):

   ```python
   class BatchAgent(Protocol):
       def choose_batch(self, views: list[PlayerView]) -> list[Action]: ...

   def play_many(positions, agents_by_seat, *, games_in_flight=256, workers=None)
       -> Iterator[Deal]   # finished deals, in any order
   ```

   Plain `Agent`s (`choose(view)`) should be wrapped automatically.

## Rules of the branch

- **No behaviour changes.** Same rules, same `PlayerView` fields and meaning,
  same order of `legal_actions()`, records replay identically. Add a test that
  replays a fixed set of recorded random deals made *before* your changes and
  checks every step matches.
- The engine stays pure Python with no required dependencies. Native or
  compiled speed-ups are allowed only as an optional path with the pure
  Python one kept and cross-checked by tests.
- Hidden information stays hidden: views never gain fields that leak.
- Keep it readable; a 10% gain is not worth an unreadable engine.

## Deliverables

1. A benchmark command (e.g. `python -m learn.bench`) reporting deals/s and
   decisions/s for one core and for all cores, with `RandomBot` and
   `RuleBot`, and the cost split of `view` / `legal_actions` / `apply`.
2. Profile-driven engine speed-ups, each with before/after numbers.
3. `learn/runner.py` with batched, multi-process play as above, plus tests.
4. Notes here: final numbers, Python version used (try 3.12 and 3.13), and
   anything the learning side should know.

Work in small commits; run `pytest` and `ruff check . && ruff format --check .`
before each. Merge `main` in before opening a pull request to `main`, and put
the benchmark numbers in the pull request.
