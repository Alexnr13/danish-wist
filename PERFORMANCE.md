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

   def play_many(positions, agents_by_seat, *, games_in_flight=256, workers=None,
                 finish=None)
       -> Iterator[Deal]   # finished deals (or finish(deal)), in any order
   ```

   Plain `Agent`s (`choose(view)`) should be wrapped automatically.
   **Done**; see "The runner" below for the details.

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

## Results

Measured on the M1 Pro with `python -m learn.bench` (one core: best of three
runs of 3,000 deals; runs vary by about 2%). "Before" is the engine at
`aef412d`; the golden test (`tests/test_golden.py`) checks that nothing but
speed changed.

| Python 3.13 (uv build), deals/s | RandomBot | RuleBot |
|---|---:|---:|
| One core, before | 1,358 | 1,022 |
| **One core, after** | **4,134** | **2,400** |
| 10 processes each playing alone, before | 9,619 | 7,532 |
| 10 processes each playing alone, after | 30,024 | 17,286 |
| Runner, 10 workers, sending whole deals back | 18,670 | 13,523 |
| Runner, 10 workers, sending a number back | 22,444 | 15,028 |

About 64 decisions per deal, so one core now makes some 260,000 RandomBot
decisions/s. Time per decision with RandomBot: `view` 6.0 → 1.5 µs, `apply`
4.6 → 1.9 µs (this now includes working out the next legal actions), and
`legal_actions()` 2.6 → 0.1 µs (a copy of what is already known).

Each step, one core, RandomBot / RuleBot deals/s:

| Change | RandomBot | RuleBot |
|---|---:|---:|
| Before | 1,348 | 1,029 |
| Legal actions and whose turn worked out once per state | 1,998 | 1,348 |
| One object per card: identity equality and hashing, in C | 2,595 | 1,617 |
| Plays checked first; finished tricks kept for views; deck built once | 3,143 | 1,801 |
| One shared `Play` per card in legal actions | 3,395 | 1,894 |
| Views built without the frozen dataclass's per-field `setattr` | 3,981 | 2,060 |
| RuleBot finds winning cards in one pass (`tricks.winning_cards`) | | 2,214 |
| `Suit` and `Attachment` hashed by identity | 4,106 | 2,277 |
| RuleBot checks plays first; `best_suit` in one list per suit | | 2,448 |
| `apply` records the engine's own action object (for pickling) | same | same |

**Python versions** (final code, one core, RandomBot / RuleBot deals/s):

| Build | deals/s |
|---|---:|
| CPython 3.12.13, uv | 3,789 / 2,237 |
| CPython 3.13.14, uv | 4,073 / 2,439 |
| CPython 3.14.6, uv | 4,030 / 2,389 |

Use **3.13 or 3.14 from uv** (`uv venv -p 3.13 --managed-python`). The
miniforge (conda-forge) 3.13 on this laptop is built without profile-guided
optimisation and ran the old engine about 30% slower than uv's build (928
against 1,305 deals/s). The experimental JIT (`PYTHON_JIT=1`) made things
2–4% slower on both 3.13 and 3.14.

**Cores.** Through the runner, sending a number back (RandomBot / RuleBot
deals/s): 2 workers 6,662 / 4,400; 4 workers 12,590 / 8,505; 6 workers
18,522 / 12,430; 8 workers 22,714 / 14,749; 10 workers 23,221 / 15,895. The
two efficiency cores add little, so use 8 workers when the main process has
work of its own, such as training.

## The runner

`learn/runner.py`, as in the interface above:

- `agents_by_seat` is four agents, or a function from a worker number (0, 1,
  ...) to four agents. Use the function for agents with randomness, so that
  each worker gets its own, or with state too big to send to every worker.
  The same agent object in several seats gets one batch for all of them.
- `workers=None` means one process per core; `workers=1` plays in this
  process, lazily, with the given agents. Workers are started with *spawn*,
  so agent factories and `finish` must be importable, module-level functions
  (a script run as a file with `if __name__ == "__main__":` is fine; code
  typed into stdin is not).
- `positions` is read lazily, in chunks, with a limit on how many are
  queued, so it may be endless.
- `finish(deal)` runs in the worker, and its result is sent back instead of
  the deal.

## Notes for the learning side

- The engine is no longer the bottleneck: about 4 µs per decision, against
  about 8 µs for `learn.encoding.observe(view)` on the learning branch (14 µs
  with the old engine), before any network inference.
- Every card is a single object: `Card.parse("AS") is Card(14, Suit.SPADES)`,
  also after pickling or copying. Cards, suits and attachments hash by
  identity, so dictionaries keyed by them (card and action indices) are
  quick.
- `deal.history` holds the engine's own action objects, which are equal to
  the ones passed to `apply`.
- `Deal.view()` fills the frozen `PlayerView` directly; every view equals
  `PlayerView(**vars(view))`.
- Batches cost the engine some cache locality: in one process, 64 deals in
  flight run about 7% faster than 256, and 1,024 about 20% slower than 256.
  Use the smallest batch the network is efficient at.
- Sending a whole finished deal back from a worker costs about 25 µs there
  and 40 µs in the main process, which then caps all cores at about 15,000
  to 20,000 deals/s. Send back compact results with `finish`, such as NumPy
  arrays.
- Each `play_many` call with workers starts a fresh pool (spawn plus imports,
  well under a second without PyTorch). PPO iterations that collect a few
  hundred deals will want a pool that stays up and takes new weights: see
  "Next".
- Positions made in the main process cost about 9 µs each (the shuffle).

## Next

`LEARNING.md` (on `learning`) asks the runner to let the learner record,
for each of its decisions, the deal and seat it belongs to, and to see the
`Deal` in the worker for its training-only critic tokens; its collector also
seats different agents in different deals. With a pool that stays up between
PPO iterations, that is the next runner milestone. It changes the interface
above, so it goes through `main` first.
