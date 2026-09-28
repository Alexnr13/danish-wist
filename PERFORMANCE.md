# Performance — brief for the `performance` branch

*Training now runs on an RTX 5090 workstation: see "The RTX 5090 workstation"
below for its numbers, and "Training throughput" for how self-play has run
there since 28 September. The rest describes the laptop, where this began.*

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
   **Done**, and extended for self-play training with a `Runner` that stays
   up between plays, as agreed with the learning side in pull request #3.
   See "The runner" below.

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
| Runner, 8 workers, sending whole deals back | 16,648 | 12,274 |
| Runner, 8 workers, sending a number back | 20,918 | 14,443 |

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
deals/s): 2 workers 6,597 / 4,349; 4 workers 12,836 / 8,537; 6 workers
17,523 / 12,337; 8 workers 22,537 / 15,459; 10 workers 20,744 / 15,493. The
runner gives every worker the same share of the games (that is what makes
runs reproducible), so the slowest worker sets the pace, and the two
efficiency cores are slower. **Use 8 workers**, which also leaves the main
process a core for training.

## The runner

`learn/runner.py`. A `Runner` is a pool of worker processes, each with its
own agents, that stays up between plays:

```python
class Runner:
    def __init__(self, make_agents, *, workers=None, games_in_flight=256):
        """make_agents(worker) -> {name: agent}, run once in each worker (0, 1, ...)."""

    def play(self, games, finish=None) -> Iterator[Any]:
        """Games are (Position, [agent name for each seat]) pairs, read lazily.

        finish(game, deal, agents) runs in the worker; by default the deal is sent.
        """

    def broadcast(self, name, method, *args) -> list[Any]:
        """agents[name].method(*args) in every worker; their results, in worker order."""

    def close(self) -> None:
        """Stop the workers. A Runner is also a context manager."""


@dataclass(frozen=True)
class Decision:
    game: int  # the game's index in the stream given to play
    seat: int
    view: PlayerView  # what the player may know: a policy uses only this
    deal: Deal  # everything, hidden cards included: for training targets only
```

- **Agents** may define `choose(view)`, `choose_batch(views)` or
  `choose_decisions(decisions)`. Each round, each agent object gets one call
  with all of its decisions, across seats and games. Only an agent that
  defines `choose_decisions` sees `Decision.deal`, and only to build training
  targets: the critic's tokens (`encode_oracle`) and the belief head's
  targets (`belief_targets`).
- **Workers** are started with *spawn*, so `make_agents` and `finish` must
  be module-level functions (or `functools.partial`s of them), in an
  importable module or in a script run as a file under
  `if __name__ == "__main__":`. `workers=1` plays in this process with the
  agents made here; `workers=None` means one per core. Starting 8 workers
  with PyTorch takes about a second, once.
- **Reproducible.** Games go out in chunks, chunk k to worker k mod
  `workers`, and each worker plays its chunks in order. An agent that seeds
  itself from a broadcast seed and its worker number therefore makes the
  same choices whenever the same games are played with the same number of
  workers. A list of games is split evenly between the workers; an endless
  stream is read as it is needed.
- **One play at a time**, and no `broadcast` during a play. An error in a
  worker (in an agent, `finish` or `make_agents`) is raised here as the
  original exception, with the worker's traceback in a note. A play stopped
  early, or one that failed, leaves the runner ready for the next.
- `play_many(positions, agents_by_seat, ...)` is the one-off form with the
  same four agents in every deal; it starts a `Runner` for that call.

## Self-play with the policy

*The laptop, with the networks in the workers. Self-play now runs its networks in the main process for all the workers: see "Training throughput".*

Measured with the learning side's network (`NetConfig()`: width 128, 4
layers) in each worker, PyTorch on one thread per worker, recording steps,
critic tokens and belief targets as `collect()` does and sending trajectories
back (the code is under "Next"):

| Collecting pure self-play | deals/s | memory per worker |
|---|---:|---:|
| `collect()` in one process (8 PyTorch threads) | 50 | |
| Runner, 8 workers, 8 games in flight each | 151 | 250 MB |
| Runner, 8 workers, 16 games in flight each | 151 | 274 MB |
| Runner, 8 workers, 32 games in flight each | 148 | 442 MB |
| Runner, 8 workers, 64 games in flight each | 143 | 539 MB |

So a 256-deal PPO iteration collects in about 1.7 s instead of 5 s. Bigger
batches do not help a network on one CPU thread and cost memory (256 in
flight took about 1 GB per worker), so use `games_in_flight=16`. Within a
worker, the network's forward pass takes about 75% of the time; encoding
(`observe`, `encode_oracle`) and building tensors (`collate`, NumPy arrays)
take most of the rest, and the engine about 2%.
Sending new weights to all 8 workers with `broadcast` takes about 15 ms.

## Notes for the learning side

- The engine is no longer the bottleneck: about 4 µs per decision, against
  about 8 µs for `learn.encoding.observe(view)` (14 µs with the old engine),
  and far more for the network.
- Every card is a single object: `Card.parse("AS") is Card(14, Suit.SPADES)`,
  also after pickling or copying. Cards, suits and attachments hash by
  identity, so dictionaries keyed by them (card and action indices) are
  quick.
- `deal.history` holds the engine's own action objects, which are equal to
  the ones passed to `apply`.
- `Deal.view()` fills the frozen `PlayerView` directly; every view equals
  `PlayerView(**vars(view))`.
- Sending a whole finished deal back from a worker costs about 25 µs there
  and 40 µs in the main process, which then caps all cores at about 15,000
  to 20,000 deals/s. Send back compact results from `finish`, such as NumPy
  arrays.
- Each `play_many` call starts its own pool; for repeated plays, keep one
  `Runner`.
- Positions made in the main process cost about 9 µs each (the shuffle).

## The RTX 5090 workstation (September 2026)

Training moved to a workstation: Intel Core Ultra 9 285K (24 cores: 8
performance at up to 5.7 GHz, 16 efficiency at 4.7 GHz, no hyperthreading),
60 GB, NVIDIA RTX 5090 (32 GB, compute capability 12.0), Linux. uv's CPython
3.13.14, PyTorch 2.14.0+cu130, NumPy 2.5.3, NVIDIA driver 595.91.07.

**The engine** (`python -m learn.bench`; the laptop in brackets):

| deals/s | RandomBot | RuleBot |
|---|---:|---:|
| One core, plain loop | 5,666 (4,134) | 3,456 (2,400) |
| Runner, 24 workers, sending a number | 71,738 (20,918 on 8) | 50,202 (14,443 on 8) |
| Runner, 22 workers, sending a number | | 52,380 |

**Self-play and evaluation**, from rl-004d's iteration 10 at rl-004's
settings (1024 deals per iteration, about 47,500 recorded decisions):

| | Laptop (M1 Pro, MPS) | Workstation, first try | Workstation, now |
|---|---:|---:|---:|
| Collecting 1024 deals | 7 s | 27 s | 1.3 s |
| Preparing and updating | 48 s | 12–33 s | 3.0 s |
| Evaluation, 2000 deals × 5 games | 12–24 s | | 1.7 s |
| `learn.arena`, 2000 deals | | 4 min (NumPy, 11 workers) | 7.6 s |

What it took, in order of effect:

1. **The networks in the workers run on the GPU** (`--worker-device`,
   default cuda). On one CPU thread this network managed 150–450 decisions/s
   per core here, against about 1,200 on an M1 core; on the GPU the forward
   pass is a small part of a worker's time. Sampling stays on the CPU.
2. **`collate` builds a batch with whole-array operations**: row by row it
   ran at about 1,300 decisions/s, slower than the network on a GPU; now
   about 160,000. It served the workers and the update alike.
3. **One BLAS/OpenMP thread per worker** (`Runner` sets the environment
   when it starts them). NumPy's OpenBLAS started a thread per core in each
   of 22 workers: a load average of 175, and a 2000-deal NumPy arena was not
   done after ten minutes. The same arena takes 7.6 s on the GPU.
4. **Workers stop when the process that started them dies.** Killing an
   arena left its 22 workers playing, orphaned, for 45 minutes, and they
   spoiled every timing taken meanwhile.
5. The update: minibatches cut into passes by length (2.9 → 2.7 s), plain
   attention on CUDA (−10%, the same results), statistics kept on the GPU,
   embedding lookups instead of the multi-hot product (which MPS needed).
6. `encode_oracle` reuses the tokens of the view it extends: `encode` had
   been a third of a worker's time, half of it for the second copy.

**How many workers** (for `learn.arena` and the other tools, which still put their networks in the workers; self-play no longer does, see "Training throughput"). With networks on the GPU, collecting is limited by the
GPU switching between the workers' processes (each has its own CUDA context;
kernels from different processes do not run at the same time), not by
cores. From rl-004d's 10 over 8192 deals, 256 games in flight per worker:

| Workers | deals/s | Peak GPU memory |
|---:|---:|---:|
| 8 | 770 | 8.8 GB |
| 12 | 783 | 12.6 GB |
| 16 | 820 | 16.4 GB |
| 22 | 732 | 21.3 GB |

Runs vary by about 10%. More games in flight help a little (16 × 512: about
1,050 deals/s) but cost memory: 8 workers × 1024 ran the GPU out of memory.
So the tools default to 12 workers when their networks are on a GPU
(`learn.arena.default_workers`), all cores but two otherwise. The NVIDIA
driver's multi-process service (MPS) or one inference process serving all
workers would remove the switching; neither is worth it while the update is
the slower half.

**The update** (47,500 decisions, one PPO epoch, minibatches of 512): about
2.5 s on the GPU plus 0.4 s for the critic's values. The GPU is busy for
about 2 s of it at about a tenth of its float32 peak: 512 decisions of
about 40 tokens (the critic's about 68) make small matrix products, and much
of the time is attention's backward pass and thousands of small elementwise
kernels. Tried:

| Change | Update |
|---|---:|
| As committed (chunks of 256 by length, plain attention) | 2.5 s |
| TF32 matrix products | no change |
| bfloat16 autocast | 1.8 s, but value loss 1.7 → 4.7 and approx KL ×4: not used |
| `torch.compile` of the trunk | 2.15 s, after 14 s compiling: not used yet |
| Minibatches of 2048 / 4096 | 2.07 / 2.02 s (a learning setting, not a speed one) |

So the update is about 60% of an iteration. The next steps, if more is
needed: collect the next deals while updating (the policy then lags one
iteration, which PPO's ratio to the collecting policy already allows for),
and `torch.compile`.

*28 September: `torch.compile` and bfloat16 are now in use, and the update takes about 1.2 s (see "Training throughput"). Collecting while updating is not: PPO's usual ratio does not allow for the lag as well as this said (Cleanba found even one iteration hurts); it needs the decoupled objective, described under "Training throughput", "Next".*

## Training throughput (28 September 2026)

The GPU pulsed during training: busy, then idle, over and over. This section is the review of why, what was changed, and what was measured, with the literature it drew on. The measurements used the real training command with rl-005's settings, `nvidia-smi` and `/proc/stat` sampled every 100 ms, an Nsight Systems trace of ten iterations, and timers around each phase.

### What it was

Like for like, on rl-005's own iterations 3901–3999 (its log) and on a copy of rl-005 resumed with the new code for iterations 4002–4040, both with the full league of 50:

| Per iteration (1024 deals, about 36,800 decisions) | rl-005 as run | Now |
|---|---:|---:|
| Collecting | 2.30 s | 1.00 s |
| Preparing and updating | 2.20 s | 1.20 s |
| Evaluation, 2000 deals × 5 games, every 10th | 1.5 s | 1.6 s |
| **Wall time per iteration**, evaluation included | **4.99 s** | **2.45 s** |
| GPU memory | 11.6 GB | 6.2 GB |

A 20-iteration exploiter now takes about 1.2 s per iteration (rl-005's: 1.8 s).

Why it pulsed:

- **The two halves took turns.** Collecting used the GPU from 12 worker processes, then the update used it from the main process, and nothing overlapped. The pulsing in a resource monitor is the phases taking turns, and it still shows (see "Next" below for the change that would end it).
- **The update was a stream of tiny kernels.** One iteration launched about 184,000 kernels (mean 9.4 µs); the GPU was truly busy for about 1.7 s of the 2.5. Of that, 46% was float32 matrix products and 8% one LayerNorm backward kernel running on 4 thread blocks of the GPU's 170 multiprocessors.
- **Collecting was split across 13 CUDA contexts.** Each worker held its own copy of every network and ran about 27,000 kernels an iteration (mean 2.8 µs) on batches of about 86 decisions split between up to nine networks. Kernels from different processes never run at the same time: the GPU switches between them.
- **`nvidia-smi`'s "utilization" is not how busy the GPU is.** It is the share of time any kernel was running. It read about 80% during updates that used a small fraction of the GPU. Wall time per iteration is the measure; since the changes the GPU's average "utilization" is lower (55% against 79%) because the same work finishes sooner.
- **On this GPU TF32 is no faster than float32; bfloat16 is.** Measured on 4096 × 4096 products: 65 TFLOPS in float32, 213 in bfloat16. That is why TF32 "changed nothing" earlier.

### What changed

1. **The update** (`learn/selfplay.py`). The batch goes to the GPU once (`Padded`). Every pass has one shape per batch: `chunk` rows (now 512, the whole minibatch), a short one filled with copies of a row that weigh 0, and as many tokens as the batch's longest row, rounded up to 16. Each loss runs as one `torch.compile`d function (`_compiled`), compiled once for each shape (`dynamic=False`). The trunk runs in bfloat16 and the heads, softmaxes, log-chances and losses in float32 (`_summarise`). The earlier bfloat16 try (value loss 1.7 to 4.7) had the heads in bfloat16; the logits' rounding is what spoiled it. On a saved batch of 41,620 decisions, preparing and updating went from 2.55 s to 1.29 s with the same losses. `tests/test_selfplay.py` checks that the compiled update agrees with the CPU's: to 1e-7 in float32, within rounding in bfloat16.
2. **The network** (`learn/model.py`, `Net.summarise`). Only the summary token's output is read, so the last encoder layer works out only that. It gives the same numbers as the whole encoder (to 1e-6) for about a fifth less work (update 1.64 to 1.39 s).
3. **Collecting** (`learn/runner.py`, `learn/model.py`). The networks run in the main process for every worker (`Runner(networks=...)`, `ServedAgent`, `model.Networks`), in the pattern of SEED RL and Sample Factory.
   - **Workers:** a worker's served agents turn their decisions into one `Question` (the tokens and legal actions as arrays) and sample from the answer, the legal actions' logits.
   - **Rounds:** each round, the main process takes one message from every worker still playing and runs each network once on all of their questions. The workers keep in step, so the batches, and so the learner's decisions, are the same whenever the same games are played with the same number of workers (a test).
   - **The forward pass:** it is replayed from CUDA graphs, one per batch shape, captured on one copy of the network into which each network's weights are copied first (`_Graphed`). Its trunk runs in bfloat16 as in the update. The recorded log-chances are now closer to the update's (mean gap 0.0010, largest 0.035) than they were with float32 collecting (0.0015 and 0.169).
   - **Workers use CPUs only:** self-play now defaults to all cores but two (`--worker-device` is gone; old `run.json` files still resume).
4. **Weights** no longer go to 12 workers by pickle every iteration; loading them into the served networks is a copy on the GPU. With that, everything outside collecting, updating and evaluating fell from about 0.35 s to about 0.06 s an iteration. Writing the checkpoints and `state.pt` takes about 27 ms (1%), so it stays in the main thread.

The tests: 215 pass (208 before).

### Tried, and not kept

- **CUDA MPS** (NVIDIA's multi-process service) let the 12 workers' kernels overlap: collecting went from 1.49 to 0.81 s with no code change. About eight minutes later the GPU hung (`Xid 13, Graphics FECS Exception`, then `NV_ERR_RESET_REQUIRED`) and needed a reboot. This GPU also drives the desktop. **Do not use MPS on this machine.**
- **Compiling for any shape** (`dynamic=True`, or marked dynamic sizes): Inductor guards on the first shape's size (for example, at least 79 tokens) and recompiles when a later shape fails the guard. In a 30-iteration test that was 24 s stalls at iterations 11 and 27, and after eight recompiles the compiler falls back to running eagerly, with one warning in the log. "Unbacked" sizes failed inside PyTorch. Hence one static compile per batch shape.
- **Fused AdamW**: no measurable gain (2.14 against 2.13 s), and loading an older optimiser state quietly turns it off.
- **CUDA graphs for the update** (`mode="reduce-overhead"`): no gain once the update was limited by the GPU's arithmetic, and awkward with gradient accumulation.
- **Dropping the per-step check for non-finite gradients**: 3% faster, but a single bad step would then poison the weights. It has never fired in 4200 logged iterations; it stays.
- **Compiling the served forward pass** inside its graphs: 3.3 to 2.6 ms a round, but with many batch shapes the recompiles came back. Graphs and bfloat16 alone are kept.
- **Double-buffering the workers** (half their games stepping while the other half wait for answers, as in Sample Factory): a round's time is mostly the main process's (the networks, receiving 22 questions), and halving the batches would double that part.
- **Writing checkpoints from a background thread**: saving is 1% of an iteration.

### Where the time goes now

Collecting 1024 deals is about 70 lockstep rounds of about 13 ms. The workers compute for 2 to 3 ms, the 22 questions take about 2 ms to arrive and unpickle, the networks about 3.4 ms (now limited by the GPU's arithmetic), and the answers 0.2 ms. Most workers are idle most of the time (CPU about 18%). The update's 1.2 s is limited by the GPU: bfloat16 matrix products, attention and fused elementwise kernels.

To measure it again, run a few dozen iterations of the real command and read `collect_s`, `update_s` and `eval_s` in `log.jsonl` (the first iteration includes compiling: tens of seconds with an empty cache, which `/tmp/torchinductor_$USER` keeps until a reboot).

### Next (not agreed)

- **Overlap collecting with the update (the largest step left).** Collect the next iteration's deals while updating on this one's. An iteration would then take about the longer of the two halves, about 1.3 s instead of 2.45, and the GPU would stop pulsing. But the deals would come from a policy one update behind, which changes the algorithm. With PPO's usual objective even that lag measurably hurts learning (Cleanba, arXiv 2310.00036). The fix is the decoupled objective of Hilton et al. 2021 (arXiv 2110.00641; OpenAI's `ppo-ewma` code). The update recomputes the log-chances under the weights it starts from (the "proximal" policy), clips the ratio against those, and weighs each step by proximal over behaviour odds, capped. `ppo_objective` already has this form for explored bids. Hilton et al. found it holds up for about eight iterations of lag on Procgen; nothing measures it in a self-play league. It needs the user's agreement and a paired comparison against a synchronous run.
- **One message per worker per round**, all its agents' questions in one `Question` with the network of each row, to make receiving cheaper.
- **Serving without lockstep** (answering whatever questions have arrived, as SEED RL does) would be faster still, but a run would no longer repeat exactly.

## Next

The learning side moves `collect()` onto the `Runner`, along these lines
(this is the code the measurements above ran):

```python
class Learner:  # made once in each worker
    def __init__(self, worker):
        torch.set_num_threads(1)
        self.worker, self.net, self.steps = worker, Net(NetConfig()).eval(), {}
        self.generator = torch.Generator()

    def load(self, state):  # runner.broadcast("learner", "load", state)
        self.net.load_state_dict(state)

    def seed(self, n):  # runner.broadcast("learner", "seed", n)
        self.generator.manual_seed(n * 1000 + self.worker)

    @torch.no_grad()
    def choose_decisions(self, decisions):
        observations = [observe(d.view) for d in decisions]
        log_probs = torch.log_softmax(self.net(*collate(observations))[0], dim=-1)
        actions = torch.multinomial(log_probs.exp(), 1, generator=self.generator)
        chosen = log_probs.gather(1, actions).squeeze(-1).tolist()
        actions = actions.squeeze(-1).tolist()
        for d, o, a, p in zip(decisions, observations, actions, chosen):
            oracle = np.asarray(encode_oracle(d.deal, d.seat), dtype=np.int16)
            belief = np.asarray(belief_targets(d.deal, d.seat), dtype=np.int8)
            step = Step(_compact(o), oracle, belief, a, p)
            self.steps.setdefault((d.game, d.seat), []).append(step)
        return [ACTIONS[a] for a in actions]


def make_agents(worker):  # snapshots: more named agents, filled by broadcast
    return {"learner": Learner(worker), "rule": RuleBot()}


def trajectories(game, deal, agents):  # runs in the worker
    steps = agents["learner"].steps
    return [
        Trajectory(steps.pop((game, seat)), float(deal.scores[seat]))
        for seat in range(4)
        if (game, seat) in steps
    ]


with Runner(make_agents, workers=8, games_in_flight=16) as runner:
    for iteration in range(iterations):
        runner.broadcast("learner", "load", policy.state_dict())
        runner.broadcast("learner", "seed", iteration)
        games = [(position, ["learner", "rule", "learner", "learner"]), ...]
        batch = [t for ts in runner.play(games, trajectories) for t in ts]
```

Duplicate evaluation runs through the same pool: each position once with the
field in every seat, and once with the candidate in each seat, with `finish`
returning the scores (`tests/test_runner.py` checks that this matches
`learn.arena.duplicate`).

After that, collecting is the network's forward pass. If it has to be
faster, the choices are the learning side's: a smaller network for
self-play, the NumPy copy if it is quicker on one thread, or one copy on the
GPU fed by all the workers.
