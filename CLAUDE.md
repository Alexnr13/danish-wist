# Working on Danish Wist

- `RULES.md` is the source of truth. Change the rules there first, then the
  code and its tests. Every rule should have a test.
- Keep it small. Prefer deleting and simplifying over adding. No new
  dependencies or layers without a clear, present need.
- The engine (`danish_wist/`) is pure Python with no I/O, no randomness except
  an injected `random.Random`, and no UI. Bots, interfaces, networking and
  learning code sit on top of `Deal.view()` / `legal_actions()` / `apply()`.
- Players must only ever see what `Deal.view(seat)` gives them (partner
  identity and the cat are hidden information).
- `record.py` records are the analysis format: starting position plus every
  action. Any engine change must keep old records replaying, or bump
  `FORMAT_VERSION`.
- No art direction yet: interfaces stay plain.

Checks before committing: `pytest` and `ruff check . && ruff format --check .`

## Parallel branches

Two agents work at once. Keep to your own area and merge `main` in often.

| Branch | Who | Owns |
|---|---|---|
| `main` | — | Stable. Changed only by merging a branch with tests green. |
| `learning` | Cloud Claude session | `learn/` (encoder, networks, training, evaluation), `LEARNING.md` |
| `performance` | Claude Code on the M1 Pro laptop | Speed of `danish_wist/` internals, benchmarks, the parallel game runner (`learn/runner.py`), `PERFORMANCE.md` |

- Neither branch changes the rules, `RULES.md`, or the engine's public
  behaviour (`Deal`, `PlayerView`, actions, `legal_actions` order, record
  replay) without agreeing it through `main` first.
- The learning side codes against the public engine API and the runner
  interface in `PERFORMANCE.md`; the performance side keeps both stable.
