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

## How work is organised

One agent works at a time: Claude Code on the RTX 5090 workstation (the
MacBook Pro M1 Pro until 27 September 2026), where all development, training
and evaluation happen. It works on a branch and merges
into `main` by pull request with tests green.

| Branch | State |
|---|---|
| `main` | Stable. Changed only by merging a pull request with tests green. |
| `training` | The current line of work: `learn/`, runs' results, `TRAINING.md`. Checked out in the git worktree `~/danish-wist-training`. |
| `learning`, `performance` | Earlier parallel work (a cloud agent on `learn/`, a local one on engine speed and the runner). Fully merged; no longer active. |

- Train in the worktree `~/danish-wist-training`, with the Python in its
  `.venv`. `runs/` (checkpoints, resumable state, recorded deals) is
  gitignored and lives there, but after each run a chosen set is force-added
  (`git add -f`: what resumes it, the best checkpoints, the exploiters, the
  recorded deals; listed in `TRAINING.md`), as agreed with the user on 28
  September 2026. Agree a long run with the user before starting it.
- `TRAINING.md` says where training stands and what comes next,
  `LEARNING.md` holds the method and plan, and `PERFORMANCE.md` covers the
  engine's speed and the runner.
- Changing the rules, `RULES.md`, or the engine's public behaviour (`Deal`,
  `PlayerView`, actions, `legal_actions` order, record replay) needs the
  user's agreement first.
