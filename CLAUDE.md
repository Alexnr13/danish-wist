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
