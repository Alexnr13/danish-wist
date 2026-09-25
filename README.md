# Danish Wist

Our house version of Call-ace Whist (*Esmakker Whist*), written down and playable.

- **[RULES.md](RULES.md)**: the rules, which are the source of truth.
- **`danish_wist/`**: a pure-Python rules engine with no dependencies, plus simple bots.
- **[LEARNING.md](LEARNING.md)**: research and plan for machine-learned bots.
- **`web/`**: a small standard-library server and one plain page for playing in the browser.

## Play against bots

```sh
python -m web.server            # then open http://localhost:8000
python -m web.server --log games.jsonl   # also keep a record of every deal
```

You sit South; three `RuleBot`s play the other seats (`danish_wist/bots.py`).

## Development

```sh
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

## Using the engine

```python
import random
from danish_wist import Deal

deal = Deal.new(dealer=0, rng=random.Random())
while not deal.is_over:
    view = deal.view(deal.to_act)       # only what that seat may know
    deal.apply(random.choice(view.legal_actions))
print(deal.scores)
```

Every deal can be saved as a replayable record (see `danish_wist/record.py`):

```python
import json
from danish_wist.record import replay, to_record

line = json.dumps(to_record(deal, match="friday", deal_number=1))
assert replay(json.loads(line)).scores == deal.scores
```

## Next: learned bots

The next piece of work is machine-learned bots of varying strength, up to
optimal play. The groundwork is in place:

- `Deal.view(seat)` gives exactly what a player may know (hidden partner and
  cat), with `legal_actions` listed, so agents cannot peek.
- `Deal.apply()` and `Match` run headless and fast; `RandomBot` and `RuleBot`
  are baselines to measure against.
- `record.py` stores every deal losslessly (JSON Lines), for analysing large
  numbers of self-play games.

Plan: [LEARNING.md](LEARNING.md). Compare bots by duplicate play (the same
cards replayed with the candidate in each seat):

```sh
python -m learn.arena --candidate rule --field random --deals 1000
```
