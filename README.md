# Danish Wist

Our house version of Call-ace Whist (*Esmakker Whist*), written down and playable.

- **[RULES.md](RULES.md)**: the rules, which are the source of truth.
- **`danish_wist/`**: a pure-Python rules engine with no dependencies.

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
