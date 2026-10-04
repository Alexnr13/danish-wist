# Danish Wist

Our house version of Call-ace Whist (*Esmakker Whist*), written down and playable.

- **[RULES.md](RULES.md)**: the rules, which are the source of truth.
- **`danish_wist/`**: a pure-Python rules engine with no dependencies, plus simple bots.
- **[LEARNING.md](LEARNING.md)**: research and plan for machine-learned bots.
- **`web/`**: a small standard-library server and one plain page for playing in the browser;
  the same page also runs on its own, online, with the engine in the browser.

## Play against bots

```sh
pip install -e ".[play]"        # NumPy, for the trained bots
python -m web.server            # then open http://localhost:8000
python -m web.server --log games.jsonl   # also keep a record of every deal
python -m web.server --bot rule          # hand-written RuleBots instead (no NumPy needed)
```

You sit South; three copies of the best trained bot so far play the other
seats: `web/bot.npz`, rl-006's magnet at 18,000 (see [TRAINING.md](TRAINING.md)).
`--bot` takes another exported network (`.npz`), or `rule` for the RuleBots
in `danish_wist/bots.py`.

## Play online

**https://alexnr13.github.io/danish-wist/** plays the same game in any browser, with
nothing to install: the page loads Pyodide (Python built for the browser) and NumPy from a
CDN and runs the engine and the trained bot there, so no server is involved. GitHub Pages
publishes it from `main` on every push (`.github/workflows/pages.yml`). A reload starts a
new match, and nothing is recorded. To try the static site locally:

```sh
python -m web.build _site && python -m http.server -d _site   # http://localhost:8000
```

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
    view = deal.view(deal.to_act)  # only what that seat may know
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

Plan: [LEARNING.md](LEARNING.md). Running training: [TRAINING.md](TRAINING.md). Compare bots by duplicate play (the same
cards replayed with the candidate in each seat):

```sh
python -m learn.arena --candidate rule --field random --deals 1000
python -m learn.arena --candidate rule --field random --record games.jsonl
python -m learn.report games.jsonl     # how each agent bids and plays
```

Training needs PyTorch; playing a trained bot needs only NumPy:

```sh
pip install -e ".[learn]"                     # numpy + torch
python -m learn.imitate --deals 5000 --out runs/bc.pt   # also writes runs/bc.npz
pip install -e ".[play]"                      # numpy only
python -m web.server --bot runs/bc.npz        # play against the network
python -m learn.selfplay --init runs/bc.pt --out runs/rl   # self-play, all cores + MPS
```
