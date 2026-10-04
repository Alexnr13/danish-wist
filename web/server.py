"""Play Danish Wist in the browser against three bots.

    python -m web.server [--port 8000] [--log games.jsonl] [--seed N]

The page first asks how strong each bot should be (`LEVELS`): RuleBot, or one of two trained
networks, which need NumPy (`pip install -e ".[play]"`).

A deliberately small, single-player server built on the standard library.
The engine runs here; the page only ever receives the human player's view.

The page also runs without this server, as a static site (`web/build.py`): it then
loads this module into the browser with Pyodide and asks `in_browser()` instead.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from functools import cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from danish_wist import Card, Match, Phase
from danish_wist.actions import Discard, decode, encode
from danish_wist.bots import Agent, RuleBot
from danish_wist.record import to_record
from danish_wist.tricks import trick_winner

HUMAN = 0
WEB = Path(__file__).parent
PAGE = WEB / "index.html"
LEVELS = {  # the bots a player can choose, weakest first: who plays, and their network
    "weak": ("RuleBot", None),
    "strong": ("rl-005", WEB / "rl-005.npz"),  # rl-005's iteration 3600
    "very-strong": ("rl-006", WEB / "rl-006.npz"),  # rl-006's magnet at 18,000, the best so far
}


@cache
def opponent(level: str) -> Agent:
    if level not in LEVELS:
        raise ValueError(f"no such level: {level!r}")
    network = LEVELS[level][1]
    if network is None:
        return RuleBot()
    try:
        from learn.inference import NumpyAgent  # needs NumPy, unlike the rest of the game
    except ImportError:
        raise ValueError('the trained bots need NumPy: pip install -e ".[play]"') from None
    return NumpyAgent(str(network))


class Table:
    """One human (seat 0) and three bots playing a running match."""

    def __init__(
        self,
        rng: random.Random,
        log_path: Path | None = None,
        levels: Sequence[str] = ("weak",) * 3,
    ) -> None:
        self.rng = rng
        self.log_path = log_path
        self.new_match(levels)

    def new_match(self, levels: Sequence[str]) -> None:
        """Start a new match against bots of these levels, in seats 1 to 3 (West, North, East)."""
        if len(levels) != 3:
            raise ValueError("choose a level for each of the three bots")
        self.bots = {seat: opponent(level) for seat, level in enumerate(levels, start=1)}
        self.levels = list(levels)
        self.match = Match(self.rng, dealer=self.rng.randrange(4))
        self.deal = self.match.new_deal()
        self.picked_up: list[Card] = []  # what the human took from the cat, if they exchanged

    def act(self, text: str) -> None:
        if self.deal.to_act != HUMAN:
            raise ValueError("it is not your turn")
        action = decode(text)
        hand = Counter(self.deal.view(HUMAN).hand)
        self.deal.apply(action)
        view = self.deal.view(HUMAN)
        if isinstance(action, Discard) and view.phase is not Phase.DISCARD:
            # The third discard picks up the cat: what is new in the hand came from it.
            hand[action.card] -= 1
            self.picked_up = list((Counter(view.hand) - hand).elements())

    def step(self) -> None:
        """Let the bot whose turn it is make one move."""
        seat = self.deal.to_act
        if seat is None or seat == HUMAN:
            raise ValueError("no bot to move")
        self.deal.apply(self.bots[seat].choose(self.deal.view(seat)))

    def next_deal(self) -> None:
        if not self.deal.is_over:
            raise ValueError("the deal is not over")
        self.match.record(self.deal)
        if self.log_path:
            seats = ["human"] + [LEVELS[level][0] for level in self.levels]
            record = to_record(self.deal, deal_number=self.match.deals_played, seats=seats)
            with self.log_path.open("a") as log:
                log.write(json.dumps(record) + "\n")
        self.deal = self.match.new_deal()
        self.picked_up = []

    def state(self) -> dict:
        view = self.deal.view(HUMAN)
        deal = self.deal
        return {
            "phase": view.phase.name,
            "to_act": view.to_act,
            "dealer": view.dealer,
            "hand": [str(c) for c in view.hand],
            "legal": [encode(a) for a in view.legal_actions],
            "auction": [[seat, encode_bid(bid)] for seat, bid in view.auction],
            "declarer": view.declarer,
            "bid": str(view.bid) if view.bid else None,
            "called": str(view.called_suit) if view.called_suit else None,
            "partner": view.partner,
            "trumps_decided": view.phase
            in (Phase.EXCHANGE, Phase.DISCARD, Phase.FUCDIC, Phase.PLAY, Phase.DONE),
            "trumps": str(view.trumps) if view.trumps else None,
            "turned_cat": [str(c) for c in view.turned_cat],
            "discards": [str(c) for c in view.discards],
            "picked_up": [str(c) for c in self.picked_up],
            "took_cat": view.took_cat,
            "fucdic_declared": view.fucdic_declared,
            "fucdic": str(view.fucdic) if view.fucdic else None,
            "trick": [[seat, str(card)] for seat, card in view.trick],
            "last_trick": [[seat, str(card)] for seat, card in view.tricks[-1]]
            if view.tricks
            else [],
            "last_trick_winner": trick_winner(list(view.tricks[-1]), view.trumps)
            if view.tricks
            else None,
            "tricks_won": list(view.tricks_won),
            "redeal": deal.redeal,
            "scores": list(view.scores) if view.scores else None,
            "match_scores": self.match.scores,
            "opponents": self.levels,
            "deals_played": self.match.deals_played,
        }


def encode_bid(bid) -> str:
    return "pass" if bid is None else encode(bid)


ROUTES = {
    "/api/new": lambda table, body: table.new_match(body["levels"]),
    "/api/act": lambda table, body: table.act(body["action"]),
    "/api/step": lambda table, body: table.step(),
    "/api/next": lambda table, body: table.next_deal(),
}


def respond(table: Table, path: str, body: dict | None = None) -> tuple[int, dict]:
    """Answer one of the page's requests: a GET if `body` is None, else a POST."""
    if body is None:
        if path == "/api/state":
            return 200, table.state()
        return 404, {"error": "not found"}
    route = ROUTES.get(path)
    if route is None:
        return 404, {"error": "not found"}
    try:
        route(table, body)
    except ValueError as error:
        return 400, {"error": str(error)}
    return 200, table.state()


def in_browser() -> Callable[[str, str], str]:
    """The page's backend on the static site: a table answering a request's path and JSON
    body ("" for a GET) with `[status, state]` as JSON."""
    table = Table(random.Random())
    return lambda path, body: json.dumps(respond(table, path, json.loads(body) if body else None))


def make_handler(table: Table) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send_json(*respond(table, self.path))

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            self._send_json(*respond(table, self.path, body))

        def _send_json(self, status: int, data: dict) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass  # keep the terminal quiet

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log", type=Path, help="append a record of each deal to this file")
    parser.add_argument("--seed", type=int, help="seed for reproducible deals")
    args = parser.parse_args()

    table = Table(random.Random(args.seed), args.log)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(table))
    print(f"Danish Wist: open http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
