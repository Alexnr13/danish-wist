"""Play Danish Wist in the browser against three bots.

    python -m web.server [--port 8000] [--log games.jsonl] [--seed N]

A deliberately small, single-player server built on the standard library.
The engine runs here; the page only ever receives the human player's view.
"""

from __future__ import annotations

import argparse
import json
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from danish_wist import Match, Phase
from danish_wist.actions import decode, encode
from danish_wist.bots import RuleBot
from danish_wist.record import to_record

HUMAN = 0
PAGE = Path(__file__).with_name("index.html")


class Table:
    """One human (seat 0) and three bots playing a running match."""

    def __init__(self, rng: random.Random, log_path: Path | None = None) -> None:
        self.match = Match(rng, dealer=rng.randrange(4))
        self.bots = {seat: RuleBot() for seat in range(1, 4)}
        self.log_path = log_path
        self.deal = self.match.new_deal()

    def act(self, text: str) -> None:
        if self.deal.to_act != HUMAN:
            raise ValueError("it is not your turn")
        self.deal.apply(decode(text))

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
            record = to_record(self.deal, deal_number=self.match.deals_played)
            with self.log_path.open("a") as log:
                log.write(json.dumps(record) + "\n")
        self.deal = self.match.new_deal()

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
            "trumps_decided": view.phase in (Phase.EXCHANGE, Phase.DISCARD, Phase.PLAY, Phase.DONE),
            "trumps": str(view.trumps) if view.trumps else None,
            "turned_cat": [str(c) for c in view.turned_cat],
            "discards": [str(c) for c in view.discards],
            "trick": [[seat, str(card)] for seat, card in view.trick],
            "last_trick": [[seat, str(card)] for seat, card in view.tricks[-1]]
            if view.tricks
            else [],
            "tricks_won": list(view.tricks_won),
            "redeal": deal.redeal,
            "scores": list(view.scores) if view.scores else None,
            "match_scores": self.match.scores,
            "deals_played": self.match.deals_played,
        }


def encode_bid(bid) -> str:
    return "pass" if bid is None else encode(bid)


def make_handler(table: Table) -> type[BaseHTTPRequestHandler]:
    routes = {
        "/api/act": lambda body: table.act(body["action"]),
        "/api/step": lambda body: table.step(),
        "/api/next": lambda body: table.next_deal(),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send_json(200, table.state())
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            route = routes.get(self.path)
            if route is None:
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                route(body)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            self._send_json(200, table.state())

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
