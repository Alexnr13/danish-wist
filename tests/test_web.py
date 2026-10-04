import json
import random
import subprocess
import sys
import zipfile

from helpers import auction_won_by, deal_with

from danish_wist import Bid, CallAce, NameTrumps, Suit
from learn.inference import NumpyAgent
from web.build import build
from web.server import BOT, HUMAN, Table, respond


def test_a_human_and_bots_can_play_deals_and_log_them(tmp_path):
    log = tmp_path / "games.jsonl"
    table = Table(random.Random(9), log)
    for _ in range(3):
        while table.deal.to_act is not None:
            if table.deal.to_act == HUMAN:
                table.act(table.state()["legal"][0])
            else:
                table.step()
        table.next_deal()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 3
    assert json.dumps(table.state())  # serialisable


def test_state_shows_only_the_humans_cards():
    table = Table(random.Random(9))
    state = table.state()
    assert len(state["hand"]) == 13
    assert all(str(c) in state["hand"] for c in table.deal.hands[HUMAN])


def test_after_exchanging_the_human_sees_what_they_picked_up_from_the_cat():
    table = Table(random.Random(9))
    table.deal = deal_with({HUMAN: "KS JK 2C 3C"}, cat="JK AD AC")
    auction_won_by(table.deal, HUMAN, Bid(8))
    table.deal.apply(CallAce(Suit.HEARTS))
    table.deal.apply(NameTrumps(Suit.SPADES))
    table.act("take-cat")
    table.act("discard 2C")
    table.act("discard 3C")
    assert table.state()["picked_up"] == []  # the cat comes up after the third discard
    table.act("discard JK")  # a Joker goes down and another comes up
    state = table.state()
    assert state["discards"] == ["2C", "3C", "JK"]
    assert sorted(state["picked_up"]) == sorted(str(card) for card in table.deal.cat)
    while table.deal.to_act is not None:
        if table.deal.to_act == HUMAN:
            table.act(table.state()["legal"][0])
        else:
            table.step()
    table.next_deal()
    assert table.state()["picked_up"] == []


def test_the_default_bot_is_a_trained_network_that_plays_a_deal():
    table = Table(random.Random(9), bot=NumpyAgent(str(BOT)))
    assert all(isinstance(bot, NumpyAgent) for bot in table.bots.values())
    while table.deal.to_act is not None:
        if table.deal.to_act == HUMAN:
            table.act(table.state()["legal"][0])
        else:
            table.step()
    assert table.deal.is_over


def test_the_pages_requests_are_answered_with_the_state_or_an_error():
    table = Table(random.Random(9))
    assert respond(table, "/api/state") == (200, table.state())
    assert respond(table, "/api/nowhere")[0] == 404
    assert respond(table, "/api/nowhere", {})[0] == 404
    assert respond(table, "/api/next", {}) == (400, {"error": "the deal is not over"})
    while table.deal.to_act != HUMAN:
        assert respond(table, "/api/step", {}) == (200, table.state())
    assert respond(table, "/api/step", {}) == (400, {"error": "no bot to move"})
    assert respond(table, "/api/act", {"action": table.state()["legal"][0]})[0] == 200


# What the page runs under Pyodide: wist.zip's Python alone, through in_browser().
PLAY_IN_BROWSER = """
import json, sys
sys.path.insert(0, sys.argv[1])
import danish_wist, learn.inference, web.server
assert all(m.__file__.startswith(sys.argv[1]) for m in (danish_wist, learn.inference, web.server))
answer = web.server.in_browser()
status, state = json.loads(answer("/api/state", ""))
while state["phase"] != "DONE":
    if state["to_act"] == 0:
        status, state = json.loads(answer("/api/act", json.dumps({"action": state["legal"][0]})))
    else:
        status, state = json.loads(answer("/api/step", "{}"))
    assert status == 200, state
"""


def test_the_static_site_plays_a_deal_with_only_its_own_python(tmp_path):
    build(tmp_path / "site")
    assert "in_browser" in (tmp_path / "site" / "index.html").read_text()
    with zipfile.ZipFile(tmp_path / "site" / "wist.zip") as archive:
        archive.extractall(tmp_path / "wist")
    subprocess.run(
        [sys.executable, "-c", PLAY_IN_BROWSER, str(tmp_path / "wist")], cwd=tmp_path, check=True
    )
