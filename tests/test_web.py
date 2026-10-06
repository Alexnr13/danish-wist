import hashlib
import json
import random
import subprocess
import sys
import zipfile

from helpers import auction_won_by, deal_with

from danish_wist import Attachment, Bid, CallAce, NameTrumps, Play, Suit
from danish_wist.actions import encode
from danish_wist.bots import RuleBot
from learn.inference import NumpyAgent
from web.build import build
from web.server import HUMAN, Table, opponent, respond


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


def test_the_player_chooses_each_bots_level_and_the_record_names_them(tmp_path):
    log = tmp_path / "games.jsonl"
    table = Table(random.Random(9), log)
    levels = ["weak", "strong", "very-strong"]
    assert respond(table, "/api/new", {"levels": levels})[0] == 200
    assert table.state()["opponents"] == levels
    assert isinstance(table.bots[1], RuleBot)
    assert isinstance(table.bots[2], NumpyAgent) and isinstance(table.bots[3], NumpyAgent)
    assert table.bots[2].net.weights["policy.bias"].tolist() != (
        table.bots[3].net.weights["policy.bias"].tolist()
    )  # two different networks
    while table.deal.to_act is not None:
        if table.deal.to_act == HUMAN:
            table.act(table.state()["legal"][0])
        else:
            table.step()
    table.next_deal()
    assert json.loads(log.read_text())["meta"]["seats"] == ["human", "RuleBot", "rl-005", "rl-006"]


def test_a_new_match_needs_a_known_level_for_each_bot():
    table = Table(random.Random(9))
    unknown = respond(table, "/api/new", {"levels": ["weak", "weak", "expert"]})
    assert unknown == (400, {"error": "no such level: 'expert'"})
    assert respond(table, "/api/new", {"levels": ["weak"]})[0] == 400
    assert table.state()["opponents"] == ["weak", "weak", "weak"]  # the match carries on


def finish_deal(table, choose=lambda state: state["legal"][0]):
    """Play the deal out: the human as `choose` says, the bots their own way."""
    while table.deal.to_act is not None:
        if table.deal.to_act == HUMAN:
            table.act(choose(table.state()))
        else:
            table.step()


def test_cheats_are_off_unless_chosen_and_a_record_names_them(tmp_path):
    log = tmp_path / "games.jsonl"
    table = Table(random.Random(9), log)
    state = table.state()
    assert (state["cheats"], state["trump_count"], state["best_move"]) == ([], None, None)
    refused = respond(table, "/api/new", {"levels": ["weak"] * 3, "cheats": ["x-ray"]})
    assert refused == (400, {"error": "no such cheat: 'x-ray'"})
    assert respond(table, "/api/new", {"levels": ["weak"] * 3, "cheats": ["trump-count"]})[0] == 200
    finish_deal(table)
    table.next_deal()
    assert json.loads(log.read_text())["meta"]["cheats"] == ["trump-count"]


def test_the_trump_counter_counts_trumps_played_and_those_in_the_hand():
    table = Table(random.Random(3))
    table.new_match(["weak"] * 3, ["trump-count"])
    trumps_played = 0
    for _ in range(3):
        while table.deal.to_act is not None:
            deal, expected = table.deal, None
            if deal.trumps is not None:  # named, and not no trumps
                held = [card.suit is deal.trumps for card in deal.hands[HUMAN]]
                played = [
                    a.card.suit is deal.trumps for _, a in deal.history if isinstance(a, Play)
                ]
                expected = [sum(played), sum(played) + sum(held)]
                trumps_played = max(trumps_played, sum(played))
            assert table.state()["trump_count"] == expected
            if deal.to_act == HUMAN:
                table.act(table.state()["legal"][0])
            else:
                table.step()
        table.next_deal()
    assert trumps_played >= 4  # the deals had trumps to count


def test_the_best_move_is_what_the_very_strong_bot_would_do_in_the_humans_place():
    table = Table(random.Random(9))
    table.new_match(["weak"] * 3, ["best-move"])
    adviser = opponent("very-strong")
    advised = 0
    while table.deal.to_act is not None:
        state = table.state()
        if table.deal.to_act == HUMAN:
            assert state["best_move"] == encode(adviser.choose(table.deal.view(HUMAN)))
            assert state["best_move"] in state["legal"]
            table.act(state["best_move"])  # take the advice
            advised += 1
        else:
            assert state["best_move"] is None
            table.step()
    assert advised >= 13 and table.state()["best_move"] is None


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


WEST = 1


def play_out(declarer: int, bid: Bid, called: Suit = Suit.SPADES) -> Table:
    """A deal `declarer` wins with `bid`, calling an ace (the human holds the spade ace), with
    hearts trumps and every play the first legal one: the human takes 6 tricks, West none."""
    table = Table(random.Random(9))
    table.deal = deal_with({HUMAN: "AS AH KH QH JH", WEST: "AD KD"})
    auction_won_by(table.deal, declarer, bid)
    table.deal.apply(CallAce(called))
    table.deal.apply(NameTrumps(Suit.HEARTS))
    while table.deal.to_act is not None:
        assert not table.state()["leunged"]
        table.deal.apply(table.deal.legal_actions()[0])
    return table


def test_the_page_celebrates_a_bots_halves_failed_by_four_with_the_human_as_partner():
    leunged = play_out(WEST, Bid(10, Attachment.HALVES))
    assert leunged.deal.tricks_won[:2] == [6, 0]
    assert leunged.state()["leunged"]
    assert not play_out(WEST, Bid(9, Attachment.HALVES)).state()["leunged"]  # failed by 3
    assert not play_out(WEST, Bid(10)).state()["leunged"]  # not Halves
    # Failed by 4 or more, but East was the partner, or the human declared (alone).
    assert not play_out(WEST, Bid(13, Attachment.HALVES), Suit.CLUBS).state()["leunged"]
    assert not play_out(HUMAN, Bid(13, Attachment.HALVES)).state()["leunged"]


# What the page runs under Pyodide: wist.zip's Python alone, through in_browser().
PLAY_IN_BROWSER = """
import json, sys
sys.path.insert(0, sys.argv[1])
import danish_wist, learn.inference, web.server
assert all(m.__file__.startswith(sys.argv[1]) for m in (danish_wist, learn.inference, web.server))
answer = web.server.in_browser()
levels = json.dumps({"levels": ["weak", "strong", "very-strong"]})
status, state = json.loads(answer("/api/new", levels))
assert state["trump_count"] is None and state["best_move"] is None  # no cheats chosen
new = {"levels": ["weak", "strong", "very-strong"], "cheats": ["trump-count", "best-move"]}
status, state = json.loads(answer("/api/new", json.dumps(new)))
assert status == 200, state
while state["phase"] != "DONE":
    if state["to_act"] == 0:
        status, state = json.loads(answer("/api/act", json.dumps({"action": state["best_move"]})))
    else:
        status, state = json.loads(answer("/api/step", "{}"))
    assert status == 200, state
"""


def test_the_static_site_plays_a_deal_with_only_its_own_python(tmp_path):
    build(tmp_path / "site")
    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    version = hashlib.sha256((tmp_path / "site" / "wist.zip").read_bytes()).hexdigest()[:12]
    assert "in_browser" in page and f'fetch("wist.zip?v={version}")' in page  # never a stale one
    with zipfile.ZipFile(tmp_path / "site" / "wist.zip") as archive:
        archive.extractall(tmp_path / "wist")
    subprocess.run(
        [sys.executable, "-c", PLAY_IN_BROWSER, str(tmp_path / "wist")], cwd=tmp_path, check=True
    )
