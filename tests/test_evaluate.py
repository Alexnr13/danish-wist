import json
import random

from danish_wist.bots import RuleBot
from danish_wist.record import replay
from learn.arena import duplicate, random_positions
from learn.evaluate import run
from learn.report import report, top_cards


def test_an_agent_is_level_with_itself():
    assert run("rule", "rule", 5).per_deal == [0.0] * 5


def test_matches_the_single_process_arena_for_deterministic_agents():
    positions = random_positions(6, random.Random(0))
    expected = duplicate(RuleBot(), RuleBot(), positions)
    assert run("rule", "rule", 6, seed=0).per_deal == expected.per_deal


def test_worker_processes_give_the_same_answer_every_time():
    first = run("rule", "random", 8, seed=3, workers=2)
    second = run("rule", "random", 8, seed=3, workers=2)
    assert first.per_deal == second.per_deal and first.mean > 0


def test_every_deal_is_recorded_and_replays(tmp_path):
    path = tmp_path / "games.jsonl"
    run("rule", "random", 3, seed=4, record=path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 3 * 5
    assert records[0]["meta"]["seats"] == ["random"] * 4
    assert sorted(records[1]["meta"]["seats"]) == ["random", "random", "random", "rule"]
    assert all(replay(r).scores == r["outcome"].get("scores", [0] * 4) for r in records)


def test_report_summarises_each_agent(tmp_path):
    path = tmp_path / "games.jsonl"
    run("rule", "random", 10, seed=5, record=path)
    text = report([json.loads(line) for line in path.read_text().splitlines()])
    assert "== rule:" in text and "== random:" in text
    assert "As declarer:" in text and "Bidding by top cards" in text


def test_top_cards_counts_aces_and_jokers():
    assert top_cards("AS AH JK KS 2C") == 3
