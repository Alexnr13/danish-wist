import json
import random

from web.server import HUMAN, Table


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
