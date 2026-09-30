import gzip
import json

import pytest

from learn.curve import curve, paired


def test_a_paired_change_is_the_mean_change_per_deal():
    mean, ci = paired([0, 10, 20, 30], [5, 15, 25, 35])
    assert mean == 5 and ci == 0
    mean, ci = paired([0, 0, 0, 0], [1, -1, 3, 1])
    assert mean == 1 and ci == pytest.approx(1.96 * 1.633 / 2, rel=1e-3)


def write_run(path, results):
    log, evals = [], []
    for i, per_deal in enumerate(results, start=1):
        n = len(per_deal)
        mean = sum(per_deal) / n
        log.append({"iteration": 10 * i, "vs_rulebot": mean, "vs_rulebot_ci95": 1.0})
        evals.append({"iteration": 10 * i, "per_deal": per_deal})
    for name, rows in [("log.jsonl", log), ("evals.jsonl", evals)]:
        (path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_curve_counts_significant_falls_in_a_row(tmp_path):
    base = [float(i % 7) for i in range(50)]
    results = [base, [x - 5 for x in base], [x - 10 for x in base], [x - 9 for x in base]]
    write_run(tmp_path, results)
    lines = curve(tmp_path)
    assert len(lines) == 1 + 4
    falls = [line.split()[7] for line in lines[2:]]  # iter, result ± ci, change ± ci, falls
    assert falls == ["1", "2", "0"]


def test_the_curve_reads_a_gzipped_copy_of_the_evaluations(tmp_path):
    base = [float(i % 7) for i in range(50)]
    write_run(tmp_path, [base, [x - 5 for x in base]])
    expected = curve(tmp_path)
    evals = tmp_path / "evals.jsonl"
    (tmp_path / "evals.jsonl.gz").write_bytes(gzip.compress(evals.read_bytes()))
    evals.unlink()
    assert curve(tmp_path) == expected
