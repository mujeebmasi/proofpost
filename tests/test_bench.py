from pathlib import Path

from proofpost.bench import auc, run_bench

DATA = Path(__file__).resolve().parents[1] / "evals" / "data"


def test_auc_basics():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([1], [1]) == 0.5


def test_voice_score_separates_real_from_other_author_and_ai():
    r = run_bench(DATA)
    assert r.auc_vs_other_author >= 0.9
    assert r.auc_vs_ai >= 0.9
    assert r.ai_flagged_by_phrases == 1.0
    assert r.own_flagged_by_phrases == 0.0
