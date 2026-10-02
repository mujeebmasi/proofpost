from pathlib import Path

import pytest

from proofpost.checks.voice import build_profile, score_voice, split_samples

DATA = Path(__file__).resolve().parents[1] / "evals" / "data"


def test_split_samples_on_dash_lines():
    assert split_samples("a\n---\nb\n\n---\n  \nc") == ["a", "b", "c"]


def test_needs_at_least_two_samples():
    with pytest.raises(ValueError):
        build_profile(["only one"])


def test_own_post_scores_higher_than_generic_ai_post(samples):
    profile = build_profile(samples[1:])
    own = score_voice(samples[0], profile).score
    ai_posts = split_samples((DATA / "ai.md").read_text(encoding="utf-8"))
    assert all(own > score_voice(p, profile).score for p in ai_posts)


def test_few_samples_means_low_confidence(samples):
    assert score_voice(samples[0], build_profile(samples[:3])).confidence == "low"
    assert score_voice(samples[0], build_profile(samples)).confidence == "ok"


def test_deviations_name_the_worst_features_first(samples):
    profile = build_profile(samples)
    long_post = " ".join(["This is a very long and formal sentence about operational throughput."] * 6)
    report = score_voice(long_post, profile)
    zs = [abs(d.z) for d in report.deviations]
    assert zs == sorted(zs, reverse=True)
    assert 0 <= report.score <= 100
