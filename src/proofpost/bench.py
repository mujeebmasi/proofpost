"""A small, honest benchmark of the voice and stock-phrase checks.

Question it answers: can the voice score tell a person's real posts from (a) another person's
posts and (b) generic AI-style posts? Each real post is scored against a profile built from
the author's OTHER posts (leave-one-out), so no post is graded against itself.

The data is small and written for this project (evals/data/). Treat the result as "the
scorer behaves sensibly", not as a proven accuracy figure.
"""

from dataclasses import dataclass
from pathlib import Path

from .checks.generic import count_generic
from .checks.voice import build_profile, score_voice, split_samples


@dataclass
class BenchResult:
    own_mean: float
    other_author_mean: float
    ai_mean: float
    auc_vs_other_author: float
    auc_vs_ai: float
    ai_flagged_by_phrases: float  # share of AI posts with at least one stock-phrase hit
    own_flagged_by_phrases: float


def auc(positives: list[float], negatives: list[float]) -> float:
    """Chance that a random real post outscores a random fake one (0.5 = coin flip)."""
    pairs = [(p, n) for p in positives for n in negatives]
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs) / len(pairs)


def _load(path: Path) -> list[str]:
    return split_samples(path.read_text(encoding="utf-8"))


def run_bench(data_dir: Path) -> BenchResult:
    maya, dev, ai = _load(data_dir / "maya.md"), _load(data_dir / "dev.md"), _load(data_dir / "ai.md")

    own, other, fake = [], [], []
    for author_posts, others_posts in ((maya, dev), (dev, maya)):
        for i, post in enumerate(author_posts):
            profile = build_profile(author_posts[:i] + author_posts[i + 1 :])
            own.append(score_voice(post, profile).score)
            fake.append(score_voice(ai[i % len(ai)], profile).score)
        profile_all = build_profile(author_posts)
        other += [score_voice(p, profile_all).score for p in others_posts]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    return BenchResult(
        own_mean=mean(own),
        other_author_mean=mean(other),
        ai_mean=mean(fake),
        auc_vs_other_author=auc(own, other),
        auc_vs_ai=auc(own, fake),
        ai_flagged_by_phrases=sum(1 for p in ai if count_generic(p)) / len(ai),
        own_flagged_by_phrases=sum(1 for p in maya + dev if count_generic(p)) / len(maya + dev),
    )
