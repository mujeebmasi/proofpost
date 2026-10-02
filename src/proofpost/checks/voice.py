"""Measures how close a draft is to how one person actually writes.

Method (kept simple on purpose, so it can be explained in two minutes):
  1. For each sample post, measure 12 surface features (sentence length, line length, how
     often they ask questions, emoji, contractions, ...).
  2. For each feature, take the mean and the spread across the samples. That is the profile.
  3. For a new draft, measure the same features and ask "how many spreads away from this
     person's mean is it?" (a z-score). Average the distances, cap each at 3, turn into 0-100.

It sees style, not meaning. It will not tell you a post is true or good; the grounding checks
and the support judge handle truth. See docs/DESIGN.md for what this can and cannot detect.
"""

import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from ..models import VoiceDeviation, VoiceReport

MIN_SAMPLES_OK = 5  # fewer than this and the "usual spread" is a guess
Z_CAP = 3.0  # one wild feature should not zero the whole score

EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿⭐✅]")
WORD = re.compile(r"[A-Za-z0-9']+")
FIRST_PERSON = {"i", "i'm", "i've", "i'd", "i'll", "my", "me", "we", "we're", "we've", "our", "us"}


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p for p in (s.strip() for s in parts) if WORD.search(p)]


def _words(text: str) -> list[str]:
    return WORD.findall(text)


def _per100(count: int, words: int) -> float:
    return 100.0 * count / words if words else 0.0


@dataclass(frozen=True)
class Feature:
    label: str
    measure: Callable[[str], float]
    min_spread: float  # floor for the spread, so a feature that never varies cannot dominate


def _avg_sentence_words(t: str) -> float:
    s = _sentences(t)
    return statistics.fmean(len(_words(x)) for x in s) if s else 0.0


def _sentence_spread(t: str) -> float:
    lengths = [len(_words(x)) for x in _sentences(t)]
    return statistics.pstdev(lengths) if len(lengths) > 1 else 0.0


def _avg_line_words(t: str) -> float:
    ls = _lines(t)
    return statistics.fmean(len(_words(x)) for x in ls) if ls else 0.0


def _lowercase_start(t: str) -> float:
    ls = _lines(t)
    return 100.0 * sum(1 for x in ls if x[0].islower()) / len(ls) if ls else 0.0


def _questions(t: str) -> float:
    return _per100(t.count("?"), len(_words(t)))


def _exclaims(t: str) -> float:
    return _per100(t.count("!"), len(_words(t)))


def _emoji(t: str) -> float:
    return _per100(len(EMOJI.findall(t)), len(_words(t)))


def _first_person(t: str) -> float:
    w = [x.lower() for x in _words(t)]
    return _per100(sum(1 for x in w if x in FIRST_PERSON), len(w))


def _contractions(t: str) -> float:
    w = _words(t)
    return _per100(sum(1 for x in w if "'" in x.strip("'")), len(w))


def _avg_word_len(t: str) -> float:
    w = _words(t)
    return statistics.fmean(len(x) for x in w) if w else 0.0


def _numbers(t: str) -> float:
    w = _words(t)
    return _per100(sum(1 for x in w if x[0].isdigit()), len(w))


FEATURES: dict[str, Feature] = {
    "sentence_len": Feature("average sentence length (words)", _avg_sentence_words, 1.5),
    "sentence_var": Feature("variation in sentence length", _sentence_spread, 1.0),
    "line_len": Feature("average line length (words)", _avg_line_words, 1.5),
    "n_lines": Feature("number of lines", lambda t: float(len(_lines(t))), 1.0),
    "lowercase_start": Feature("lines starting lowercase (%)", _lowercase_start, 10.0),
    "questions": Feature("questions per 100 words", _questions, 0.5),
    "exclaims": Feature("exclamation marks per 100 words", _exclaims, 0.5),
    "emoji": Feature("emoji per 100 words", _emoji, 0.3),
    "first_person": Feature("first-person words per 100", _first_person, 1.0),
    "contractions": Feature("contractions per 100 words", _contractions, 0.8),
    "word_len": Feature("average word length (letters)", _avg_word_len, 0.15),
    "numbers": Feature("numbers per 100 words", _numbers, 0.5),
}


def measure(text: str) -> dict[str, float]:
    return {name: f.measure(text) for name, f in FEATURES.items()}


@dataclass
class Profile:
    means: dict[str, float]
    spreads: dict[str, float]
    n_samples: int


def build_profile(samples: list[str]) -> Profile:
    if len(samples) < 2:
        raise ValueError("Need at least 2 sample posts to learn a voice (5+ is better).")
    rows = [measure(s) for s in samples]
    means, spreads = {}, {}
    for name, feature in FEATURES.items():
        values = [r[name] for r in rows]
        means[name] = statistics.fmean(values)
        # Sample stdev, never below the feature's floor: with a handful of posts the spread
        # is noisy and could be near zero by luck.
        spreads[name] = max(statistics.stdev(values), feature.min_spread)
    return Profile(means=means, spreads=spreads, n_samples=len(samples))


def score_voice(text: str, profile: Profile, top: int = 4) -> VoiceReport:
    values = measure(text)
    deviations = []
    for name, feature in FEATURES.items():
        z = (values[name] - profile.means[name]) / profile.spreads[name]
        deviations.append(
            VoiceDeviation(
                feature=name,
                label=feature.label,
                draft=round(values[name], 1),
                typical=round(profile.means[name], 1),
                z=round(z, 2),
            )
        )
    distance = statistics.fmean(min(abs(d.z), Z_CAP) for d in deviations)
    score = round(100 * (1 - distance / Z_CAP))
    worst = sorted(deviations, key=lambda d: abs(d.z), reverse=True)[:top]
    confidence = "ok" if profile.n_samples >= MIN_SAMPLES_OK else "low"
    return VoiceReport(score=score, confidence=confidence, deviations=worst)


def split_samples(text: str) -> list[str]:
    """A voice file holds several posts separated by a line containing only `---`."""
    return [p.strip() for p in re.split(r"^\s*---\s*$", text, flags=re.MULTILINE) if p.strip()]
