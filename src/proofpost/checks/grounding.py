"""Plain-code checks that a draft only says what the evidence says.

No model is involved here. These run in milliseconds and cannot be talked out of a verdict.
"""

import re

from ..models import Claim, Draft, Issue, Passage

NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "fifty": 50, "hundred": 100, "thousand": 1000,
}  # fmt: skip
# "one" is left out on purpose: "no one", "one thing" would be false alarms.

# A digit run that is not glued to a letter ("B2B" is not a number).
DIGITS = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Every number in `text` as a normal string: '1,200' -> '1200', 'eleven' -> '11'."""
    found = {m.group().replace(",", "").rstrip(".") for m in DIGITS.finditer(text)}
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in NUMBER_WORDS:
            found.add(str(NUMBER_WORDS[word]))
    return found


# Capitalised words that are fine without evidence.
COMMON_CAPS = {"i", "linkedin", "ai", "ceo", "b2b", "saas", "monday", "tuesday", "wednesday",
               "thursday", "friday", "saturday", "sunday", "january", "february", "march",
               "april", "may", "june", "july", "august", "september", "october", "november",
               "december"}  # fmt: skip


def _mid_sentence_capitals(text: str) -> list[str]:
    """Capitalised words that are not simply the first word of a sentence."""
    words = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        tokens = re.findall(r"[A-Za-z][A-Za-z'&-]*", sentence)
        words.extend(t for t in tokens[1:] if t[0].isupper())
    return words


def check_grounding(
    draft: Draft, claims: list[Claim], passages: list[Passage], author: str
) -> list[Issue]:
    by_id = {c.id: c for c in claims}
    corpus = " ".join(p.text for p in passages).lower()
    author_words = {w.lower() for w in re.findall(r"[A-Za-z]+", author)}
    issues: list[Issue] = []

    for i, line in enumerate(draft.lines):
        unknown = [cid for cid in line.claim_ids if cid not in by_id]
        if unknown:
            issues.append(
                Issue(
                    severity="error", code="unknown_claim", line=i,
                    message=f"Line {i + 1} cites {', '.join(unknown)}, which is not a verified claim.",
                )  # fmt: skip
            )
        cited = [by_id[cid] for cid in line.claim_ids if cid in by_id]

        line_numbers = numbers_in(line.text)
        if (line.kind == "claim" or line_numbers) and not cited:
            issues.append(
                Issue(
                    severity="error", code="uncited_claim", line=i,
                    message=f"Line {i + 1} states a fact or number but cites no evidence.",
                )  # fmt: skip
            )

        allowed: set[str] = set()
        for claim in cited:
            allowed |= numbers_in(claim.text) | numbers_in(claim.quote)
        for number in sorted(line_numbers - allowed):
            issues.append(
                Issue(
                    severity="error", code="unsupported_number", line=i,
                    message=(
                        f"Line {i + 1} uses the number {number}, which is not in the evidence "
                        "it cites. Use only numbers that appear in the cited claim."
                    ),
                )  # fmt: skip
            )

        for word in _mid_sentence_capitals(line.text):
            lowered = word.lower().strip("'")
            if lowered in COMMON_CAPS or lowered in author_words or lowered in corpus:
                continue
            issues.append(
                Issue(
                    severity="warning", code="unknown_name", line=i,
                    message=f"Line {i + 1} mentions '{word}', which does not appear in the evidence.",
                )  # fmt: skip
            )
    return issues
