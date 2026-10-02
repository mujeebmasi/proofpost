"""Flags the phrases that make a post read like it came out of a chatbot.

The list is deliberately short and visible. Anyone can read it, argue with it, and add to it.
"""

import re

from ..models import Draft, Issue

# (rule name, pattern). Matching is case-insensitive.
GENERIC_PATTERNS: list[tuple[str, str]] = [
    ("fast-paced world", r"\bin today'?s (fast[- ]paced|digital|ever[- ]changing|competitive)\b"),
    ("game-changer", r"\bgame[- ]?chang\w*"),
    ("delve", r"\bdelv(e|es|ing)\b"),
    ("unlock the power", r"\bunlock(ing|s)? the (power|potential)\b"),
    ("leverage", r"\bleverag(e|es|ing)\b"),
    ("at the end of the day", r"\bat the end of the day\b"),
    ("here's the thing", r"\bhere'?s the (thing|kicker|truth|secret)\b"),
    ("let that sink in", r"\blet that sink in\b"),
    ("it's not just X, it's Y", r"\bit'?s not (just|only|about)\b[^.\n]{1,60}(,|;|—|-)\s*it'?s\b"),
    ("seamless", r"\bseamless(ly)?\b"),
    ("cutting-edge", r"\bcutting[- ]edge\b"),
    ("revolutionize", r"\brevolutioni[sz]\w*"),
    ("thrilled to share", r"\b(thrilled|excited|humbled|proud) to (announce|share)\b"),
    ("dive into", r"\bdiv(e|ing) (deep )?into\b"),
    ("journey", r"\b(our|my|the) (\w+ )?journey\b"),
    ("landscape", r"\b(navigate|navigating) the\b[^.\n]{0,25}\blandscape\b"),
    ("testament", r"\btestament to\b"),
    ("elevate", r"\belevat(e|es|ing)\b"),
    ("empower", r"\bempower(s|ing)?\b"),
    ("synergy", r"\bsynerg\w+"),
    ("engagement bait", r"\b(thoughts\?|agree\?|what do you think\?)\s*$"),
]
COMPILED = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in GENERIC_PATTERNS]
MAX_EM_DASHES = 1  # a favourite tell of machine-written posts


def find_generic(text: str) -> list[tuple[str, str]]:
    """Return (rule, matched text) for each hit in one block of text."""
    return [(name, m.group().strip()) for name, rx in COMPILED for m in rx.finditer(text)]


def count_generic(text: str) -> int:
    """How many tells a whole post has: stock phrases, plus one for em-dash overuse."""
    return len(find_generic(text)) + (1 if text.count("—") > MAX_EM_DASHES else 0)


def check_generic(draft: Draft) -> list[Issue]:
    issues: list[Issue] = []
    for i, line in enumerate(draft.lines):
        for rule, matched in find_generic(line.text):
            issues.append(
                Issue(
                    severity="warning", code="generic_phrase", line=i,
                    message=f"Line {i + 1} uses a stock phrase ('{matched}', rule: {rule}). "
                    "Say the specific thing instead.",
                )  # fmt: skip
            )
    # em dashes are counted over the whole post, not per line
    whole = draft.render()
    if whole.count("—") > MAX_EM_DASHES:
        issues.append(
            Issue(
                severity="warning", code="generic_phrase",
                message=f"The post has {whole.count(chr(0x2014))} em dashes. Use at most {MAX_EM_DASHES}.",
            )  # fmt: skip
        )
    return issues
