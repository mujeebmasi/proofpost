"""Loading evidence and checking that a quote really appears in it."""

import re
from pathlib import Path

from .models import Passage

MAX_EVIDENCE_CHARS = 60_000  # about 15k tokens; refuse more rather than silently cut it
MIN_QUOTE_CHARS = 20  # a 3-word "quote" proves nothing
MIN_PASSAGE_CHARS = 15  # skips stray headings and blank-ish lines


class EvidenceError(ValueError):
    """The evidence is unusable (empty, too big). The message is safe to show the user."""


def normalize(text: str) -> str:
    """Make two strings comparable: straight quotes, one space, lower case."""
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    return re.sub(r"\s+", " ", text).strip().casefold()


def quote_in(quote: str, text: str) -> bool:
    """True if `quote` appears word-for-word in `text` (ignoring case and spacing)."""
    q = normalize(quote)
    return len(q) >= MIN_QUOTE_CHARS and q in normalize(text)


def split_text(source: str, text: str) -> list[Passage]:
    """Cut one document into paragraphs. Each paragraph becomes a citable passage."""
    stem = re.sub(r"[^a-z0-9]+", "-", Path(source).stem.lower()).strip("-") or "doc"
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
    kept = [p for p in paragraphs if len(p) >= MIN_PASSAGE_CHARS]
    return [Passage(id=f"{stem}-{i}", source=source, text=p) for i, p in enumerate(kept, start=1)]


def build_passages(documents: list[tuple[str, str]]) -> list[Passage]:
    """`documents` is a list of (file name, text). Raises EvidenceError if unusable."""
    passages: list[Passage] = []
    for name, text in documents:
        passages.extend(split_text(name, text))
    if not passages:
        raise EvidenceError("No evidence found. Add at least one document with real text.")
    total = sum(len(p.text) for p in passages)
    if total > MAX_EVIDENCE_CHARS:
        raise EvidenceError(
            f"Evidence is {total:,} characters; the limit is {MAX_EVIDENCE_CHARS:,}. "
            "Remove documents or trim them to the relevant parts."
        )
    return passages


def load_dir(folder: Path) -> list[Passage]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise EvidenceError(f"No .md or .txt files in {folder}")
    return build_passages([(f.name, f.read_text(encoding="utf-8")) for f in files])
