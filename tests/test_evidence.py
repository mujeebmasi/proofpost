import pytest

from proofpost.evidence import MAX_EVIDENCE_CHARS, EvidenceError, build_passages, quote_in


def test_quote_matches_ignoring_case_spacing_and_curly_quotes():
    text = "Joon: We didn\u2019t   trust it\nat first, honestly."
    assert quote_in("we didn't trust it at first, honestly", text)


def test_quote_must_be_word_for_word():
    text = "Our first-pass rejection rate was 14 percent."
    assert not quote_in("Our first-pass rejection rate was 40 percent.", text)
    assert not quote_in("rejection rate was fourteen percent", text)


def test_very_short_quote_is_not_accepted():
    assert not quote_in("14 percent", "Our first-pass rejection rate was 14 percent.")


def test_passages_get_stable_ids_and_skip_tiny_fragments():
    ps = build_passages([("My Notes.md", "# T\n\nThis paragraph is long enough.\n\nAnother long paragraph here.")])
    assert [p.id for p in ps] == ["my-notes-1", "my-notes-2"]


def test_empty_evidence_is_refused():
    with pytest.raises(EvidenceError):
        build_passages([("a.md", "   ")])


def test_oversized_evidence_is_refused_not_truncated():
    big = ("word " * 30 + "\n\n") * (MAX_EVIDENCE_CHARS // 150 + 50)
    with pytest.raises(EvidenceError, match="limit"):
        build_passages([("big.md", big)])
