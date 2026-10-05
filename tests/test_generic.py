import pytest

from proofpost.checks.generic import check_generic, count_generic, find_generic
from proofpost.models import Draft, DraftLine


@pytest.mark.parametrize(
    "text",
    [
        "In today's fast-paced world, speed matters.",
        "A real game-changer for brokers.",
        "Let's delve into the data.",
        "It's not just software, it's a movement.",
        "I'm thrilled to announce our launch.",
        "What do you think?",
    ],
)
def test_stock_phrases_are_caught(text):
    assert find_generic(text)


@pytest.mark.parametrize(
    "text",
    [
        "his team used to spend about 11 minutes per shipment.",
        "we shipped a small thing today.",
        "the person who signs should be able to see why.",
    ],
)
def test_plain_writing_is_not_flagged(text):
    assert count_generic(text) == 0


def test_em_dash_overuse_counts_once_for_the_post():
    assert count_generic("one \u2014 two \u2014 three") == 1
    d = Draft(lines=[DraftLine(text="a \u2014 b \u2014 c", kind="opinion", claim_ids=[])])
    assert [i.code for i in check_generic(d)] == ["generic_phrase"]


@pytest.mark.parametrize(
    "text",
    ["the call stays with the broker. that's not a limitation, it's the point.", "This is not a tool, it is a teammate."],
)
def test_not_x_its_y_pattern_is_caught(text):
    assert any(rule == "not X, it's Y" for rule, _ in find_generic(text))
