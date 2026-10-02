"""AnthropicLLM is tested against a stand-in client; no network, no API key."""

from types import SimpleNamespace

import pytest

from proofpost.llm import AnthropicLLM, LLMError
from proofpost.models import Draft


def make_llm(response):
    llm = AnthropicLLM.__new__(AnthropicLLM)  # skip __init__: it would build a real client
    llm.model = "test-model"
    llm.client = SimpleNamespace(messages=SimpleNamespace(parse=lambda **_: response))
    return llm


def ask(llm):
    return llm.structured(step="draft", system="s", user="u", schema=Draft)


def test_returns_parsed_output():
    parsed = Draft(lines=[])
    assert ask(make_llm(SimpleNamespace(stop_reason="end_turn", parsed_output=parsed))) is parsed


@pytest.mark.parametrize(
    ("stop_reason", "parsed", "message"),
    [
        ("refusal", None, "declined"),
        ("max_tokens", None, "cut off"),
        ("end_turn", None, "expected shape"),
    ],
)
def test_bad_responses_become_clear_errors(stop_reason, parsed, message):
    with pytest.raises(LLMError, match=message):
        ask(make_llm(SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed)))
