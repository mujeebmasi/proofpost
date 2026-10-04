"""OpenAICompatLLM is tested against a stand-in client; no network, no API key."""

from types import SimpleNamespace

import pytest

from proofpost.llm import LLMError, OpenAICompatLLM
from proofpost.models import Draft

GOOD = '{"lines": [{"text": "hi", "kind": "hook", "claim_ids": []}]}'


def reply(content, finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))]
    )


def make_llm(*replies):
    """An LLM whose fake client hands back `replies` in order and records each request."""
    llm = OpenAICompatLLM(api_key="test-key")
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return replies[len(sent) - 1]

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm.sent = sent
    return llm


def ask(llm):
    return llm.structured(step="draft", system="Write.", user="Go.", schema=Draft)


def test_valid_json_is_parsed_and_the_request_asks_for_json_mode():
    llm = make_llm(reply(GOOD))
    assert ask(llm).lines[0].text == "hi"
    request = llm.sent[0]
    assert request["response_format"] == {"type": "json_object"}
    assert "json" in request["messages"][0]["content"].lower()
    assert '"claim_ids"' in request["messages"][0]["content"]  # the schema is in the prompt


def test_a_json_code_fence_is_tolerated():
    assert ask(make_llm(reply(f"```json\n{GOOD}\n```"))).lines[0].kind == "hook"


def test_empty_or_wrong_shape_is_retried_with_the_problem_explained():
    llm = make_llm(reply(""), reply('{"lines": "nope"}'), reply(GOOD))
    assert ask(llm).lines[0].text == "hi"
    assert len(llm.sent) == 3
    assert "not valid" in llm.sent[2]["messages"][-1]["content"]


def test_gives_up_after_three_bad_answers():
    with pytest.raises(LLMError, match="never matched"):
        ask(make_llm(reply(""), reply("{}"), reply("not json")))


@pytest.mark.parametrize(
    ("finish", "message"), [("length", "cut off"), ("content_filter", "content filter")]
)
def test_truncated_or_filtered_answers_become_clear_errors(finish, message):
    with pytest.raises(LLMError, match=message):
        ask(make_llm(reply(GOOD, finish)))


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("PROOFPOST_API_KEY", raising=False)
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        OpenAICompatLLM()


def test_provider_comes_from_environment(monkeypatch):
    monkeypatch.setenv("PROOFPOST_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("PROOFPOST_MODEL", "llama3")
    llm = OpenAICompatLLM(api_key="x")
    assert llm.model == "llama3"
    assert str(llm.client.base_url).startswith("http://localhost:11434/v1")
