"""The only file that talks to a model.

Everything else depends on the tiny `LLM` interface, so tests and the offline demo can swap
in `ScriptedLLM` and the real checks still run on whatever text comes back.

`OpenAICompatLLM` works with any provider that speaks the OpenAI chat format: DeepSeek (the
default), OpenRouter, Groq, Together, a local Ollama or vLLM server. Switch provider with
three environment variables; no code changes.
"""

import json
import os
import re
from pathlib import Path
from typing import Protocol, TypeVar

import openai
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"  # check your provider's docs; names change
MAX_TRIES = 3  # JSON mode can return empty or malformed text now and then


class LLMError(RuntimeError):
    """A model call failed. The message is safe to show the user."""


class LLM(Protocol):
    def structured(self, *, step: str, system: str, user: str, schema: type[T]) -> T:
        """Ask the model for an answer shaped like `schema`. `step` names the pipeline stage."""
        ...


def configured_api_key() -> str | None:
    """PROOFPOST_API_KEY wins; DEEPSEEK_API_KEY is the convenient default."""
    return os.environ.get("PROOFPOST_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")


def _parse_json(text: str, schema: type[T]) -> T:
    """Validate the model's text against `schema`, tolerating a ```json fence."""
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    return schema.model_validate_json(text)


class OpenAICompatLLM:
    def __init__(
        self, model: str | None = None, base_url: str | None = None, api_key: str | None = None
    ) -> None:
        key = api_key or configured_api_key()
        if not key:
            raise LLMError("No API key. Set DEEPSEEK_API_KEY (or PROOFPOST_API_KEY).")
        self.model = model or os.environ.get("PROOFPOST_MODEL", DEFAULT_MODEL)
        # The SDK already retries 429/5xx twice; the timeout stops one call hanging a request.
        self.client = openai.OpenAI(
            api_key=key,
            base_url=base_url or os.environ.get("PROOFPOST_BASE_URL", DEFAULT_BASE_URL),
            timeout=120.0,
        )

    def structured(self, *, step: str, system: str, user: str, schema: type[T]) -> T:
        # "JSON mode" only promises valid JSON, not our shape. So the schema goes into the
        # prompt (DeepSeek also requires the word "json" there) and pydantic checks the result.
        system = (
            f"{system}\n\nReply with one json object and nothing else. "
            f"It must match this JSON Schema:\n{json.dumps(schema.model_json_schema())}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        problem = ""
        for _ in range(MAX_TRIES):
            text = self._complete(step, messages)
            try:
                return _parse_json(text, schema)
            except (ValidationError, ValueError) as err:
                problem = str(err)[:300]
                messages += [
                    {"role": "assistant", "content": text or "(empty)"},
                    {"role": "user", "content": f"That was not valid. Fix it and return only the json. Problem: {problem}"},
                ]
        raise LLMError(f"The model's answer for '{step}' never matched the expected shape.")

    def _complete(self, step: str, messages: list[dict]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=8000,
            )
        except openai.AuthenticationError as err:
            raise LLMError("The provider rejected the API key. Check DEEPSEEK_API_KEY.") from err
        except openai.RateLimitError as err:
            raise LLMError("Provider rate limit hit. Wait a minute and try again.") from err
        except openai.APIConnectionError as err:
            raise LLMError("Could not reach the model provider. Check your connection.") from err
        except openai.APIStatusError as err:
            raise LLMError(f"Model API error ({err.status_code}) during '{step}'.") from err

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(f"The model's answer for '{step}' was cut off. Try less evidence.")
        if choice.finish_reason == "content_filter":
            raise LLMError(f"The provider's content filter blocked the '{step}' request.")
        return choice.message.content or ""


class ScriptedLLM:
    """Returns pre-written answers, in order, per step. Used by tests and `--demo`.

    These answers are written by hand (see examples/), not recorded from a model. The point
    is to run the real checks on a known-flawed draft and then a repaired one.
    """

    def __init__(self, script: dict[str, list[dict]]) -> None:
        self._script = {step: list(answers) for step, answers in script.items()}
        self.calls: list[str] = []

    @classmethod
    def from_file(cls, path: Path) -> "ScriptedLLM":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def structured(self, *, step: str, system: str, user: str, schema: type[T]) -> T:
        self.calls.append(step)
        answers = self._script.get(step)
        if not answers:
            raise LLMError(f"Scripted model has no answer left for step '{step}'.")
        return schema.model_validate(answers.pop(0))
