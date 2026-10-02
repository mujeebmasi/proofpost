"""The only file that talks to a model.

Everything else depends on the tiny `LLM` interface, so tests and the offline demo can swap
in `ScriptedLLM` and the real checks still run on whatever text comes back.
"""

import json
import os
from pathlib import Path
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5-5"


class LLMError(RuntimeError):
    """A model call failed. The message is safe to show the user."""


class LLM(Protocol):
    def structured(self, *, step: str, system: str, user: str, schema: type[T]) -> T:
        """Ask the model for an answer shaped like `schema`. `step` names the pipeline stage."""
        ...


class AnthropicLLM:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("PROOFPOST_MODEL", DEFAULT_MODEL)
        # The SDK already retries 429/5xx twice; the timeout stops one call hanging a request.
        self.client = anthropic.Anthropic(timeout=120.0)

    def structured(self, *, step: str, system: str, user: str, schema: type[T]) -> T:
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except anthropic.AuthenticationError as err:
            raise LLMError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.") from err
        except anthropic.RateLimitError as err:
            raise LLMError("Anthropic rate limit hit. Wait a minute and try again.") from err
        except anthropic.APIConnectionError as err:
            raise LLMError("Could not reach the Anthropic API. Check your connection.") from err
        except anthropic.APIStatusError as err:
            raise LLMError(f"Anthropic API error ({err.status_code}) during '{step}'.") from err

        if response.stop_reason == "refusal":
            raise LLMError(f"The model declined the '{step}' request.")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"The model's answer for '{step}' was cut off. Try less evidence.")
        if response.parsed_output is None:
            raise LLMError(f"The model's answer for '{step}' did not match the expected shape.")
        return response.parsed_output


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
