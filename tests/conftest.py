from pathlib import Path

import pytest

from proofpost.checks.voice import split_samples
from proofpost.evidence import load_dir
from proofpost.llm import ScriptedLLM

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tidewater"


@pytest.fixture
def passages():
    return load_dir(EXAMPLE / "evidence")


@pytest.fixture
def samples():
    return split_samples((EXAMPLE / "voice.md").read_text(encoding="utf-8"))


@pytest.fixture
def goal():
    return (EXAMPLE / "goal.txt").read_text(encoding="utf-8").strip()


@pytest.fixture
def script():
    """A fresh copy of the demo script each time (ScriptedLLM consumes its answers)."""
    import json

    return json.loads((EXAMPLE / "replay.json").read_text(encoding="utf-8"))


@pytest.fixture
def demo_llm(script):
    return ScriptedLLM(script)
