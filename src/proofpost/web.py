"""Small web app: paste evidence and past posts in the browser, get a checked draft back.

The API key stays on the server. There is no database and nothing is stored: each request
carries its own evidence and returns its own result.
"""

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .checks.voice import split_samples
from .cli import EXAMPLE
from .evidence import EvidenceError, build_passages
from .llm import LLMError, OpenAICompatLLM, ScriptedLLM, configured_api_key
from .models import Result
from .pipeline import PipelineError, run
from .report import render_markdown

log = logging.getLogger("proofpost.web")
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="proofpost", docs_url=None, redoc_url=None)

# Live runs cost real money, so each client IP gets a small hourly allowance.
RUNS_PER_HOUR = int(os.environ.get("PROOFPOST_RUNS_PER_HOUR", "10"))
_recent: dict[str, deque[float]] = defaultdict(deque)


def _check_rate_limit(ip: str) -> None:
    now, window = time.time(), _recent[ip]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= RUNS_PER_HOUR:
        raise HTTPException(429, "Hourly limit reached. Try again later.")
    window.append(now)


class Document(BaseModel):
    name: str = Field(max_length=120)
    text: str = Field(max_length=60_000)


class RunRequest(BaseModel):
    author: str = Field(min_length=1, max_length=80)
    goal: str = Field(min_length=1, max_length=400)
    voice: str = Field(max_length=30_000)  # past posts separated by --- lines
    documents: list[Document] = Field(min_length=1, max_length=20)


class RunResponse(BaseModel):
    result: Result
    markdown: str


def _execute(req: RunRequest, llm) -> RunResponse:
    try:
        passages = build_passages([(d.name, d.text) for d in req.documents])
        result = run(passages, req.goal, req.author, split_samples(req.voice), llm)
    except (EvidenceError, PipelineError) as err:
        raise HTTPException(422, str(err)) from err
    except LLMError as err:
        log.warning("model call failed: %s", err)
        raise HTTPException(502, str(err)) from err
    return RunResponse(result=result, markdown=render_markdown(result))


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/example")
def example() -> RunRequest:
    """The bundled example, so the page can pre-fill its form."""
    docs = [
        Document(name=f.name, text=f.read_text(encoding="utf-8"))
        for f in sorted((EXAMPLE / "evidence").glob("*.md"))
    ]
    return RunRequest(
        author="Maya Lindqvist",
        goal=(EXAMPLE / "goal.txt").read_text(encoding="utf-8").strip(),
        voice=(EXAMPLE / "voice.md").read_text(encoding="utf-8"),
        documents=docs,
    )


@app.post("/api/demo")
def demo(req: RunRequest) -> RunResponse:
    """Runs the real checks on scripted model answers. Free, works with no API key."""
    return _execute(req, ScriptedLLM.from_file(EXAMPLE / "replay.json"))


@app.post("/api/run")
def live(req: RunRequest, request: Request) -> RunResponse:
    if not configured_api_key():
        raise HTTPException(503, "No model API key on the server. Use demo mode.")
    _check_rate_limit(request.client.host if request.client else "unknown")
    try:
        llm = OpenAICompatLLM()
    except LLMError as err:
        raise HTTPException(503, str(err)) from err
    return _execute(req, llm)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
