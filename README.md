# proofpost

Drafts a founder's LinkedIn post from the company's own evidence, then **checks the draft before a human publishes it**.

Every factual line must cite a verbatim quote from a transcript, email or product note. Numbers are verified in code. Stock "AI phrases" are flagged. The voice is measured against the author's real posts. If a check fails, the model is shown exactly what failed and gets a bounded number of repair attempts. Nothing is ever published automatically.

Built as a proof-of-work for [Byro](https://byro.ee), whose pitch is a reputation workspace where "every recommendation shows its evidence" and a "named author approves before anything goes public". This repo is a small, working take on the *verification* half of that idea.

## See it in 30 seconds (no API key)

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Linux/macOS: .venv/bin/pip
proofpost run --demo      # flawed first draft gets caught, repaired draft passes
proofpost serve           # same thing in a browser at http://localhost:8000
proofpost bench           # voice-check benchmark
pytest
```

In demo mode the **model's answers are scripted** (`examples/tidewater/replay.json`, hand-written, not recorded) but **every check runs for real** on them. The scripted first draft contains an invented "80%", a stock phrase and an overstated claim, so you can watch each one get caught.

## Run it live

```bash
export DEEPSEEK_API_KEY=...        # PowerShell: $env:DEEPSEEK_API_KEY="..."
proofpost run --evidence ./my-docs --voice ./my-posts.md \
  --author "Maya Lindqvist" --goal "Be known for X among Y"
```

- `--evidence`: a folder of `.md` / `.txt` files (call transcripts, customer emails, release notes).
- `--voice`: the author's past posts in one file, separated by lines containing only `---` (5+ recommended).
- Exit code is `1` if the final draft is **blocked**, so it can gate a script or CI job.
- Default provider is DeepSeek (`deepseek-flash`). Any OpenAI-compatible API works by setting `PROOFPOST_BASE_URL`, `PROOFPOST_MODEL` and `PROOFPOST_API_KEY` (OpenRouter, Groq, Together, or a local Ollama/vLLM server). Model names change; check your provider's docs.

## How it works

```
evidence docs ─► split into numbered passages
                      │
        model: propose claims (each with a verbatim quote)
                      │
        CODE: keep a claim only if its quote is really in the passage ──► rest discarded
                      │
        model: draft the post, citing claim ids per line, in the author's voice
                      │
   ┌──────────────────┴───────────────────────────────────────────┐
   │ CODE   every number appears in the cited quote               │
   │ CODE   fact lines have citations; cited ids exist            │
   │ CODE   stock-phrase list; em-dash overuse                    │
   │ CODE   voice score vs the author's own posts (12 features)   │
   │ MODEL  judge: does the quote actually back the line?         │
   └──────────────────┬───────────────────────────────────────────┘
                      │ fail → feed the exact issues back, redraft (max 2 repairs)
                      ▼
        ok / review / blocked  +  evidence trail  →  a named human posts it
```

The principle: **the model writes, ordinary code decides.** A quote is either in the source or it is not; a number is either in the cited evidence or it is not. Only "does this quote support that wording?" needs a model, and that is a separate, narrow call that only grades.

| Verdict | Meaning |
|---|---|
| `ok` | No errors, no warnings, voice match above threshold. Ready for a human to read. |
| `review` | No factual errors, but style warnings remain (stock phrase, low voice match, unfamiliar name). |
| `blocked` | A number isn't in the evidence, a fact has no citation, or a line overstates its quote. Do not publish. |

## What it deliberately does not do

- **Publish.** No LinkedIn login, no posting, no scheduling. A named human copies the post.
- **Trust the model's citations.** Quotes are re-verified in code; made-up ones are dropped and shown in the report.
- **Follow instructions hidden in evidence.** Evidence is fenced as data in the prompts, and since a claim must be a verbatim quote, injected text cannot invent facts (see `test_prompt_injection_in_evidence_cannot_create_a_claim`).
- **Store anything.** The web app is stateless; evidence lives only in the request.

## Known limits (read these before trusting it)

- **Voice scoring measures style, not quality or truth.** It uses 12 surface features (sentence/line length, questions, emoji, contractions, ...). With under 5 sample posts it reports `low` confidence. It cannot see tone, humour or ideas.
- **Number check is strict and literal.** `10k` vs `10,000` counts as a mismatch. That errs toward blocking, on purpose.
- **Words and dates are only partly covered.** A made-up date ("last month") or an invented adjective is not caught by the code checks; only the model judge looks at wording. Unfamiliar capitalised names raise a warning.
- **The stock-phrase list is a visible, hand-written list** (`checks/generic.py`). It is easy to extend and easy to argue with, not a trained detector.
- **The judge is a model.** It can be wrong; its verdicts are shown to the human with reasons, never hidden.
- **Benchmark is small and synthetic** (below).

## Benchmark

`proofpost bench` scores each real post against a profile built from the author's *other* posts (leave-one-out), compared with the other author's posts and with generic AI-style posts.

| | mean voice score |
|---|---|
| author's own posts | 78 |
| the other author's posts | 51 |
| generic AI-style posts | 49 |

AUC is 1.00 on both comparisons, and the stock-phrase check flags 100% of the AI-style posts and 0% of the real ones. **Do not read that as an accuracy claim**: 14 real posts and 6 AI posts, all written for this project by the same hand that wrote the phrase list. It shows the scorer behaves sensibly. Validating it properly needs a real founder's posts and real model output; the data folder (`evals/data/`) and the benchmark are set up so that is a file swap.

## Layout

```
src/proofpost/
  pipeline.py      the whole flow (read this first)
  prompts.py       every prompt in one file
  llm.py           the only file that talks to a model (+ ScriptedLLM for tests/demo)
  evidence.py      loading, passage ids, "is this quote really there?"
  checks/          grounding.py, generic.py, voice.py
  report.py        human-readable output
  web.py, static/  FastAPI app + single-page UI
  cli.py, bench.py
tests/             56 tests, no network, no API key needed
examples/tidewater a fictional company: evidence, past posts, scripted model answers
evals/data         posts used by the benchmark
docs/DESIGN.md     decisions and trade-offs
```

All names in `examples/` (Tidewater, Maya Lindqvist, Harbor & Finch, ...) are fictional.

## Status

Tests, lint, the benchmark and the offline demo pass locally (56 tests) and are wired into a GitHub Actions workflow that has not run yet. The live call path (`OpenAICompatLLM`) has been run end to end against DeepSeek on the bundled example (about 20 seconds per run). In those runs the checks caught a draft inventing a product version label and another attributing a quote to the wrong person, and the repaired drafts passed. That is two runs on one fictional example, not an evaluation: expect to keep tuning prompts on real evidence, and models differ in how well they follow the line-count and no-new-numbers rules.
