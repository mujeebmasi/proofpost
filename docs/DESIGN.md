# Design notes

## The problem

AI-written founder content fails in two ways: it **sounds generic**, and it **says things that aren't true** (an invented stat, an overstated result). Both destroy the one thing a founder-led brand sells: credibility. A tool that only generates faster makes both worse. So this project puts the effort into *checking*.

## Decisions

**1. Claims are quotes, verified in code.**
The model proposes claims, each with a verbatim `quote` and a `passage_id`. `evidence.quote_in` normalises case, whitespace and curly quotes, then does a substring match. No match, the claim is discarded and surfaced in the report. If the model named the wrong passage but the quote exists elsewhere, it is re-attributed. This turns "hallucinated evidence" from a prompt problem into a lookup.

**2. Numbers are checked by a regex, not by a model.**
Every number in a line must appear in the claims that line cites (`1,200`, `eleven`, `14%` are normalised). Derived numbers ("that's an 80% cut") are blocked too: the prompt forbids calculating, and the check enforces it. This is the single most common way generated posts lie, and it is cheap to catch exactly.

**3. One narrow model call grades support.**
Quote-to-wording faithfulness ("fell from 14% to 4%" vs "errors are basically gone") cannot be regexed. A separate `judge_support` call sees only the line and its cited quotes (not the goal, not the author voice) and returns supported / overstated / unsupported with a reason. Keeping it separate avoids the writer grading its own work.

**4. Voice is measured, not vibes.**
12 surface features, a per-author mean and spread, a capped z-score distance turned into 0-100. It is explainable in two minutes and every score comes with the biggest gaps in plain words ("average line length: draft 15, usual 11"). Spreads have per-feature floors because 5-8 posts gives noisy standard deviations. It is *not* a style model; see README limits.

**5. Bounded repair loop.**
On failure the exact issues go back to the model with the numbered previous draft. Max 2 repairs (configurable). The final attempt is reported as-is; a stubborn model ends `blocked`, not silently "fixed". Exit code 1 makes that scriptable.

**6. Human stays in the loop by construction.**
There is no publishing code. Each line in the output carries its quote and source file so a reviewer can verify in seconds instead of re-reading transcripts.

**7. Model access is one file behind one method.**
`LLM.structured(step, system, user, schema)` returns a validated pydantic object. The provider is anything speaking the OpenAI chat format (DeepSeek by default), chosen by environment variables. Most such providers offer "JSON mode", which guarantees valid JSON but not *our* shape, so the schema is put in the prompt and pydantic validates the reply; an empty or malformed answer is retried up to 3 times with the problem explained. Truncation and filtered answers become `LLMError`. `ScriptedLLM` implements the same method, which is why the demo and the tests run offline and the real checks still execute.

**8. Untrusted evidence.**
Transcripts and emails may contain instructions. Prompts fence evidence in tags and state it is data. More importantly, the architecture does not depend on the model resisting: a claim must be a verbatim quote, so injected text can't mint facts, and the judge only ever compares a line against quotes.

## Things I would do next, in order

1. **Run live with a real founder's posts and evidence**, tune the prompts, and replace the synthetic benchmark with real held-out data. This is the main gap.
2. **Learn from edits.** When a human rewrites a line, store the diff as a per-author rule ("never ends with a question") and enforce the ones that are checkable in code.
3. **Goal-aware selection.** Today the model picks which claims serve the goal; add an explicit scoring step so the choice is inspectable.
4. **Freshness and dates.** Attach dates to passages and flag stale claims or invented time references.
5. **Evidence ingestion** beyond paste: Google Docs / call-recording transcripts with speaker consent metadata.
6. **Persistence and auth** for multi-user use; right now the web app is intentionally stateless and rate-limited per IP to protect the API key.
