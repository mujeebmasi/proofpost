"""The whole flow: extract claims -> draft -> check -> (repair) -> report."""

import logging
from dataclasses import dataclass

from . import prompts
from .checks.generic import check_generic
from .checks.grounding import check_grounding
from .checks.voice import Profile, build_profile, score_voice
from .evidence import normalize, quote_in
from .llm import LLM
from .models import (
    Attempt,
    Claim,
    Draft,
    ExtractionResult,
    Issue,
    JudgeResult,
    Passage,
    RawClaim,
    Result,
    SupportVerdict,
    Verdict,
)

log = logging.getLogger("proofpost")


class PipelineError(ValueError):
    """The run cannot continue. The message is safe to show the user."""


@dataclass
class Settings:
    max_repairs: int = 2  # extra drafts allowed after the first one fails
    min_voice_score: int = 60  # below this a draft needs a human look


# --- step 1: claims -----------------------------------------------------------


def extract_claims(
    passages: list[Passage], goal: str, llm: LLM
) -> tuple[list[Claim], list[RawClaim]]:
    """Ask the model for claims, then keep only those whose quote is really in the evidence."""
    raw = llm.structured(
        step="extract_claims",
        system=prompts.EXTRACT_SYSTEM,
        user=prompts.extract_user(passages, goal),
        schema=ExtractionResult,
    )
    by_id = {p.id: p for p in passages}
    kept: list[Claim] = []
    rejected: list[RawClaim] = []
    seen: set[str] = set()

    for item in raw.claims:
        passage = by_id.get(item.passage_id)
        if passage is None or not quote_in(item.quote, passage.text):
            # The model may have named the wrong passage; accept if another one has the quote.
            passage = next((p for p in passages if quote_in(item.quote, p.text)), None)
        key = normalize(item.quote)
        if passage is None:
            rejected.append(item)
        elif key not in seen:  # drop duplicates
            seen.add(key)
            kept.append(
                Claim(
                    id=f"C{len(kept) + 1}", text=item.text, quote=item.quote.strip(),
                    passage_id=passage.id, source=passage.source,
                )  # fmt: skip
            )
    if rejected:
        log.warning("Rejected %d claim(s) whose quote was not found in the evidence", len(rejected))
    if not kept:
        raise PipelineError(
            "No claim could be verified against the evidence. The documents may be too thin "
            "or too vague to write a grounded post from."
        )
    return kept, rejected


# --- step 2/3: draft and check --------------------------------------------------


def judge_support(draft: Draft, claims: list[Claim], llm: LLM) -> list[SupportVerdict]:
    """A second model call that only grades 'does the quote back this line?'."""
    if not any(line.claim_ids for line in draft.lines):
        return []
    result = llm.structured(
        step="judge_support",
        system=prompts.JUDGE_SYSTEM,
        user=prompts.judge_user(draft, claims),
        schema=JudgeResult,
    )
    return [v for v in result.verdicts if 0 <= v.line < len(draft.lines)]


def check_draft(
    draft: Draft,
    claims: list[Claim],
    passages: list[Passage],
    author: str,
    profile: Profile,
    llm: LLM,
    settings: Settings,
) -> Attempt:
    text = draft.render()
    issues: list[Issue] = []
    issues += check_grounding(draft, claims, passages, author)
    issues += check_generic(draft)

    support = judge_support(draft, claims, llm)
    for v in support:
        if v.verdict != "supported":
            issues.append(
                Issue(
                    severity="error", code=f"support_{v.verdict}", line=v.line,
                    message=f"Line {v.line + 1} is {v.verdict} by its evidence: {v.reason}",
                )  # fmt: skip
            )

    voice = score_voice(text, profile)
    if voice.score < settings.min_voice_score:
        worst = voice.deviations[0]
        issues.append(
            Issue(
                severity="warning", code="voice_mismatch",
                message=(
                    f"Voice match is {voice.score}/100 (minimum {settings.min_voice_score}). "
                    f"Biggest gap: {worst.label} is {worst.draft}, this author's usual is {worst.typical}."
                ),
            )  # fmt: skip
        )

    if any(i.severity == "error" for i in issues):
        verdict: Verdict = "blocked"
    elif issues:
        verdict = "review"
    else:
        verdict = "ok"
    return Attempt(
        draft=draft, text=text, issues=issues, voice=voice, support=support, verdict=verdict
    )


# --- the loop -----------------------------------------------------------------


def run(
    passages: list[Passage],
    goal: str,
    author: str,
    voice_samples: list[str],
    llm: LLM,
    settings: Settings | None = None,
) -> Result:
    settings = settings or Settings()
    if not goal.strip():
        raise PipelineError("Write the reputation goal first, e.g. 'be known for X among Y'.")
    try:
        profile = build_profile(voice_samples)
    except ValueError as err:
        raise PipelineError(str(err)) from err

    claims, rejected = extract_claims(passages, goal, llm)

    attempts: list[Attempt] = []
    previous: Draft | None = None
    issues: list[Issue] | None = None
    for _ in range(settings.max_repairs + 1):
        draft = llm.structured(
            step="draft",
            system=prompts.DRAFT_SYSTEM,
            user=prompts.draft_user(author, goal, claims, voice_samples, profile, previous, issues),
            schema=Draft,
        )
        if not draft.lines:
            raise PipelineError("The model returned an empty draft.")
        attempt = check_draft(draft, claims, passages, author, profile, llm, settings)
        attempts.append(attempt)
        if attempt.verdict == "ok":
            break
        previous, issues = draft, attempt.issues

    return Result(
        goal=goal, author=author, claims=claims, rejected_claims=rejected, attempts=attempts
    )
