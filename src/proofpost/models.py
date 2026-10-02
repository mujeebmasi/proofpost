"""Data shapes shared by every step. Plain pydantic models, no behaviour."""

from typing import Literal

from pydantic import BaseModel

# --- evidence -------------------------------------------------------------


class Passage(BaseModel):
    """One paragraph of source material, e.g. a single turn of a call transcript."""

    id: str  # "<file-stem>-<n>", e.g. "call-harbor-finch-3"
    source: str  # file name shown to the reader
    text: str


class RawClaim(BaseModel):
    """What the model returns when asked to pull claims out of the evidence."""

    text: str  # the claim, in plain words
    quote: str  # copied exactly from the passage; the code checks this
    passage_id: str


class ExtractionResult(BaseModel):
    claims: list[RawClaim]


class Claim(BaseModel):
    """A claim whose quote the code has confirmed exists in the evidence."""

    id: str  # "C1", "C2", ...
    text: str
    quote: str
    passage_id: str
    source: str


# --- draft ----------------------------------------------------------------


class DraftLine(BaseModel):
    text: str
    kind: Literal["hook", "claim", "opinion", "question"]
    claim_ids: list[str]  # which verified claims back this line


class Draft(BaseModel):
    lines: list[DraftLine]

    def render(self) -> str:
        """The post as it would be pasted into LinkedIn: one blank line between lines."""
        return "\n\n".join(line.text.strip() for line in self.lines)


# --- checks ---------------------------------------------------------------


class Issue(BaseModel):
    severity: Literal["error", "warning"]
    code: str  # short machine-readable name, e.g. "unsupported_number"
    message: str  # plain-English explanation shown to the human and fed back to the model
    line: int | None = None  # 0-based index into Draft.lines, if it points at one line


class SupportVerdict(BaseModel):
    line: int
    verdict: Literal["supported", "overstated", "unsupported"]
    reason: str


class JudgeResult(BaseModel):
    verdicts: list[SupportVerdict]


class VoiceDeviation(BaseModel):
    feature: str
    label: str  # e.g. "average sentence length (words)"
    draft: float
    typical: float
    z: float  # how many "usual swings" away from this author's norm


class VoiceReport(BaseModel):
    score: int  # 0-100, higher means closer to the author's own posts
    confidence: Literal["ok", "low"]  # low when we have very few sample posts
    deviations: list[VoiceDeviation]  # the biggest mismatches, worst first


Verdict = Literal["ok", "review", "blocked"]


class Attempt(BaseModel):
    """One draft plus everything the checks said about it."""

    draft: Draft
    text: str
    issues: list[Issue]
    voice: VoiceReport
    support: list[SupportVerdict]
    verdict: Verdict


class Result(BaseModel):
    goal: str
    author: str
    claims: list[Claim]
    rejected_claims: list[RawClaim]  # model "quotes" that were not in the evidence
    attempts: list[Attempt]

    @property
    def final(self) -> Attempt:
        return self.attempts[-1]
