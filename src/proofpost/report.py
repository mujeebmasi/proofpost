"""Turns a Result into something a human can read and sign off on."""

from .models import Attempt, Result

VERDICT_TEXT = {
    "ok": "READY FOR HUMAN REVIEW: all checks passed",
    "review": "NEEDS A LOOK: no factual errors, but style warnings remain",
    "blocked": "BLOCKED: factual problems remain, do not publish",
}


def _attempt_summary(n: int, a: Attempt) -> str:
    errors = sum(1 for i in a.issues if i.severity == "error")
    warnings = len(a.issues) - errors
    return (
        f"- Draft {n}: {a.verdict} (voice {a.voice.score}/100, {errors} error(s), "
        f"{warnings} warning(s))"
    )


def render_markdown(result: Result) -> str:
    final = result.final
    claims = {c.id: c for c in result.claims}
    out = [
        f"# Draft post for {result.author}",
        f"**Status:** {VERDICT_TEXT[final.verdict]}",
        f"**Goal:** {result.goal}",
        "",
        "## Post",
        "",
        final.text,
        "",
        "## Evidence trail",
        "",
    ]
    for i, line in enumerate(final.draft.lines, start=1):
        out.append(f"{i}. {line.text}")
        for cid in line.claim_ids:
            c = claims.get(cid)
            if c:
                out.append(f'   - {cid} ({c.source}): "{c.quote}"')
        if not line.claim_ids:
            out.append("   - no factual claim; author's own words")
    out += ["", "## Checks", "", f"Voice match: {final.voice.score}/100 ({final.voice.confidence} confidence)"]
    for d in final.voice.deviations:
        out.append(f"- {d.label}: draft {d.draft}, usual {d.typical}")
    if final.issues:
        out += ["", "Open issues:"] + [f"- [{i.severity}] {i.message}" for i in final.issues]
    else:
        out += ["", "No open issues."]

    out += ["", "## What the checks caught along the way", ""]
    out += [_attempt_summary(n, a) for n, a in enumerate(result.attempts, start=1)]
    for n, a in enumerate(result.attempts[:-1], start=1):
        for issue in a.issues:
            out.append(f"  - Draft {n}: {issue.message}")
    if result.rejected_claims:
        out += ["", f"{len(result.rejected_claims)} model-proposed claim(s) were discarded because "
                "their quote is not in the evidence:"]
        out += [f'- "{c.quote}"' for c in result.rejected_claims]
    out += ["", "_Nothing is published automatically. A named human approves and posts._"]
    return "\n".join(out)
