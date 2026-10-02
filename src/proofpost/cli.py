"""Command line: `proofpost run | voice-score | bench | serve`."""

import argparse
import logging
import sys
from pathlib import Path

from .bench import run_bench
from .checks.voice import build_profile, score_voice, split_samples
from .evidence import EvidenceError, load_dir
from .llm import LLM, AnthropicLLM, LLMError, ScriptedLLM
from .pipeline import PipelineError, Settings, run
from .report import render_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "tidewater"
BENCH_DATA = REPO_ROOT / "evals" / "data"


def _cmd_run(args: argparse.Namespace) -> int:
    if args.demo:
        evidence_dir, voice_file = EXAMPLE / "evidence", EXAMPLE / "voice.md"
        goal = (EXAMPLE / "goal.txt").read_text(encoding="utf-8").strip()
        author = "Maya Lindqvist"
        llm: LLM = ScriptedLLM.from_file(EXAMPLE / "replay.json")
        print("(demo: model answers are scripted from examples/tidewater/replay.json)\n", file=sys.stderr)
    else:
        missing = [n for n in ("evidence", "voice", "goal", "author") if not getattr(args, n)]
        if missing:
            print(f"error: missing --{', --'.join(missing)} (or use --demo)", file=sys.stderr)
            return 2
        evidence_dir, voice_file, goal, author = args.evidence, args.voice, args.goal, args.author
        llm = AnthropicLLM(args.model)

    passages = load_dir(Path(evidence_dir))
    samples = split_samples(Path(voice_file).read_text(encoding="utf-8"))
    result = run(passages, goal, author, samples, llm, Settings(max_repairs=args.repairs))

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(render_markdown(result))
    # exit code lets CI or a script refuse to continue on a blocked draft
    return 1 if result.final.verdict == "blocked" else 0


def _cmd_voice_score(args: argparse.Namespace) -> int:
    profile = build_profile(split_samples(Path(args.voice).read_text(encoding="utf-8")))
    report = score_voice(Path(args.post).read_text(encoding="utf-8"), profile)
    print(f"Voice match: {report.score}/100 ({report.confidence} confidence)")
    for d in report.deviations:
        print(f"  {d.label}: post {d.draft}, usual {d.typical}")
    return 0


def _cmd_bench(_: argparse.Namespace) -> int:
    r = run_bench(BENCH_DATA)
    print(f"Mean voice score, author's own posts (leave-one-out): {r.own_mean:.0f}")
    print(f"Mean voice score, the other author's posts:           {r.other_author_mean:.0f}")
    print(f"Mean voice score, generic AI-style posts:             {r.ai_mean:.0f}")
    print(f"AUC own vs other author: {r.auc_vs_other_author:.2f}   own vs AI: {r.auc_vs_ai:.2f}")
    print(f"Stock-phrase check flags {r.ai_flagged_by_phrases:.0%} of AI posts, {r.own_flagged_by_phrases:.0%} of real posts")
    print("Small synthetic set. Shows the scorer behaves sensibly; it is not an accuracy claim.")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("proofpost.web:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proofpost", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="draft a post from evidence and verify it")
    p.add_argument("--evidence", help="folder of .md/.txt source documents")
    p.add_argument("--voice", help="file of the author's past posts, separated by --- lines")
    p.add_argument("--goal", help="the reputation goal, in one sentence")
    p.add_argument("--author", help="the named author")
    p.add_argument("--model", help="override PROOFPOST_MODEL")
    p.add_argument("--repairs", type=int, default=2, help="extra drafts allowed after a failure")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--demo", action="store_true", help="run offline on the bundled example")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("voice-score", help="score one post against an author's past posts")
    p.add_argument("--voice", required=True)
    p.add_argument("--post", required=True)
    p.set_defaults(func=_cmd_voice_score)

    p = sub.add_parser("bench", help="run the voice-check benchmark")
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("serve", help="start the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (EvidenceError, PipelineError, LLMError, OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
