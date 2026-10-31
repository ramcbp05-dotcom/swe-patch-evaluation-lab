"""``patchlab``: list, inspect and evaluate the bug-fixing cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import harness, models

CASES_DIR = harness.REPO_ROOT / "cases"


def _load(selected: list[str] | None) -> list[models.CaseSpec]:
    specs = models.discover(CASES_DIR)
    if not selected:
        return specs
    by_id = {s.case_id: s for s in specs}
    missing = [c for c in selected if c not in by_id]
    if missing:
        raise SystemExit(f"unknown case id(s): {missing}\navailable: {sorted(by_id)}")
    return [by_id[c] for c in selected]


def cmd_list(args: argparse.Namespace) -> int:
    specs = _load(None)
    width = max(len(s.case_id) for s in specs)
    print(f"{'CASE'.ljust(width)}  DIFF    BUG CLASS                 SEVERITY              CANDIDATES")
    for spec in specs:
        n_ok = sum(c.kind == "correct" for c in spec.candidates)
        n_bad = sum(c.kind == "incorrect" for c in spec.candidates)
        print(f"{spec.case_id.ljust(width)}  {spec.difficulty:<6}  {spec.bug_class:<24}  "
              f"{spec.severity:<20}  {n_ok} correct / {n_bad} incorrect")
    print(f"\n{len(specs)} cases, "
          f"{sum(len(s.candidates) for s in specs)} candidate patches "
          f"({sum(sum(c.kind == 'incorrect' for c in s.candidates) for s in specs)} "
          "deliberately wrong)")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    for spec in _load(args.case):
        print("=" * 78)
        print(f"{spec.case_id}  [{spec.difficulty}/{spec.bug_class}/{spec.severity}]")
        print("=" * 78)
        print(spec.issue_path.read_text())
        if args.notes:
            print("-" * 78)
            print(spec.notes_path.read_text())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    specs = _load(args.case)
    keep_root = Path(args.keep_logs) if args.keep_logs else None

    results: list[harness.CaseResult] = []
    for spec in specs:
        if args.candidate:
            runs = [harness.evaluate(spec)] + [
                harness.evaluate(spec, name) for name in args.candidate
            ]
        else:
            runs = harness.evaluate_case(spec, keep_root=keep_root)

        for res in runs:
            flag = "OK " if res.calibrated else "BAD"
            label = res.candidate or "(baseline)"
            if res.candidate is None:
                verdict = "reproduces" if res.reproduced else "NO REPRO"
                extra = (f"p2p {res.pass_to_pass['n_passed']}p/"
                         f"{res.pass_to_pass['n_failed']}f")
            else:
                verdict = "resolved" if res.resolved else "unresolved"
                extra = res.detail
            print(f"[{flag}] {spec.case_id:<34} {label:<22} {verdict:<11} {extra}")
        results.extend(runs)

    summary = harness.summarise(results)
    print(f"\n{summary['resolved']}/{summary['n_candidate_runs']} candidate patches "
          f"resolved their case; "
          f"{summary['calibrated']}/{summary['total_runs']} runs calibrated as expected")
    for bad in summary["miscalibrated"]:
        print(f"  MISCALIBRATED {bad['case']} / {bad['candidate'] or '(baseline)'}: "
              f"{bad['detail']}")

    if args.report:
        harness.write_report(summary, Path(args.report))
        print(f"report written to {args.report}")
    return 0 if not summary["miscalibrated"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patchlab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list every case").set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="print a case's issue report")
    p_show.add_argument("case", nargs="+")
    p_show.add_argument("--notes", action="store_true",
                        help="also print the root-cause analysis (spoilers)")
    p_show.set_defaults(func=cmd_show)

    p_eval = sub.add_parser("evaluate", help="evaluate candidate patches")
    p_eval.add_argument("-c", "--case", action="append", help="case id (default all)")
    p_eval.add_argument("-p", "--candidate", action="append",
                        help="candidate patch name (default all declared)")
    p_eval.add_argument("--report", default=None, help="write a JSON summary here")
    p_eval.add_argument("--keep-logs", default=None,
                        help="directory to keep per-candidate pytest output in")
    p_eval.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
