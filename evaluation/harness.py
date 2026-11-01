"""Apply a seeded bug, apply a candidate patch, run the two test sets.

The evaluation protocol follows the SWE-bench shape: a case declares a
**fail-to-pass** set (the tests that encode the issue) and a **pass-to-pass**
set (the project's existing suite, which a fix must not regress). A candidate
*resolves* the case only when both sets pass.

Two things are checked that an oracle-only run would miss:

* **Reproduction.** Before any candidate is applied, the fail-to-pass set must
  fail. A case whose issue tests pass on the broken library is not testing the
  bug.
* **Discrimination.** Every case ships at least one deliberately incorrect
  candidate that must *not* resolve it. Without that, a case only shows that
  the gold patch works, not that the tests would catch anything else.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .models import CaseSpec

PACKAGE_ROOT = Path(__file__).resolve().parent
IMPORT_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = IMPORT_ROOT if (IMPORT_ROOT / "cases").is_dir() else PACKAGE_ROOT / "_data"
LIB_TESTS = "tests"
CASE_TESTS = "case_tests"


class HarnessError(RuntimeError):
    """A failure in the harness itself, not in the code under evaluation."""


@dataclasses.dataclass
class TestOutcome:
    passed: bool
    returncode: int
    n_passed: int
    n_failed: int
    tail: str

    @classmethod
    def from_process(cls, proc: subprocess.CompletedProcess) -> "TestOutcome":
        text = proc.stdout + proc.stderr
        n_passed = _count(text, "passed")
        n_failed = _count(text, "failed")
        if proc.returncode not in (0, 1) or n_passed + n_failed == 0 or (
            proc.returncode == 1 and n_failed == 0
        ) or _has_pytest_errors(text):
            raise HarnessError(
                f"pytest did not complete a valid test run (exit {proc.returncode}):\n{text[-4000:]}"
            )
        return cls(
            passed=proc.returncode == 0,
            returncode=proc.returncode,
            n_passed=n_passed,
            n_failed=n_failed,
            tail="\n".join(text.splitlines()[-40:]),
        )

    def as_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data.pop("tail")
        return data


def _has_pytest_errors(text: str) -> bool:
    import re
    return bool(re.search(r"\b\d+ errors?\b", text))


def _count(text: str, word: str) -> int:
    import re
    hits = re.findall(rf"(\d+) {word}", text)
    return int(hits[-1]) if hits else 0


@dataclasses.dataclass
class CaseResult:
    case_id: str
    candidate: str | None
    kind: str
    applied: bool
    reproduced: bool
    fail_to_pass: dict | None
    pass_to_pass: dict | None
    resolved: bool
    expected_resolved: bool | None
    duration_sec: float
    detail: str = ""

    @property
    def calibrated(self) -> bool:
        """Did this run do what the case says it should?"""
        if not self.reproduced:
            return False
        if self.expected_resolved is None:
            return True
        return self.resolved is self.expected_resolved

    def as_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["calibrated"] = self.calibrated
        return data


def _run_pytest(workdir: Path, target: str, extra: list[str] | None = None):
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header",
         "-p", "no:cacheprovider", target, *(extra or [])],
        cwd=str(workdir), capture_output=True, text=True, timeout=600,
        env={**_clean_env(), "PYTHONPATH": str(workdir / "src")},
    )
    return TestOutcome.from_process(proc)


def _clean_env() -> dict:
    import os
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    return env


def _apply(workdir: Path, patch: Path, reverse: bool = False) -> tuple[bool, str]:
    """Apply a unified diff with ``git apply``. Works outside a git repository."""
    argv = ["git", "apply", "--whitespace=nowarn", "-p1"]
    if reverse:
        argv.append("--reverse")
    argv.append(str(patch))
    proc = subprocess.run(argv, cwd=str(workdir), capture_output=True, text=True,
                          timeout=120)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def _prepare(spec: CaseSpec, workdir: Path) -> None:
    """Copy the library, its suite and the case's issue tests into a scratch tree."""
    shutil.copytree(REPO_ROOT / "src", workdir / "src")
    shutil.copytree(REPO_ROOT / LIB_TESTS, workdir / LIB_TESTS)
    shutil.copytree(spec.issue_tests, workdir / CASE_TESTS)
    # The issue tests reuse the library suite's fixtures.
    shutil.copy2(REPO_ROOT / LIB_TESTS / "conftest.py", workdir / CASE_TESTS / "conftest.py")


def evaluate(spec: CaseSpec, candidate_name: str | None = None,
             keep: Path | None = None) -> CaseResult:
    """Evaluate one candidate against one case, or the broken baseline."""
    candidate = None
    if candidate_name is not None:
        matches = [c for c in spec.candidates if c.name == candidate_name]
        if not matches:
            raise HarnessError(
                f"{spec.case_id}: unknown candidate {candidate_name!r}; "
                f"available: {[c.name for c in spec.candidates]}"
            )
        candidate = matches[0]

    started = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix=f"patchlab-{spec.case_id}-"))
    try:
        workdir = tmp / "repo"
        workdir.mkdir()
        _prepare(spec, workdir)

        ok, message = _apply(workdir, spec.bug_patch)
        if not ok:
            raise HarnessError(
                f"{spec.case_id}: bug.patch does not apply to the current library.\n"
                f"{message}\nRegenerate it with scripts/build_patches.py."
            )

        # Reproduction: the issue tests must fail on the broken library.
        broken_f2p = _run_pytest(workdir, CASE_TESTS)
        reproduced = broken_f2p.returncode == 1 and broken_f2p.n_failed > 0
        broken_p2p = _run_pytest(workdir, LIB_TESTS)

        result = CaseResult(
            case_id=spec.case_id, candidate=None, kind="baseline",
            applied=True, reproduced=reproduced,
            fail_to_pass=broken_f2p.as_dict(), pass_to_pass=broken_p2p.as_dict(),
            resolved=False, expected_resolved=None,
            duration_sec=round(time.perf_counter() - started, 2),
            detail="" if reproduced else
                   "issue tests PASS on the broken library: the case does not "
                   "reproduce its own bug",
        )
        if candidate is None:
            return result

        applied, message = _apply(workdir, spec.candidate_patch(candidate.name))
        if not applied:
            return CaseResult(
                case_id=spec.case_id, candidate=candidate.name, kind=candidate.kind,
                applied=False, reproduced=reproduced,
                fail_to_pass=None, pass_to_pass=None, resolved=False,
                expected_resolved=candidate.expected_resolved,
                duration_sec=round(time.perf_counter() - started, 2),
                detail=f"patch did not apply: {message}",
            )

        f2p = _run_pytest(workdir, CASE_TESTS)
        p2p = _run_pytest(workdir, LIB_TESTS)
        resolved = f2p.passed and p2p.passed

        if keep is not None:
            keep.mkdir(parents=True, exist_ok=True)
            (keep / "fail_to_pass.log").write_text(f2p.tail + "\n")
            (keep / "pass_to_pass.log").write_text(p2p.tail + "\n")

        return CaseResult(
            case_id=spec.case_id, candidate=candidate.name, kind=candidate.kind,
            applied=True, reproduced=reproduced,
            fail_to_pass=f2p.as_dict(), pass_to_pass=p2p.as_dict(),
            resolved=resolved, expected_resolved=candidate.expected_resolved,
            duration_sec=round(time.perf_counter() - started, 2),
            detail=_explain(f2p, p2p),
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _explain(f2p: TestOutcome, p2p: TestOutcome) -> str:
    if f2p.passed and p2p.passed:
        return "fail-to-pass and pass-to-pass both green"
    parts = []
    if not f2p.passed:
        parts.append(f"{f2p.n_failed} issue test(s) still failing")
    if not p2p.passed:
        parts.append(f"{p2p.n_failed} existing test(s) regressed")
    return "; ".join(parts)


def evaluate_case(spec: CaseSpec, keep_root: Path | None = None) -> list[CaseResult]:
    """Baseline plus every declared candidate for one case."""
    results = [evaluate(spec)]
    for candidate in spec.candidates:
        keep = None if keep_root is None else keep_root / spec.case_id / candidate.name
        results.append(evaluate(spec, candidate.name, keep=keep))
    return results


def summarise(results: list[CaseResult]) -> dict:
    graded = [r for r in results if r.expected_resolved is not None]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": sorted({r.case_id for r in results}),
        "n_cases": len({r.case_id for r in results}),
        "n_candidate_runs": len(graded),
        "resolved": sum(r.resolved for r in graded),
        "calibrated": sum(r.calibrated for r in results),
        "total_runs": len(results),
        "miscalibrated": [
            {"case": r.case_id, "candidate": r.candidate, "detail": r.detail}
            for r in results if not r.calibrated
        ],
        "results": [r.as_dict() for r in results],
    }


def write_report(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
