"""Case specification: the on-disk contract for a bug-fixing case.

Layout of ``cases/<case-id>/``::

    case.yaml             metadata parsed into :class:`CaseSpec`
    issue.md              the bug report, as a user would file it
    notes.md              root-cause analysis and the review of each candidate
    bug.patch             the SEEDED BUG: clean library -> broken library
    tests/test_issue.py   the fail-to-pass test(s)
    patches/gold.patch    the correct fix
    patches/wrong_*.patch plausible candidates that must not resolve the case
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

DIFFICULTIES = ("easy", "medium", "hard")
SEVERITIES = ("crash", "silent-wrong-result", "data-corruption", "contract-violation")
KINDS = ("correct", "incorrect")


class CaseSpecError(ValueError):
    """Raised when a ``case.yaml`` is malformed or the case layout is incomplete."""


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One patch submitted against a case."""
    name: str
    kind: str          # correct | incorrect
    summary: str
    breaks: str = ""   # for an incorrect candidate: what it still gets wrong

    @property
    def expected_resolved(self) -> bool:
        return self.kind == "correct"


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    module: str
    bug_class: str
    difficulty: str
    severity: str
    summary: str
    candidates: tuple[Candidate, ...]
    root: Path

    @property
    def issue_path(self) -> Path:
        return self.root / "issue.md"

    @property
    def notes_path(self) -> Path:
        return self.root / "notes.md"

    @property
    def bug_patch(self) -> Path:
        return self.root / "bug.patch"

    @property
    def issue_tests(self) -> Path:
        return self.root / "tests"

    def candidate_patch(self, name: str) -> Path:
        return self.root / "patches" / f"{name}.patch"

    @classmethod
    def load(cls, root: Path) -> "CaseSpec":
        root = Path(root)
        meta_path = root / "case.yaml"
        if not meta_path.is_file():
            raise CaseSpecError(f"{root}: missing case.yaml")
        data: dict[str, Any] = yaml.safe_load(meta_path.read_text()) or {}

        required = ("id", "title", "module", "bug_class", "difficulty", "severity",
                    "summary", "candidates")
        missing = [k for k in required if k not in data]
        if missing:
            raise CaseSpecError(f"{meta_path}: missing keys {missing}")
        if data["id"] != root.name:
            raise CaseSpecError(f"{meta_path}: id {data['id']!r} != directory {root.name!r}")
        if data["difficulty"] not in DIFFICULTIES:
            raise CaseSpecError(f"{meta_path}: difficulty must be one of {DIFFICULTIES}")
        if data["severity"] not in SEVERITIES:
            raise CaseSpecError(f"{meta_path}: severity must be one of {SEVERITIES}")

        candidates = []
        for entry in data["candidates"]:
            if entry.get("kind") not in KINDS:
                raise CaseSpecError(
                    f"{meta_path}: candidate {entry.get('name')!r} has kind "
                    f"{entry.get('kind')!r}, expected one of {KINDS}"
                )
            candidates.append(Candidate(
                name=entry["name"], kind=entry["kind"],
                summary=entry.get("summary", ""), breaks=entry.get("breaks", ""),
            ))

        if not any(c.kind == "correct" for c in candidates):
            raise CaseSpecError(f"{meta_path}: no candidate of kind 'correct'")
        if not any(c.kind == "incorrect" for c in candidates):
            raise CaseSpecError(
                f"{meta_path}: no candidate of kind 'incorrect'. A case with only a "
                "gold patch cannot show that the tests discriminate."
            )

        spec = cls(
            case_id=data["id"], title=data["title"], module=data["module"],
            bug_class=data["bug_class"], difficulty=data["difficulty"],
            severity=data["severity"], summary=data["summary"],
            candidates=tuple(candidates), root=root,
        )
        spec.validate_layout()
        return spec

    def validate_layout(self) -> None:
        for path in (self.issue_path, self.notes_path, self.bug_patch, self.issue_tests):
            if not path.exists():
                raise CaseSpecError(
                    f"{self.case_id}: missing {path.relative_to(self.root)}"
                )
        if not any(self.issue_tests.glob("test_*.py")):
            raise CaseSpecError(f"{self.case_id}: tests/ contains no test_*.py")
        for candidate in self.candidates:
            patch = self.candidate_patch(candidate.name)
            if not patch.is_file():
                raise CaseSpecError(
                    f"{self.case_id}: candidate {candidate.name!r} has no "
                    f"patches/{candidate.name}.patch"
                )


def discover(cases_dir: Path) -> list[CaseSpec]:
    """Load every case under ``cases_dir``, sorted by id."""
    cases_dir = Path(cases_dir)
    return [
        CaseSpec.load(child)
        for child in sorted(cases_dir.iterdir())
        if child.is_dir() and (child / "case.yaml").is_file()
    ]
