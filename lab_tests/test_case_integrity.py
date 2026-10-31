"""Integrity checks on the case collection itself.

These are deliberately kept out of ``tests/``, because ``tests/`` is the
pass-to-pass set the harness copies into every evaluation sandbox. A lab test
that needs ``cases/`` on disk would fail there.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from evaluation import models

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = REPO_ROOT / "cases"

SPECS = models.discover(CASES_DIR)
IDS = [s.case_id for s in SPECS]


def test_collection_size_matches_the_readme():
    assert 8 <= len(SPECS) <= 10, f"the README advertises 8-10 cases; found {len(SPECS)}"


def test_case_ids_are_kebab_case_and_unique():
    assert len(set(IDS)) == len(IDS), "duplicate case ids"
    for case_id in IDS:
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", case_id), f"{case_id!r} is not kebab-case"


def test_bug_classes_are_all_different():
    """A lab with three normalisation bugs measures one skill three times."""
    classes = [s.bug_class for s in SPECS]
    duplicates = {c for c in classes if classes.count(c) > 1}
    assert not duplicates, f"repeated bug classes: {sorted(duplicates)}"


def test_every_difficulty_tier_is_represented():
    counts = {tier: sum(s.difficulty == tier for s in SPECS) for tier in models.DIFFICULTIES}
    for tier, count in counts.items():
        assert count >= 2, f"only {count} {tier} case(s); distribution: {counts}"


def test_modules_under_test_are_spread_across_the_library():
    modules = {s.module for s in SPECS}
    assert len(modules) >= 4, (
        f"all cases target {sorted(modules)}; a lab concentrated in one module "
        "does not exercise much"
    )


def test_all_severities_are_declared_and_mostly_silent():
    """A crash is the easy case; most real defects return a plausible number."""
    silent = sum(s.severity != "crash" for s in SPECS)
    assert silent >= len(SPECS) - 2, (
        f"only {silent}/{len(SPECS)} cases are non-crash. A patch-evaluation lab "
        "made of tracebacks does not test much judgement."
    )


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_case_is_labelled_as_seeded(spec):
    data = yaml.safe_load((spec.root / "case.yaml").read_text())
    assert data.get("seeded") is True, (
        f"{spec.case_id}: case.yaml must carry `seeded: true`. Every bug in this "
        "lab is synthetic and has to say so."
    )
    assert "Seeded bug." in spec.notes_path.read_text(), (
        f"{spec.case_id}: notes.md is missing the seeded-bug banner"
    )


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_candidate_mix(spec):
    correct = [c for c in spec.candidates if c.kind == "correct"]
    incorrect = [c for c in spec.candidates if c.kind == "incorrect"]
    assert len(correct) == 1, f"{spec.case_id}: expected exactly one gold patch"
    assert correct[0].name == "gold", f"{spec.case_id}: the correct patch must be 'gold'"
    assert len(incorrect) >= 2, (
        f"{spec.case_id}: only {len(incorrect)} incorrect candidate(s). Two is the "
        "minimum that shows the tests discriminate rather than just accept gold."
    )
    for candidate in incorrect:
        assert candidate.name.startswith("wrong_"), (
            f"{spec.case_id}: incorrect candidate {candidate.name!r} should be "
            "named wrong_*"
        )
        assert candidate.breaks.strip(), (
            f"{spec.case_id}/{candidate.name}: case.yaml does not say what it "
            "still gets wrong"
        )


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_issue_report_reads_like_a_bug_report(spec):
    text = spec.issue_path.read_text()
    assert text.startswith("# "), f"{spec.case_id}: issue.md must open with an H1"
    assert len(text.split()) >= 120, f"{spec.case_id}: issue.md is too thin to act on"
    assert "```" in text, f"{spec.case_id}: issue.md has no reproduction snippet"
    assert "Expected behaviour" in text, (
        f"{spec.case_id}: issue.md never states the expected behaviour"
    )


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_issue_report_does_not_give_away_the_patch(spec):
    """The issue is what a submission sees; notes.md is the spoiler."""
    text = spec.issue_path.read_text()
    for leak in ("gold.patch", "bug.patch", "patches/", "notes.md", "build_patches"):
        assert leak not in text, f"{spec.case_id}: issue.md mentions {leak!r}"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_notes_review_every_candidate(spec):
    notes = spec.notes_path.read_text()
    assert "## The defect" in notes, f"{spec.case_id}: notes.md has no defect section"
    assert "## Candidate review" in notes, f"{spec.case_id}: notes.md has no review table"
    for candidate in spec.candidates:
        assert f"`{candidate.name}`" in notes, (
            f"{spec.case_id}: notes.md never discusses candidate {candidate.name!r}"
        )
    assert "Lesson for the test suite" in notes, (
        f"{spec.case_id}: notes.md draws no conclusion for the test suite"
    )


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_issue_tests_are_substantial_and_explain_themselves(spec):
    files = sorted(spec.issue_tests.glob("test_*.py"))
    assert files, f"{spec.case_id}: no issue tests"
    for path in files:
        source = path.read_text()
        n_tests = len(re.findall(r"^\s*def test_", source, re.MULTILINE))
        assert n_tests >= 8, (
            f"{path.name}: only {n_tests} test functions. A case needs enough "
            "assertions that a partial patch fails some and passes others."
        )
        bare = [
            (i + 1, line.strip())
            for i, line in enumerate(source.splitlines())
            if re.match(r"^\s*assert .+", line)
            and "," not in line
            and not line.rstrip().endswith("(")
        ]
        assert not bare, (
            f"{path.relative_to(spec.root)}: asserts without a message at lines "
            f"{[n for n, _ in bare]}"
        )


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_patches_are_unified_diffs_against_the_declared_module(spec):
    patches = [spec.bug_patch] + [spec.candidate_patch(c.name) for c in spec.candidates]
    for patch in patches:
        text = patch.read_text()
        assert text.startswith("--- a/src/"), (
            f"{patch.name}: does not start with a unified-diff header"
        )
        assert f"src/{spec.module}" in text, (
            f"{patch.name}: does not touch the declared module src/{spec.module}"
        )
        assert "@@" in text, f"{patch.name}: no hunk header"


def test_committed_patches_match_the_build_table():
    """scripts/build_patches.py is the source of truth; the patches are derived."""
    proc = subprocess.run(
        [sys.executable, "scripts/build_patches.py", "--check"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        "the committed patches are out of date with scripts/build_patches.py.\n"
        f"{proc.stdout}\n{proc.stderr}\nRun: python scripts/build_patches.py"
    )


def test_library_suite_is_not_polluted_by_lab_tests():
    """tests/ is the pass-to-pass set and must not depend on cases/."""
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        source = path.read_text()
        for forbidden in ("evaluation", "cases/", "build_patches"):
            assert forbidden not in source, (
                f"{path.relative_to(REPO_ROOT)} references {forbidden!r}; the "
                "pass-to-pass suite runs in a sandbox that has neither"
            )
