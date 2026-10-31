# Evaluation protocol

## What a case is

A case is one documented defect in `fieldlab`, plus everything needed to decide
whether a patch fixes it:

```
cases/<case-id>/
├── case.yaml             metadata, and the declared verdict for each candidate
├── issue.md              the bug report: all a submission is allowed to see
├── notes.md              root-cause analysis and the candidate review (spoilers)
├── bug.patch             the seeded bug: clean library -> broken library
├── tests/test_issue.py   the fail-to-pass set
└── patches/
    ├── gold.patch        the correct fix
    └── wrong_*.patch     plausible candidates that must not resolve the case
```

Every bug in this lab is **seeded**: introduced deliberately by
`scripts/build_patches.py`, never present in a released version, never found in
the wild. Each `notes.md` says so in a banner at the top, and
`lab_tests/test_case_integrity.py` enforces both that banner and the `seeded:
true` key in `case.yaml`.

## The two test sets

Following the SWE-bench shape:

**fail-to-pass**: the case's `tests/`. These encode the issue. They must fail
on the broken library and pass after a correct patch.

**pass-to-pass**: the project's own `tests/`, all 113 of them. A fix must not
regress them. This is the set that catches a patch which satisfies the issue by
changing behaviour other callers depend on.

A candidate **resolves** a case only when both sets are fully green.

## What a run does

```
1. mkdtemp; copy src/, tests/, and the case's tests/ into it
2. git apply bug.patch                     -> the broken library
3. run the fail-to-pass set                -> must FAIL  (reproduction check)
4. run the pass-to-pass set                -> recorded as blast radius
5. git apply patches/<candidate>.patch
6. run the fail-to-pass set                -> pass?
7. run the pass-to-pass set                -> pass?
8. resolved = (6) and (7)
```

Step 3 is not optional bookkeeping. A case whose issue tests pass on the broken
library is not testing its own bug, and the harness marks the whole case
miscalibrated. Step 4 is what the report turns into the blast-radius table.

Each run gets a fresh temporary tree and nothing is cached, so results
reproduce exactly.

## Why every case ships wrong patches

A case with only a gold patch demonstrates that the gold patch works. It says
nothing about whether the tests would catch anything else: and "anything else"
is what a patch-evaluation harness is for.

So every case declares at least two `wrong_*` candidates, and the harness
*requires* them to stay unresolved. They are not straw men. Each one is the
patch a competent engineer would plausibly write after reading the issue:

* the patch that fixes the **reported symptom** and leaves the defect
  (`wrong_clamp_negative` silences the negative-variance traceback;
  `wrong_copy_too_late` stops the return value aliasing and keeps the input
  corruption);
* the patch that is **right in form and wrong in a constant**
  (`wrong_half_stencil` has the correct one-sided difference with a spurious
  `0.5`);
* the patch **verified against a single data point** (`wrong_offset_only` picks
  a Fahrenheit offset of `241.15` so that `32 °F` lands exactly on `273.15 K`,
  and is 80 K out at boiling);
* the **partial** fix (`wrong_celsius_only` repairs Celsius and leaves
  Fahrenheit);
* the patch that **fixes the issue and breaks the contract**
  (`wrong_ignores_ddof` is numerically correct and quietly turns the sample
  variance into the population variance).

Measured: 6 of the 20 incorrect patches pass at least two thirds of their
case's issue tests (five above it, one exactly at it). See [the evaluation
report](../results/evaluation-report.md).

## Patches are generated, not hand-written

`scripts/build_patches.py` holds a declarative table: per case, the edits that
turn the clean library into the broken one, and the edits that turn the broken
one into each candidate. It emits all 38 patch files, and `gold.patch` is
derived as the reverse of `bug.patch`.

This matters for maintenance. A change to `fieldlab` would otherwise leave
hand-written diffs silently stale, and a stale `bug.patch` makes a case stop
reproducing. `python scripts/build_patches.py --check` fails if any committed
patch disagrees with the table, and CI runs it.

## Difficulty tiers

**easy**: the defect is visible from the traceback or from one obviously wrong
number, and the existing suite already catches it (6+ pass-to-pass failures).
`shape-mismatch-pairwise-distance`, `normalization-by-peak`,
`affine-unit-conversion`.

**medium**: the defect is data-dependent or the wrong patches are hard to
reject by eye. `periodic-index-precedence`, `silent-unsorted-interpolation`,
`input-array-mutated`, `floor-division-weights`.

**hard**: rejecting the plausible patches needs domain knowledge: discrete
conservation laws, or how floating-point error scales with the condition
number. `neumann-boundary-flux`, `variance-cancellation`.

## Severity

`crash`, `silent-wrong-result`, `data-corruption`, `contract-violation`. Only
one case is a crash, by design: a traceback tells you where to look, and the
defects that cost real time are the ones that return a plausible number.
`lab_tests` enforces that at most two cases are crashes.

## Adding a case

See [CONTRIBUTING.md](../CONTRIBUTING.md). The short version: add the edits to
`scripts/build_patches.py`, write `issue.md` without giving the fix away, write
`tests/test_issue.py` with at least eight assertions, write at least two
`wrong_*` variants, and make `python -m evaluation.cli evaluate` come back
fully calibrated.
