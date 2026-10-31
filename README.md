# Scientific Software Bug-Fix and Patch Evaluation Lab

Nine documented bug-fixing cases in a small scientific Python library, each
with an issue report, a failing test set, a correct patch, and **two or three
plausible patches that are wrong**.

The point of the lab is the last part. Showing that a gold patch passes proves
very little; what is worth measuring is whether a test suite can tell a real
fix from the patch a competent engineer would write after reading the ticket.
Measured here: **6 of the 20 incorrect patches pass at least two thirds of
their case's issue tests**, and they all fail.

```
9 cases · 29 candidate patches (20 deliberately wrong) · 113-test library suite
9/29 patches resolve their case: exactly the 9 gold patches
38/38 runs calibrated as expected
```

Every bug in this repository is a **seeded bug**: introduced deliberately by
`scripts/build_patches.py`, never present in a released version, never found in
the wild. Each case's `notes.md` says so in a banner, and the integrity tests
enforce it.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -m evaluation.cli list
```

```bash
./scripts/run_evaluation.sh
```

Read a case the way a submission would see it: the issue report only:

```bash
python -m evaluation.cli show variance-cancellation
```

Add `--notes` for the root-cause analysis and the candidate review.

## How a case is graded

Following the
[SWE-bench](https://www.swebench.com/SWE-bench/guides/evaluation/) shape, each
case declares two test sets:

* **fail-to-pass**: the case's own `tests/`, which encode the issue. They must
  fail on the broken library and pass after a correct patch.
* **pass-to-pass**: the library's 113-test suite. A fix must not regress it.

A candidate **resolves** a case only when both are fully green. One run:

```
mkdtemp → copy src/ + tests/ + the case's tests/
        → git apply bug.patch           (the broken library)
        → run fail-to-pass              MUST FAIL   ← reproduction check
        → run pass-to-pass              (recorded as blast radius)
        → git apply the candidate
        → run both sets                 resolved = both green
```

The reproduction check is not bookkeeping. A case whose issue tests pass on the
broken library is not testing its own bug, and the harness marks it
miscalibrated. Full protocol in
[docs/evaluation-protocol.md](docs/evaluation-protocol.md).

## The nine cases

| case | tier | bug class | severity | the defect |
|---|---|---|---|---|
| [`shape-mismatch-pairwise-distance`](cases/shape-mismatch-pairwise-distance) | easy | shape error | crash | Outer-difference axis inserted in the wrong position; works only when the feature dimension happens to equal `n_b` |
| [`normalization-by-peak`](cases/normalization-by-peak) | easy | incorrect normalisation | silent | Divides by peak height instead of integrated area, so line strength scales with line width |
| [`affine-unit-conversion`](cases/affine-unit-conversion) | easy | domain logic | silent | °C modelled as a factor of `274.15`: correct at exactly 1 °C and nowhere else |
| [`periodic-index-precedence`](cases/periodic-index-precedence) | medium | operator precedence | contract | `i + 1 % n` is `i + (1 % n)`, so the modulo never fires |
| [`silent-unsorted-interpolation`](cases/silent-unsorted-interpolation) | medium | missing validation | silent | Monotonicity guard deleted; `np.interp` returns garbage for unsorted input without warning |
| [`input-array-mutated`](cases/input-array-mutated) | medium | aliasing | data corruption | In-place centring writes through `np.asarray` into the caller's array, including a view of a larger dataset |
| [`floor-division-weights`](cases/floor-division-weights) | medium | integer division | silent | `weights // total` zeroes every weight, and `0.0` looked like a well-calibrated run |
| [`neumann-boundary-flux`](cases/neumann-boundary-flux) | hard | boundary condition | silent | Zero-flux wall mirrors about the cell instead of the face: edge stencil doubled, mass not conserved |
| [`variance-cancellation`](cases/variance-cancellation) | hard | floating point | silent | One-pass sum-of-squares variance returns a negative number at an offset of 1e8 |

Only one case is a crash, by design. A traceback tells you where to look; the
defects that cost real time return a plausible number.

## The near-misses

These are the candidates that make the lab worth running:

**`variance-cancellation` / `wrong_ignores_ddof`** (replaces the unstable
formula with a stable two-pass computation. Every precision test passes. It
also hardcodes division by `n`, silently turning the sample variance into the
population variance, which moves every existing caller's answer by 20 % at `n =
5`. It is the only wrong candidate in the lab that is *numerically* right, and
it regresses three pass-to-pass tests) which is exactly what that set is for.

**`input-array-mutated` / `wrong_restore_after_mutating`**: subtracts the mean
in place, then adds it back before returning. The input really is restored, and
the return value really is distinct, so the two obvious assertions pass. It
still fails on a read-only array, the restore is not bit-exact, and nothing
restores the array if an exception fires in between.

**`affine-unit-conversion` / `wrong_offset_only`**: gives Fahrenheit an offset
of `241.15`, chosen so that `32 °F` maps to exactly `273.15 K`. One anchor
matched, the slope missing, 80 K out at boiling. An affine map has two
parameters, and a single test case can only ever constrain one.

**`neumann-boundary-flux` / `wrong_half_stencil`**: the correct one-sided
difference with a spurious factor of `0.5`, found by scaling until one
hand-checked case matched. The hardest candidate in the lab to reject by
reading the diff.

Per-candidate numbers:
[results/evaluation-report.md](results/evaluation-report.md).

## Blast radius

Running the library suite against each broken baseline shows how loud each bug
is:

| seeded bug | existing tests broken |
|---|---|
| `periodic-index-precedence` | 7 |
| `shape-mismatch-pairwise-distance` | 6 |
| `affine-unit-conversion` | 6 |
| `floor-division-weights` | 3 |
| `variance-cancellation` | 3 |
| `neumann-boundary-flux` | 2 |
| `normalization-by-peak` | 2 |
| `silent-unsorted-interpolation` | 2 |
| `input-array-mutated` | **1** |

The two ends are the useful part. A one-character precedence mistake breaks
seven tests and would never ship. Silently corrupting the caller's array breaks
exactly one: and only because somebody thought to assert that a function
documented as pure leaves its argument alone.

## Layout

```
src/fieldlab/         the library under test: arrays, normalize, grids,
                      stats, interpolate, units
tests/                its own 113-test suite, the pass-to-pass baseline
cases/<case-id>/      one documented defect; see docs/evaluation-protocol.md
evaluation/           models.py, harness.py, cli.py
lab_tests/            integrity checks on the case collection itself
scripts/              build_patches.py (generates all 38 patches), run_evaluation.sh
results/              the evaluation report and raw JSON
docs/                 the evaluation protocol
```

`lab_tests/` is deliberately separate from `tests/`, because `tests/` is copied
into every evaluation sandbox as the pass-to-pass set and must not depend on
`cases/` or `evaluation/`.

```bash
python -m pytest tests/ -q        # 113 passed  (the library)
python -m pytest lab_tests/ -q    #  71 passed  (the lab)
```

## Patches are generated

`scripts/build_patches.py` holds a declarative table (per case, the edits that
break the library and the edits that produce each candidate) and emits all 38
patch files. `gold.patch` is derived as the reverse of `bug.patch`.

This is a maintenance property, not a stylistic one. Hand-written diffs go
stale the moment the library changes, and a stale `bug.patch` makes a case stop
reproducing its own bug without anything failing. CI runs `python
scripts/build_patches.py --check`.

## A note on the issue reports

Each `issue.md` is written in the voice of someone who hit the bug, and every
number in it was measured against the broken code rather than estimated. Two
reports were corrected during development because the plausible-sounding
figures were wrong: `neumann-boundary-flux` was first written as a 25 % mass
loss when the real effect is a **+1.6 % gain that saturates**, and
`silent-unsorted-interpolation` was written as interpolated values escaping the
data range when in fact the output range **collapses**: which rules out the
output-bound check as a detector and leaves input validation as the only
reliable one. Both corrections changed the diagnosis, not just the digits.

## Adding a case

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT: see [LICENSE](LICENSE).
