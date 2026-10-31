# Adding a case

A pull request that adds a case must leave `./scripts/run_evaluation.sh` fully
calibrated: the case reproduces its own bug, `gold` resolves it, and every
`wrong_*` candidate does not.

## 1. Declare the bug in the build table

Patches are generated, not hand-written. Add an entry to `CASES` in
[`scripts/build_patches.py`](scripts/build_patches.py):

```python
"my-case": {
    "module": "src/fieldlab/stats.py",
    "bug": [(clean_source_fragment, broken_source_fragment)],
    "variants": {
        "wrong_partial": [(broken_fragment, partially_fixed_fragment)],
    },
},
```

Each anchor must appear **exactly once** in the file; the script fails loudly
if it does not. Then:

```bash
python scripts/build_patches.py
```

`gold.patch` is derived automatically as the reverse of `bug.patch`.

## 2. Label it as seeded

Every bug here is synthetic. `case.yaml` carries `seeded: true`, and `notes.md`
opens with the seeded-bug banner. `lab_tests` enforces both. Write `issue.md`
in the voice of a user who hit the bug (that is the exercise) but
never imply the defect was found in a real release.

## 3. Write the issue report without giving the fix away

`issue.md` is all a submission sees. It needs a reproduction snippet, the
observed output, the expected output, and an "Expected behaviour" section. It
must not mention `patches/`, `bug.patch`, `notes.md` or `build_patches`:
`lab_tests` checks.

**Measure every number you quote.** Run the broken code and paste what it
actually prints. Two issue reports in this repo were written with plausible
figures first and corrected afterwards: one claimed a 25 % mass loss where the
real effect is a +1.6 % gain, and one claimed interpolated values escaping the
data range where in fact the range *collapses*. Both corrections changed the
diagnosis, not just the digits.

## 4. Write the fail-to-pass tests

At least eight test functions, every assertion carrying a message that names
the observed value, the expected value, and why they differ. `lab_tests`
rejects bare asserts.

Aim for tests that pin down an **invariant** rather than a table of answers.
The ones that did the most work in this repo:

* building an operator's matrix and asserting zero row sums and symmetry
  (`neumann-boundary-flux`);
* asserting a map is a bijection rather than that its values are in range
  (`periodic-index-precedence`);
* varying something the answer must not depend on: grid spacing, weight
  scaling, data offset: rather than checking one case harder;
* setting `writeable = False` on an input to turn "did it mutate my array" into
  "can it mutate an array at all" (`input-array-mutated`).

## 5. Ship at least two wrong patches

```bash
python -m evaluation.cli evaluate -c my-case
```

Each `wrong_*` must stay unresolved. If you cannot think of a plausible wrong
patch, the case is probably too simple to measure anything.

Good sources of one: fix the symptom in the traceback and nothing else; get the
form right and a constant wrong; verify against a single data point; fix one of
two affected code paths; fix the issue and change a contract on the way past.

## 6. Write the root-cause notes

`notes.md` needs a `## The defect` section, a `## Candidate review` table with
a row per candidate, and a `## Lesson for the test suite` section saying which
test would have caught this and which test was missing. `lab_tests` checks all
three.

## 7. Check the whole lab

```bash
./scripts/run_evaluation.sh
```

Library suite green, lab integrity green, 100 % calibrated.
