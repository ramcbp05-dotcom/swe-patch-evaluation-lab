# Root cause: silent-unsorted-interpolation

> **Seeded bug.** This defect was introduced deliberately, by
> `scripts/build_patches.py`, to exercise the patch-evaluation harness. It was
> never present in a released version of `fieldlab` and was not found in the
> wild. The issue report in `issue.md` is written in the voice of a user who
> hit it, and the numbers quoted there were measured against the broken code,
> but the bug itself is synthetic.

## The defect

```python
if not np.all(np.diff(xs) > 0.0):          # removed
    raise ValueError("x must be strictly increasing")
```

Three lines deleted. Nothing was added, and no other behaviour changed.

## Why deleting a check is a real bug

`numpy.interp` documents that `xp` must be increasing and does not verify it.
Given unsorted `xp` it still runs, still returns an array of the right shape,
and returns values that are not interpolations of the data. There is no warning
and no NaN to notice.

The failure signature is quiet in a specific and inconvenient way. Measured on
`sin` resampled from 60 shuffled samples, the error reaches ~2.0 (twice the
full amplitude) while the output *range collapses* (for example `(0.086,
0.497)` where the data spans `(-0.989, 1.000)`). It does not blow up, produce
NaN, or escape the data range, because `np.interp` still returns a convex
combination of *some* pair of samples; just not the right pair. In the fully
reversed case every in-range query returns `y[0]`.

That rules out the detector one would reach for first. A bound check on the
output (`min(out) >= min(y)`) passes on unsorted input, so it cannot stand in
for the monotonicity check: the only reliable test is on the *input*. The bound
check is still worth keeping for sorted data, as
`test_result_stays_within_the_data_range`, but it is not what catches this bug.

The reason this reaches production rather than being caught in review is that
the deleted check protected against *data*, not against a code path. Every
existing call site that passes sorted data keeps working, so the only test that
fails is the one that was written specifically for the check.

## Why `raise` rather than sort

The issue is explicit about this, and it is the interesting design point:
out-of-order rows mean the upstream cleanup stage did not run or did not work.
Sorting inside `resample` converts a detectable pipeline failure into a quiet
one, and the next person to hit it has no way to find out. A library function
that silently repairs its input also removes the caller's ability to decide:
some callers want to abort, some want to re-fetch, some genuinely do want to
sort, and they can call `np.argsort` themselves in one line.

## Blast radius

The seeded bug breaks 2 of the 12 tests in `tests/test_interpolate.py`
(`test_unsorted_x_raises` and `test_duplicate_x_raises`). Nothing else in the
library calls `resample`.

## Candidate review

| candidate | verdict | reasoning |
|---|---|---|
| `gold` | **resolves** | Restores the check ahead of the `extrapolate` validation, so the error order is unchanged. |
| `wrong_sorts_silently` | rejected | `np.argsort` on `x`, reordering `y` to match. This is the patch most likely to be written, because it makes the reported symptom disappear and the output becomes correct interpolation of the *sorted* data. It papers over the upstream failure the issue asks to be told about, and it cannot help with duplicate abscissae: after sorting, the duplicates are adjacent and `np.interp` still picks arbitrarily. Fails all five tests in `TestNonMonotonicIsRejected` and all three duplicate tests. |
| `wrong_allows_duplicates` | rejected | `np.diff(xs) >= 0.0` instead of `> 0.0`. Catches every reversal, which is the whole of the reported issue, so all five `TestNonMonotonicIsRejected` tests pass. Repeated `x` values slip through, and there the interpolated value depends on which of the duplicates `np.interp` lands on: an input-dependent answer with no error. Caught only by `TestDuplicatesAreRejected`. |

## Lesson for the test suite

The two wrong patches split the test suite cleanly down the middle, which is
why both halves have to exist: `wrong_sorts_silently` passes nothing in
`TestNonMonotonicIsRejected`, and `wrong_allows_duplicates` passes all of it. A
suite that only tested reversed input would merge the second one.

One negative result is worth recording: the output-range bound (`out.min() >=
y.min()`) does *not* detect this bug, because `np.interp` on unsorted input
still interpolates between two real samples and the result stays inside the
data range. Checking the output cannot substitute for validating the input
here: which is the general argument for keeping the guard in the library rather
than asking every caller to sanity-check what comes back.
