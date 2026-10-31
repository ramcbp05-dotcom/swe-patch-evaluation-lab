# Root cause: variance-cancellation

> **Seeded bug.** This defect was introduced deliberately, by
> `scripts/build_patches.py`, to exercise the patch-evaluation harness. It was
> never present in a released version of `fieldlab` and was not found in the
> wild. The issue report in `issue.md` is written in the voice of a user who
> hit it, and the numbers quoted there were measured against the broken code,
> but the bug itself is synthetic.

## The defect

```python
# broken: one pass, two accumulators
s1 = float(arr.sum())
s2 = float((arr * arr).sum())
return (s2 - s1 * s1 / n) / (n - ddof)

# correct: Welford's online update
for count, value in enumerate(arr, start=1):
    delta = value - mean
    mean += delta / count
    m2 += delta * (value - mean)
return max(m2, 0.0) / (n - ddof)
```

The two are algebraically identical. They are not numerically equivalent, and
the commit message called the change "simplify variance to a single pass".

## Why it loses every digit

For readings near `1e8`, `s2 ≈ n·1e16` and `s1²/n ≈ n·1e16`. The quantity we
want is `n·2.5e-4`. Double precision carries about 16 significant decimal
digits, so the rounding error on each operand is already around `n·1`: four
orders of magnitude larger than the answer. The subtraction returns noise, and
the sign of noise is arbitrary, which is why a *negative variance* appears.

This is catastrophic cancellation, and the important property is that **no
rearrangement of that expression fixes it**. The information is destroyed when
`x²` is formed at the wrong scale, before any subtraction happens.

Welford's recurrence never forms a raw sum of squares. It accumulates squared
deviations from the *running* mean, so every intermediate stays on the scale of
the spread rather than the scale of the data. Its relative error grows like
`n·eps·κ` with `κ = mean/σ`, instead of `κ²`.

Measured relative error (5000 samples):

| κ | Welford | two-pass + fsum | one-pass sum-of-squares |
|------|---------|-----------------|-------------------------|
| 1e9  | 5e-09   | 1e-14           | 1.3e+02 |
| 1e10 | 2e-07   | 4e-13           | 1.0e+00 |
| 1e12 | 1.4e-07 | 1.5e-08         | 2.1e+08 |

The test tolerance is `1e-5`, which sits two orders of magnitude clear of the
worst correct algorithm and seven clear of the broken one. Asking for `1e-10`
at `κ = 1e12` would fail the gold patch: double precision cannot deliver it.

## Blast radius

The seeded bug breaks 4 of the 8 tests in `TestVariance` and, through `stddev`,
one more. `weighted_mean` and `merge_moments` are untouched. `merge_moments` is
worth noting as a contrast: it already uses the numerically careful increment
form, so the same class of bug was avoided there.

## Candidate review

| candidate | verdict | reasoning |
|---|---|---|
| `gold` | **resolves** | Restores Welford. Shift invariant to `1e-7` at `κ = 1e12`, exactly zero on constant input, `ddof` unchanged. |
| `wrong_clamp_negative` | rejected | `max(..., 0.0)` is the patch the traceback asks for: `stddev` stops raising and nothing in the log looks wrong any more. The variance is now either exactly `0` (clamped) or wrong by `1e5` (when the noise lands positive). The second case is the argument against clamping: the negative sign was a *symptom*, not the defect. Fails every shift-invariance test. |
| `wrong_ignores_ddof` | rejected | Two-pass about the sample mean: genuinely stable, and all the precision tests pass. It also hardcodes division by `n`, silently turning the sample variance into the population variance. At `n = 5` that is a 20 % change in every existing caller's answer. `test_default_is_the_sample_variance` and `test_ddof_matches_numpy` catch it. This is the candidate most likely to be merged: the issue was about precision, and precision is fixed. |
| `wrong_float32_accumulate` | rejected | Right algorithm, wrong dtype. `float32` carries about 7 decimal digits, so at an offset of `1e8` the deviations of order `1e-2` are below the representable resolution and quantise to a handful of levels. Fails the extreme-κ test and `test_full_double_precision_on_easy_data`. |

## Lesson for the test suite

The original `test_offset_invariance` used offsets up to `1e9` and a `1e-6`
tolerance, which is what caught the bug. What it could not catch was
`wrong_ignores_ddof`: a precision suite that only varies the offset never
checks *which estimator* is being computed.

The pair that pins this function down is a precision axis (offsets `0` to
`1e15`) crossed with an estimator axis (`ddof` of `0`, `1`, `2` against NumPy).
Either alone admits one of the two wrong patches.
