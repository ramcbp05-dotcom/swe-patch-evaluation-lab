# Root cause: floor-division-weights

> **Seeded bug.** This defect was introduced deliberately, by
> `scripts/build_patches.py`, to exercise the patch-evaluation harness. It was
> never present in a released version of `fieldlab` and was not found in the
> wild. The issue report in `issue.md` is written in the voice of a user who
> hit it, and the numbers quoted there were measured against the broken code,
> but the bug itself is synthetic.

## The defect

```python
# broken
total = weights.sum()
normalised = weights // total           # floor division
return float(np.dot(normalised, values))

# correct
total = float(weights.sum())
return float(np.dot(weights, values) / total)
```

The restructuring is the kind that reads well in a diff ("normalise the weights
first so the dot product is already an average" is a reasonable thing to write)
and `//` instead of `/` is one character.

## Why the answer is exactly zero

Every weight is, by construction, no larger than the sum of all the weights. `w
// total` is therefore `0` for every weight except in the degenerate case of a
single non-zero weight, where it is `1`. So the normalised weight vector is all
zeros, the dot product is zero, and the function returns `0.0`.

Note that this has nothing to do with the dtype. `weights` is still
`np.asarray(w, dtype=float)`, so these are float64 floor divisions: `1.0 //
3.0` is `0.0`. Converting the weights to float, which is the first thing one
would check, changes nothing. The bug is the operator.

## Why zero was believable

This is the part worth keeping: `0.0` is not an obviously wrong answer. The
caller was averaging residuals, whose true weighted mean is near zero, so a
reported `0.0` read as a well-calibrated run. A bug that returns a *plausible*
constant is harder to find than one that returns `NaN`, and much harder than
one that raises.

The single-weight case returning the right answer made it worse: the one input
anybody checks by hand works.

## Blast radius

The seeded bug breaks 6 of the 8 tests in `TestWeightedMean` in
`tests/test_stats.py`. `variance`, `stddev` and `merge_moments` are untouched.
This case is "medium" rather than "easy" only because the two candidate patches
are hard to reject by eye, not because the bug is subtle to find.

## Candidate review

| candidate | verdict | reasoning |
|---|---|---|
| `gold` | **resolves** | Dot product first, one true division at the end. Also the numerically preferable order: a single rounding rather than one per weight. |
| `wrong_floor_the_result` | rejected | `np.dot(weights, values) // total`. The zeros are gone and the answers now look right: `weighted_mean([1.0, 11.0], [9.0, 1.0])` gives exactly `2.0`, which is the worked example in the issue. The floor division simply moved. `weighted_mean([10, 20], [1, 2])` returns `16.0` instead of `16.666...`, so the whole of `TestNoTruncation` fails, and `test_answer_is_not_an_integer` names the symptom directly. Picking a test case with an integer answer is how a patch like this gets approved. |
| `wrong_rounded_weights` | rejected | `np.round(weights / total, 2)` (true division, so the structural bug is fixed, and then the weights are rounded "for readability". The error is in the third decimal: `16.7` instead of `16.666...`. Two independent properties catch it. First, accuracy: every parametrised case in `TestNoTruncation` and `test_matches_numpy_average` fail at `rel=1e-12`. Second, and more interesting, it breaks an *invariance*) rescaling all the weights by `1e-6` changes which digits survive rounding, so `test_rescaling_the_weights_changes_nothing` fails even where the absolute error is small. |

## Lesson for the test suite

The original suite had `test_integer_weights_do_not_truncate`, which catches
the seeded bug and `wrong_floor_the_result` but not `wrong_rounded_weights`.

Two additions close the case. One is trivial and worth stating anyway: at least
one test case must have a **non-integer expected answer**, otherwise a
truncating implementation passes. The other is the invariance test: the
weighted mean is homogeneous of degree zero in the weights, so scaling them all
must change nothing. Invariance tests do not need a known answer, which makes
them cheap to write and hard to satisfy accidentally.
