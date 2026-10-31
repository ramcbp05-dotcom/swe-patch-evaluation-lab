# Root cause: normalization-by-peak

> **Seeded bug.** This defect was introduced deliberately, by
> `scripts/build_patches.py`, to exercise the patch-evaluation harness. It was
> never present in a released version of `fieldlab` and was not found in the
> wild. The issue report in `issue.md` is written in the voice of a user who
> hit it, and the numbers quoted there were measured against the broken code,
> but the bug itself is synthetic.

## The defect

```python
area = float(np.max(np.abs(ys)))        # broken: peak height
area = float(np.trapezoid(ys, xs))      # correct: integrated area
```

The variable is still called `area` and the error message still says
"integrated area", which is why the diff reads as an optimisation rather than a
change of meaning.

## Why it is worse than a wrong number

Peak normalisation and area normalisation are both legitimate operations: the
module exports both. The bug is not that the arithmetic is wrong; it is that
the function no longer provides the invariant its callers depend on. The output
is still a plausible normalised profile in `[0, 1]`, so nothing downstream
complains. The error only shows up as a *correlation between the reported
quantity and the instrument resolution*, which is the hardest kind of artefact
to attribute: it looks like physics.

For a Gaussian of width `σ`, peak normalisation leaves an integral of `σ√(2π)`,
so the reported line strength scales linearly with the line width. That is
exactly the 40 % shift in the issue.

## Blast radius

The seeded bug breaks 4 of the 7 `TestUnitArea` tests in
`tests/test_normalize.py`: including `test_width_independence`, which was
written for this invariant. `unit_peak` and `zscore` are untouched.

## Candidate review

| candidate | verdict | reasoning |
|---|---|---|
| `gold` | **resolves** | Restores the trapezoidal integral over the given `x`. |
| `wrong_uniform_spacing` | rejected | `np.trapezoid(ys)` integrates against sample *index*, i.e. it assumes `dx == 1`. Passes every test on a unit-spacing grid and is off by a factor of `dx` everywhere else. `test_grid_spacing_does_not_matter` fails at `dx = 0.001` by a factor of 1000, which is what makes the parametrised spacings worth having. |
| `wrong_rectangle_rule` | rejected | `sum(y) * (x[1] - x[0])` is the rectangle rule on an assumed-uniform grid. Two separate errors: it reads the spacing from the first interval only, so a non-uniform grid is silently wrong, and even on a uniform grid it gives the end samples full weight instead of half. `test_endpoint_weighting_is_trapezoidal` and `test_non_uniform_grid` both fail. |

## Lesson for the test suite

The existing `test_integral_is_one` used a single fixture, so a peak-normalised
result that happened to be close would have passed. The tests that actually pin
the contract down are the ones that *vary* something the answer must be
invariant to: the width, the grid spacing, and the scale of `y`. Those three
parametrisations catch all three wrong patches between them.
