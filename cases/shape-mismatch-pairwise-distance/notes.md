# Root cause: shape-mismatch-pairwise-distance

> **Seeded bug.** This defect was introduced deliberately, by
> `scripts/build_patches.py`, to exercise the patch-evaluation harness. It was
> never present in a released version of `fieldlab` and was not found in the
> wild. The issue report in `issue.md` is written in the voice of a user who
> hit it, and the numbers quoted there were measured against the broken code,
> but the bug itself is synthetic.

## The defect

```python
diff = pa[:, :, None] - pb[None, :, :]      # broken
diff = pa[:, None, :] - pb[None, :, :]      # correct
```

An outer difference over points needs a new axis inserted *between* the two
point axes, so the operands are `(n_a, 1, d)` and `(1, n_b, d)` and
broadcasting produces `(n_a, n_b, d)`. The broken line inserts the axis at the
end instead, giving `(n_a, d, 1)` against `(1, n_b, d)`. NumPy then tries to
align `d` with `n_b`.

## Why it survived review

The failure is conditional on the data, not on the code path. The broadcast
succeeds whenever `d == n_b` or either is 1, which covers:

* the square self-distance case (`d == n_b` is false, but the doc example used
  `(3, 3)` against `(3, 3)`, where `d == n_b == 3`);
* every `d == 1` case with `n_b == 1`;
* any accidental `(n_a, k)` against `(k, k)`.

The docstring example and the first smoke test both landed in that set. A `(5,
3)` against `(7, 3)` call (the actual use) is the common case that fails.

## Blast radius

`pairwise_distances` is the only affected symbol. `as_points` and
`stack_fields` are untouched, and the seeded bug breaks 3 of the 11 tests in
`tests/test_arrays.py`, so the existing suite does catch it: this case is
"easy" precisely because the signal is loud.

## Candidate review

| candidate | verdict | reasoning |
|---|---|---|
| `gold` | **resolves** | Moves the inserted axis back to the point dimension. Nothing else changes. |
| `wrong_squared_distance` | rejected | Fixes the broadcast (shapes are correct, the diagonal is zero, symmetry holds) and drops the `np.sqrt`. `test_known_right_triangle` returns 25 instead of 5, and `test_scaling_is_linear_not_quadratic` reports a ratio of 4 where it must be 2. A reviewer scanning for the shape fix would approve this. |
| `wrong_elementwise_norm` | rejected | `np.linalg.norm(pa - pb, axis=1)` computes *paired* distances, which is a different function. It raises for `n_a != n_b` (the reported case) and returns an `(n,)` vector when the sizes match. Fails the shape grid and the square-case test. |

## Lesson for the test suite

`tests/test_arrays.py` already had `test_shape_is_na_by_nb`, which is why this
bug is easy. The test it was missing is `test_scaling_is_linear_not_quadratic`:
a shape assertion cannot tell a distance from a squared distance, and the
squared-distance patch is the natural thing to write when you are thinking
about `einsum` rather than about metrics.
