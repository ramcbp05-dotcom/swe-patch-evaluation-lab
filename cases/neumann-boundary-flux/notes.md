# Root cause: neumann-boundary-flux

> **Seeded bug.** This defect was introduced deliberately, by
> `scripts/build_patches.py`, to exercise the patch-evaluation harness. It was
> never present in a released version of `fieldlab` and was not found in the
> wild. The issue report in `issue.md` is written in the voice of a user who
> hit it, and the numbers quoted there were measured against the broken code,
> but the bug itself is synthetic.

## The defect

```python
# broken: mirror about the boundary CELL, reuse the interior stencil
out[0]  = (arr[1] - 2.0 * arr[0] + arr[1]) * inv

# correct: ghost cell equals the boundary cell
out[0]  = (arr[1] - arr[0]) * inv
```

Both lines are "reflect at the wall". They differ in *what is reflected*.

A zero-flux (homogeneous Neumann) wall sits on the **face** between the
boundary cell and the ghost cell outside it. Zero flux means `u_ghost = u_0`,
and substituting that into the three-point stencil `(u_ghost - 2u_0 + u_1)`
leaves `(u_1 - u_0)`: a one-sided difference with coefficient 1, not 2.

The broken version reflects about the boundary **cell centre**, setting
`u_ghost = u_1`. That gives `2(u_1 - u_0)`: the right shape, twice the
magnitude. The field gradient at the wall is still zero in the limit, so the
operator looks consistent: it is the discrete conservation property that
breaks.

## Why conservation is the right test

The continuous statement is that `∫∇²u dx` over a closed domain equals the
boundary flux, which is zero. Discretely that means the operator matrix has
zero row sums, equivalently `L.sum(axis=1) == 0`, equivalently
`laplacian_1d(u).sum() == 0` for every `u`.

The broken operator has non-zero row sums in its two boundary rows, so each
step of a diffusion solver moves a little material through a wall that is
supposed to be sealed. Over 4000 steps that integrates to the +1.6 % in the
issue, and then stops: once the field flattens the edge gradients vanish and
the drift saturates. Nothing diverges, nothing raises, and the shape of the
solution stays physically plausible, which is why the symptom is "our tracer
budget does not close by a bit" rather than "the code crashed".

The error is also **antisymmetric between the two walls**: it is proportional
to the local gradient, so a field that is antisymmetric about the mid-point
produces equal and opposite errors and conserves mass exactly. A budget check
run on a linear ramp passes. That is the single most important reason this bug
needs a *structural* test rather than a worked example.

A second, independent check is that the operator must be **symmetric**: the
continuous Laplacian with Neumann conditions is self-adjoint. The broken
version is not, because only its boundary rows are scaled.

## Blast radius

The seeded bug breaks 3 of the 10 tests in `tests/test_grids.py`
(`test_neumann_conserves_total_mass`,
`test_neumann_boundary_stencil_is_one_sided`, and the `test_dx_scaling` /
quadratic checks are unaffected since they only look at the interior).
`gradient_1d`, `neighbour_indices` and the periodic and Dirichlet branches are
untouched.

## Candidate review

| candidate | verdict | reasoning |
|---|---|---|
| `gold` | **resolves** | One-sided difference at each wall; row sums become zero and the matrix is symmetric again. |
| `wrong_zero_at_wall` | rejected | Reads as the obvious meaning of "no flux": nothing happens at the wall. But `∇²u = 0` *in* the boundary cell is a different statement from zero flux *through* the wall, and the interior rows do not sum to zero without their boundary partners. Fails every conservation test and `test_boundary_cell_is_a_one_sided_difference`. |
| `wrong_copy_interior` | rejected | `out[0] = out[1]` extrapolates the Laplacian rather than the field. It is smooth and looks reasonable on a plot, which is how it would survive visual inspection. It has no conservation property and gets the exact edge values wrong; `test_exact_values_from_the_issue` and `test_operator_matrix_rows_sum_to_zero` both fail. |
| `wrong_half_stencil` | rejected | The *form* is now right: this is the hardest candidate to reject by eye, because the diff looks like the gold patch with one extra constant. The `0.5` was found by scaling until one hand-checked case matched. Conservation still fails, and so do the exact-value tests. |

## Lesson for the test suite

The original suite checked the interior against an analytic second derivative
and checked conservation on one random field. Both are necessary and neither is
sufficient: the interior test cannot see a boundary bug at all, and a single
random field can pass by luck for a scaled stencil.

What pins the operator down completely is building the matrix explicitly
(`laplacian_1d` applied to each basis vector) and asserting two structural
properties of it: zero row sums and symmetry. Those two together reject all
three wrong patches without reference to any particular field.
