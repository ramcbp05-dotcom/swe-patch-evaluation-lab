# Root cause: periodic-index-precedence

> **Seeded bug.** This defect was introduced deliberately, by
> `scripts/build_patches.py`, to exercise the patch-evaluation harness. It was
> never present in a released version of `fieldlab` and was not found in the
> wild. The issue report in `issue.md` is written in the voice of a user who
> hit it, and the numbers quoted there were measured against the broken code,
> but the bug itself is synthetic.

## The defect

```python
return i - 1 % n, i + 1 % n        # broken
return (i - 1) % n, (i + 1) % n    # correct
```

`%` has higher precedence than `+` and `-` in Python, so `i + 1 % n` parses as
`i + (1 % n)`. For every `n > 1`, `1 % n` is `1`, so the expression is just `i
+ 1` and the modulo contributes nothing. Likewise `i - 1 % n` is `i - 1`.

The removed line in the reference implementation is a comment warning about
exactly this, which is a small lesson in itself: the comment was deleted in the
same edit that introduced the bug.

## Why half of it went unnoticed

The left-hand side is wrong too, and it still works. `i - 1` at `i = 0` gives
`-1`, and NumPy reads a negative index from the end of the array: `u[-1]` *is*
the last element, which is what a periodic grid wants. So the left wrap is
accidentally correct when the index is used for NumPy indexing, and incorrect
as a *value*: anything that compares the index, stores it, or uses it in
arithmetic sees `-1` instead of `n - 1`.

The right-hand side has no such accident. `n` is out of range in both
directions and raises immediately.

This asymmetry is why the bug report only mentions the right end, and why the
test suite needs both `test_wrap_at_the_left_end` (which checks the returned
*value*) and the stepping tests (which check the *structure*).

## The structural invariant

On a ring of `n` cells the right-neighbour map is a cyclic permutation. Three
consequences, each a test:

* every index returned is in `[0, n)`;
* applying the map `n` times is the identity;
* the map is a bijection: `sorted(rights) == list(range(n))`.

The third is the cheapest complete check, and it is the one that rejects both
wrong patches.

## Blast radius

`neighbour_indices` is a public helper; `laplacian_1d` and `gradient_1d` build
their periodic stencils with slicing rather than by calling it, so they are
unaffected. The seeded bug breaks 4 of the 7 `TestNeighbourIndices` tests in
`tests/test_grids.py`.

## Candidate review

| candidate | verdict | reasoning |
|---|---|---|
| `gold` | **resolves** | Parenthesises both expressions. One-line fix; the comment explaining why goes back with it. |
| `wrong_clamp_instead_of_wrap` | rejected | `max(i-1, 0), min(i+1, n-1)` is the fastest way to make the `IndexError` go away, and it is the Neumann branch copied two lines up: so the diff looks consistent with the rest of the function. The grid is no longer periodic: cell 0 becomes its own left neighbour, the right-neighbour map is not a permutation, and stepping right `n` times does not return. A stencil built on it applies a reflecting wall to a problem the caller declared periodic. |
| `wrong_abs_left_neighbour` | rejected | Fixes the right neighbour properly and reaches for `abs` on the left, which is a plausible way to "keep the index positive". At `i = 0` it gives `abs(-1) % n == 1`, so cell 0's left neighbour is cell 1. Every index is in range and the interior is correct, so only the two tests that look at the left end and at the bijection catch it. Fails `test_wrap_at_the_left_end`, `test_stepping_left_n_times_returns_to_start` and `test_left_and_right_are_mutually_inverse`. |

## Lesson for the test suite

A range assertion is not enough: both wrong patches return in-range indices.
What separates "in range" from "periodic" is checking an algebraic property of
the map rather than individual values: bijection and mutual inverse. Those two
tests are three lines each and they close the whole case.
