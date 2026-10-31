# Root cause: input-array-mutated

> **Seeded bug.** This defect was introduced deliberately, by
> `scripts/build_patches.py`, to exercise the patch-evaluation harness. It was
> never present in a released version of `fieldlab` and was not found in the
> wild. The issue report in `issue.md` is written in the voice of a user who
> hit it, and the numbers quoted there were measured against the broken code,
> but the bug itself is synthetic.

## The defect

```python
# broken: "Centre in place: one fewer temporary for a large field."
arr -= arr.mean()
centred = arr

# correct
centred = arr - arr.mean()
```

The in-place form is a real optimisation, and it is safe only if `arr` is
privately owned. Two lines earlier:

```python
arr = np.asarray(a, dtype=float)
```

`np.asarray` returns the **same object** when the input is already a float64
`ndarray`. So `arr` is the caller's array, and `arr -=` writes into it.

## Why the tests did not catch it

Every doctest and every quick check in the issue uses a Python list, and
`np.asarray` has to allocate for a list. The bug is invisible for list input
and certain for array input. It is also invisible for a `float32` or integer
array, since `dtype=float` forces a conversion and therefore a copy. The only
inputs that trigger it are the ones real code passes: float64 arrays, and views
into them.

The view case is what made the issue expensive. `runs[:, j]` is a view, so
centring it writes through to `runs`, and the corruption surfaced two pipeline
stages later as "the next stage computes zero means".

## The three properties a pure array function owes its caller

1. **The input is bit-for-bit unchanged**, including on the error path.
2. **A read-only input works.** This is the one worth testing first, because
   in-place arithmetic on a read-only array raises unconditionally: there is no
   data-dependent luck involved.
3. **The result does not share memory with the input**, so the caller can write
   to what it gets back.

The broken version violates 1 and 2. A patch can satisfy 1 and 3 and still
violate 2, which is what the third candidate does.

## Blast radius

The seeded bug breaks 2 of the 5 `TestZscore` tests in
`tests/test_normalize.py`. `unit_area` and `unit_peak` already centre out of
place, so they are unaffected: the issue tests cover them anyway, because a
patch that "fixes aliasing" by adding in-place arithmetic elsewhere should not
pass.

## Candidate review

| candidate | verdict | reasoning |
|---|---|---|
| `gold` | **resolves** | Reverts to the out-of-place subtraction. One temporary, three guarantees. |
| `wrong_copy_too_late` | rejected | `centred = arr.copy()` *after* `arr -= arr.mean()`. This is the patch that follows from a bug report phrased as "the result aliases my input": the return value becomes distinct and `test_result_is_a_distinct_object` passes. The caller's buffer was already overwritten one line earlier, so every test in `TestInputIsNotModified` still fails. The copy is also pure cost: it allocates the temporary the optimisation was trying to avoid, and keeps the damage. |
| `wrong_restore_after_mutating` | rejected | Subtracts the mean in place, divides into a fresh array, then adds the mean back. The input *is* restored, to within a rounding error, and the return value is distinct: so the two obvious assertions pass and this is the candidate that would survive review. Three things remain wrong: it raises on a read-only array (`test_read_only_input`), the restore is not bit-exact so `test_random_arrays_are_untouched_bitwise` fails on roughly half of the seeds, and nothing restores the array if the `std` call or the division raises. |

## Lesson for the test suite

`test_does_not_modify_input` with `assert_array_equal` was already in the suite
and already catches the seeded bug. What it does not reliably catch is
`wrong_restore_after_mutating`, because `(a - mean) + mean == a` is exact for
some data and off by one ulp for other data: measured over six seeds of normal
data, it was bit-exact for four of them.

The test that is deterministic is the read-only one. Setting `a.flags.writeable
= False` turns "did this function happen to write to my array" into "can this
function write to an array at all", and that question has no data-dependent
answer. For any function documented as pure, it is the cheapest complete check
available.
