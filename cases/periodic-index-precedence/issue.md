# `neighbour_indices` returns an out-of-range index with `bc="periodic"`

**fieldlab 0.3.0**

On a periodic grid the right neighbour of the last cell is the first cell. It
is not:

```python
>>> from fieldlab.grids import neighbour_indices
>>> neighbour_indices(7, 8, "periodic")
(6, 8)
```

`8` is not a valid index into a grid of size 8. Indexing with it raises:

```python
>>> import numpy as np
>>> u = np.zeros(8)
>>> left, right = neighbour_indices(7, 8, "periodic")
>>> u[right]
IndexError: index 8 is out of bounds for axis 0 with size 8
```

The left neighbour of cell 0 is wrong in the same way: it comes back as `-1`,
which does not raise because NumPy reads a negative index from the end. That
one happens to give the right *answer* for a periodic grid, which is why we
only ever saw the problem at the right-hand end.

## Scope

Every grid size is affected:

```python
>>> for n in (2, 3, 8, 101):
...     print(n, neighbour_indices(n - 1, n, "periodic"))
2 (0, 2)
3 (1, 3)
8 (6, 8)
101 (99, 101)
```

The modulo appears to have no effect whatsoever.

`bc="neumann"` and `bc="dirichlet"` are fine.

## Expected behaviour

For `bc="periodic"`, both returned indices are in `[0, n)` for every `i` in
`[0, n)`, and stepping right `n` times returns to the starting cell. Cell 0's
left neighbour is `n - 1`, and cell `n - 1`'s right neighbour is `0`.
