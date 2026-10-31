# `pairwise_distances` raises a broadcast error for rectangular inputs

**fieldlab 0.3.0**, numpy 2.1

Computing distances between two point sets of different sizes fails:

```python
import numpy as np
from fieldlab.arrays import pairwise_distances

sensors = np.random.default_rng(0).normal(size=(5, 3))   # 5 points in 3-D
targets = np.random.default_rng(1).normal(size=(7, 3))   # 7 points in 3-D

pairwise_distances(sensors, targets)
```

```
ValueError: operands could not be broadcast together with shapes (5,3,1) (1,7,3)
```

Expected an `(5, 7)` matrix of distances.

## What makes it confusing

It does not fail everywhere. This works:

```python
>>> p = np.random.default_rng(0).normal(size=(3, 3))
>>> pairwise_distances(p, p).shape
(3, 3)
```

so the function looks fine in the square case we use in the docs. It also works
for `(n_a, 7)` against `(7, 7)`: any input where the feature dimension happens
to equal the number of points in the second set.

1-D input fails the same way as soon as the two sets differ in length:

```python
>>> pairwise_distances([0.0, 1.0, 2.0], [0.0, 10.0])
ValueError: operands could not be broadcast together with shapes (3,1,1) (1,2,1)
```

## Expected behaviour

`pairwise_distances(a, b)` returns an `(n_a, n_b)` array where entry `(i, j)`
is the Euclidean distance between `a[i]` and `b[j]`, for any `n_a`, `n_b` and
any shared feature dimension `d`. A mismatch in `d` (and only in `d`) should
raise.
