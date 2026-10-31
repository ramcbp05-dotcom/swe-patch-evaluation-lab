# `variance` returns a negative number, `stddev` then raises

**fieldlab 0.3.0**

Our absolute-pressure gauge reports readings around `1e8` Pa with a spread of a
few hundredths. `stats.variance` returns a negative number for that data, and
`stats.stddev` dies on the square root:

```python
import numpy as np
from fieldlab import stats

readings = 1.0e8 + np.array([-0.02, -0.01, 0.0, 0.01, 0.02])

stats.variance(readings)
```

```
-0.00531005859375
```

```python
stats.stddev(readings)
```

```
ValueError: math domain error
```

The correct answer is `0.00025`. Subtracting the offset by hand gives it:

```python
>>> stats.variance(readings - 1.0e8)
0.00025000000000000017
```

A variance cannot be negative, so this is not a tolerance problem: the result
has no correct digits at all.

## Scale of the problem

```python
>>> a = np.random.default_rng(0).normal(0, 1, 2000)
>>> for offset in (0, 1e3, 1e6, 1e9):
...     print(f"{offset:>8.0e}  {stats.variance(a + offset):.9f}")
   0e+00  0.993841806
   1e+03  0.993841806
   1e+06  0.993621826
   1e+09  -26.843545600
```

Variance is shift invariant, so all four of those must be the same number.

## Expected behaviour

`variance` is shift invariant to the accuracy that double precision allows, and
is never negative. Constant input gives exactly `0.0`. `ddof` keeps its current
meaning: `variance(x)` is the **sample** variance, dividing by `n - 1`.
