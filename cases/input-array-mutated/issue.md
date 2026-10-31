# `zscore` overwrites the array you pass it

**fieldlab 0.3.0**

`normalize.zscore` modifies its argument:

```python
import numpy as np
from fieldlab.normalize import zscore

a = np.array([1.0, 2.0, 3.0, 4.0])
zscore(a)
print(a)
```

```
[-1.5 -0.5  0.5  1.5]
```

`a` is supposed to still be `[1, 2, 3, 4]`. The module docstring says these
functions are pure and that "callers rely on that when they normalise a view of
a larger dataset": which is exactly what we do.

## How it bit us

We z-score one channel of a run at a time:

```python
>>> runs = np.loadtxt("run.csv")          # (n_samples, 8)
>>> scored = [zscore(runs[:, j]) for j in range(8)]
>>> # runs is now centred column by column, in place
```

`runs[:, j]` is a view, so every column of the original array was silently
centred. The next stage of the pipeline computed its own statistics from `runs`
and got means of zero for everything. We spent two days looking for the bug in
the next stage.

## It also broke read-only input

```python
>>> a = np.arange(5.0)
>>> a.flags.writeable = False
>>> zscore(a)
ValueError: output array is read-only
```

We mark shared arrays read-only on purpose, and this used to work.

## Not reproducible from a list

```python
>>> values = [1.0, 2.0, 3.0, 4.0]
>>> zscore(values)
array([-1.16189501, -0.38729833,  0.38729833,  1.16189501])
>>> values
[1.0, 2.0, 3.0, 4.0]
```

A list is fine, because `np.asarray` has to build a new array for it. The bug
only appears when the input is already a float64 array: which is the case in
every real call and in none of our doctests.

## Expected behaviour

`zscore` leaves its argument bit-for-bit unchanged, works on a read-only array,
and returns an array that does not share memory with the input. The same goes
for the other functions in the module.
