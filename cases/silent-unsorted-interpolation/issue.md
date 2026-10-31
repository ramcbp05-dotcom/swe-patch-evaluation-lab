# `resample` invents values when the input is out of order

**fieldlab 0.3.0**, numpy 2.1

Our logger buffers and occasionally flushes a block late, so the raw trace
arrives slightly out of order. `interpolate.resample` does not complain: it
returns numbers that are not interpolations of anything.

Fully reversed input is the clearest demonstration:

```python
import numpy as np
from fieldlab.interpolate import resample

x = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
y = np.array([40.0, 30.0, 20.0, 10.0, 0.0])

resample(x, y, [1.5, 2.5])
```

```
array([0., 0.])
```

The correct answers are `15.0` and `25.0`. Every query point in range comes
back as `y[0]`.

## Why it took us so long to find

A single inversion in *linear* data gives the right answer:

```python
>>> x = np.array([0.0, 1.0, 3.0, 2.0, 4.0])      # 3.0 before 2.0
>>> y = np.array([0.0, 10.0, 30.0, 20.0, 40.0])
>>> resample(x, y, [0.5, 1.5, 2.5, 3.5])
array([ 5., 15., 25., 35.])
```

Those are exactly right, so our smoke test passed. The error only appears once
the field has curvature:

```python
>>> x = np.array([0.0, 1.0, 4.0, 3.0, 2.0, 5.0])   # same two points swapped
>>> y = np.array([0.0, 1.0, 16.0, 9.0, 4.0, 25.0]) # y = x**2
>>> resample(x, y, [1.5, 2.5, 3.5, 4.5])
array([ 3.5,  8.5, 13.5, 21.5])
>>> # correct:
array([ 2.5,  6.5, 12.5, 20.5])
```

On a real shuffled trace the result is unrecognisable. Resampling `y = sin(x)`
from 60 scattered samples, over six different shufflings:

```
max |error|   resampled range        true data range
1.4862        ( 0.086,  0.497)       (-0.989,  1.000)
1.9995        (-1.000, -0.919)       (-1.000,  1.000)
1.9240        ( 0.502,  0.925)       (-1.000,  0.998)
```

The error reaches twice the full amplitude of the signal, and note the
resampled range: it *collapses* rather than exploding. The output is a
plausible-looking low-amplitude trace, not an obvious blow-up, so a range check
or a NaN check on the output will not flag it.

We traced it to `numpy.interp`, which documents that `xp` must be increasing
but does not check it: it binary-searches an array it assumes is sorted.
Earlier versions of `resample` rejected this input; the check seems to have
been dropped.

## Expected behaviour

`resample` raises `ValueError` for `x` that is not strictly increasing, naming
the problem. Sorting it for us is not what we want: out-of-order rows mean the
upstream cleanup stage failed, and we need to know that rather than have it
papered over. Duplicate `x` values must be rejected for the same reason: the
result would depend on which duplicate is picked.
