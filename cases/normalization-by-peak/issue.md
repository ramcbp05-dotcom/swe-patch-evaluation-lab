# `unit_area` output does not integrate to 1

**fieldlab 0.3.0**

We use `normalize.unit_area` to put emission spectra on a common footing before
comparing line strengths. The docstring says the result satisfies `trapezoid(y,
x) == 1`. It does not:

```python
import numpy as np
from fieldlab.normalize import unit_area

x = np.linspace(-20, 20, 2001)
narrow = np.exp(-0.5 * (x / 0.5) ** 2)
wide   = np.exp(-0.5 * (x / 5.0) ** 2)

print(np.trapezoid(unit_area(x, narrow), x))
print(np.trapezoid(unit_area(x, wide),   x))
```

```
1.2533141373154998
12.533141373148908
```

Both should be `1.0`. Worse, the two differ by exactly the ratio of the line
widths, so the error does not cancel when we compare two samples: it *is* the
comparison we are trying to make.

## Why we noticed

Integrated line strength from this pipeline scaled with instrument resolution.
Reprocessing the same physical sample on a higher-resolution grating changed
the reported strength by 40 %. The sample did not change; the line width did.

## Expected behaviour

After `unit_area(x, y)`, `trapezoid(result, x) == 1` to round-off, for any `x`
(uniform or not) and any feature width. The scale of `y` must not matter:
`unit_area(x, y)` and `unit_area(x, 100*y)` must be identical.

If `unit_peak` behaviour is wanted, that function already exists.
