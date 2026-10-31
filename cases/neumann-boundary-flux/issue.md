# Tracer budget does not close with `bc="neumann"`

**fieldlab 0.3.0**

A closed box has no flux through its walls, so the total amount of the diffused
quantity must be constant. With `grids.laplacian_1d(..., bc="neumann")` it
drifts:

```python
import numpy as np
from fieldlab.grids import laplacian_1d

u = np.zeros(64)
u[30:34] = 1.0                      # a blob in the middle of a closed box
total0 = u.sum()

for _ in range(4000):               # explicit diffusion, dt chosen for stability
    u = u + 0.2 * laplacian_1d(u, dx=1.0, bc="neumann")

print(total0, u.sum())
```

```
4.0 4.063447943230636
```

We gained 1.6 % of material out of nowhere. The box is closed.

It is small, and it stops growing once the field flattens out, which is why it
took three months to notice: it reads as a tolerance problem until you check
that it is the same 1.6 % for every initial condition and every step count.

## Narrowing it down

The operator itself does not sum to zero:

```python
>>> u = np.array([1.0, 2.0, 4.0, 7.0])
>>> laplacian_1d(u, dx=1.0, bc="neumann")
array([ 2.,  1.,  1., -6.])
>>> laplacian_1d(u, dx=1.0, bc="neumann").sum()
-2.0
```

A discrete Laplacian with zero flux at both ends is a sum of differences
between neighbouring cells, so every term has to cancel: the row sums of the
matrix must be zero. Here they are not, and the excess sits entirely in the two
boundary cells: the three interior values are `1, 1, 1`, which is right for
this field.

Two things hid it from us:

* A field that is **antisymmetric about the mid-point**: a linear ramp, for
  instance: gives equal and opposite errors at the two walls, so the total is
  conserved exactly and the budget check passes.
* Checking the interior against an analytic second derivative passes too:
  `laplacian_1d(x**2)` is `2.0` everywhere inside the domain. The interior
  stencil is not the problem.

## Expected behaviour

For `bc="neumann"` the operator must satisfy `laplacian_1d(u).sum() == 0` for
**every** `u`, and must reduce to the analytic second derivative in the
interior. The periodic branch already satisfies both.
