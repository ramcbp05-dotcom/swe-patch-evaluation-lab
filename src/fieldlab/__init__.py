"""fieldlab: small array utilities for 1-D scientific field data.

The library is the subject of the patch-evaluation lab in this repository: the
code here is the *reference* implementation, and each case under ``cases/``
seeds a documented bug into one of these modules and asks a patch to remove it.

Modules
-------
arrays        shape handling and pairwise geometry
normalize     normalisation that preserves a stated invariant
grids         finite differences and boundary conditions
stats         numerically stable summary statistics
interpolate   resampling with explicit out-of-range behaviour
units         conversion to SI, including affine maps
"""

from . import arrays, grids, interpolate, normalize, stats, units

__version__ = "0.3.0"
__all__ = ["arrays", "grids", "interpolate", "normalize", "stats", "units"]
