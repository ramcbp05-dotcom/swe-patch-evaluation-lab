#!/usr/bin/env python3
"""Regenerate every ``bug.patch`` and candidate patch from this declarative table.

The patches under ``cases/`` are what the harness applies, but they are not the
source of truth (this file is. Each case states the edits that turn the clean
library into the broken one, and the edits that turn the broken one into each
candidate. The gold patch is derived: it is the reverse of ``bug.patch``, i.e.
"restore the reference implementation".

Keeping the edits here rather than hand-writing diffs means a change to the
library cannot silently leave a stale patch behind) run this script, and
``patchlab evaluate`` tells you immediately if a case stopped reproducing.

    python scripts/build_patches.py            # rewrite all patches
    python scripts/build_patches.py --check    # fail if any patch is stale
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Case table.  bug: clean -> broken.  variants: broken -> candidate.
# Every string below must appear exactly once in the file it edits.
# ---------------------------------------------------------------------------
CASES: dict[str, dict] = {

    #: 1 ------------------------------------------------------------------
    "shape-mismatch-pairwise-distance": {
        "module": "src/fieldlab/arrays.py",
        "bug": [(
            """    # (n_a, 1, d) - (1, n_b, d) -> (n_a, n_b, d); the inserted axes are what make
    # this an outer operation rather than an elementwise one.
    diff = pa[:, None, :] - pb[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))""",
            """    diff = pa[:, :, None] - pb[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))""",
        )],
        "variants": {
            "wrong_squared_distance": [(
                """    diff = pa[:, :, None] - pb[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))""",
                """    diff = pa[:, None, :] - pb[None, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff)""",
            )],
            "wrong_elementwise_norm": [(
                """    diff = pa[:, :, None] - pb[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))""",
                """    return np.linalg.norm(pa - pb, axis=1)""",
            )],
        },
    },

    #: 2 ------------------------------------------------------------------
    "normalization-by-peak": {
        "module": "src/fieldlab/normalize.py",
        "bug": [(
            """    area = float(np.trapezoid(ys, xs))
    if not np.isfinite(area) or area == 0.0:
        raise ValueError(f"integrated area is {area!r}; cannot normalise to unit area")
    return ys / area""",
            """    # Peak height stands in for the area: no integration, and the result still
    # lands in [0, 1].
    area = float(np.max(np.abs(ys)))
    if not np.isfinite(area) or area == 0.0:
        raise ValueError(f"integrated area is {area!r}; cannot normalise to unit area")
    return ys / area""",
        )],
        "variants": {
            "wrong_uniform_spacing": [(
                """    area = float(np.max(np.abs(ys)))""",
                """    area = float(np.trapezoid(ys))""",
            )],
            "wrong_rectangle_rule": [(
                """    area = float(np.max(np.abs(ys)))""",
                """    area = float(np.sum(ys) * (xs[1] - xs[0]))""",
            )],
        },
    },

    #: 3 ------------------------------------------------------------------
    "neumann-boundary-flux": {
        "module": "src/fieldlab/grids.py",
        "bug": [(
            """    elif bc == "neumann":
        # ghost u[-1] = u[0] and u[n] = u[n-1]  =>  zero flux through the wall
        out[0] = (arr[1] - arr[0]) * inv
        out[-1] = (arr[-2] - arr[-1]) * inv""",
            """    elif bc == "neumann":
        # mirror the field about the boundary cell and reuse the interior stencil
        out[0] = (arr[1] - 2.0 * arr[0] + arr[1]) * inv
        out[-1] = (arr[-2] - 2.0 * arr[-1] + arr[-2]) * inv""",
        )],
        "variants": {
            "wrong_zero_at_wall": [(
                """        out[0] = (arr[1] - 2.0 * arr[0] + arr[1]) * inv
        out[-1] = (arr[-2] - 2.0 * arr[-1] + arr[-2]) * inv""",
                """        out[0] = 0.0
        out[-1] = 0.0""",
            )],
            "wrong_copy_interior": [(
                """        out[0] = (arr[1] - 2.0 * arr[0] + arr[1]) * inv
        out[-1] = (arr[-2] - 2.0 * arr[-1] + arr[-2]) * inv""",
                """        out[0] = out[1]
        out[-1] = out[-2]""",
            )],
            "wrong_half_stencil": [(
                """        out[0] = (arr[1] - 2.0 * arr[0] + arr[1]) * inv
        out[-1] = (arr[-2] - 2.0 * arr[-1] + arr[-2]) * inv""",
                """        out[0] = 0.5 * (arr[1] - arr[0]) * inv
        out[-1] = 0.5 * (arr[-2] - arr[-1]) * inv""",
            )],
        },
    },

    #: 4 ------------------------------------------------------------------
    "variance-cancellation": {
        "module": "src/fieldlab/stats.py",
        "bug": [(
            """    mean = 0.0
    m2 = 0.0
    for count, value in enumerate(arr, start=1):
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)

    return max(m2, 0.0) / (n - ddof)""",
            """    # One pass, no running mean to carry: sum and sum-of-squares is enough.
    s1 = float(arr.sum())
    s2 = float((arr * arr).sum())
    return (s2 - s1 * s1 / n) / (n - ddof)""",
        )],
        "variants": {
            "wrong_clamp_negative": [(
                """    return (s2 - s1 * s1 / n) / (n - ddof)""",
                """    return max((s2 - s1 * s1 / n) / (n - ddof), 0.0)""",
            )],
            "wrong_ignores_ddof": [(
                """    s1 = float(arr.sum())
    s2 = float((arr * arr).sum())
    return (s2 - s1 * s1 / n) / (n - ddof)""",
                """    mean = float(arr.mean())
    return float(((arr - mean) ** 2).sum()) / n""",
            )],
            "wrong_float32_accumulate": [(
                """    s1 = float(arr.sum())
    s2 = float((arr * arr).sum())
    return (s2 - s1 * s1 / n) / (n - ddof)""",
                """    small = arr.astype(np.float32)
    mean = float(small.mean())
    return float(((small - mean) ** 2).sum()) / (n - ddof)""",
            )],
        },
    },

    #: 5 ------------------------------------------------------------------
    "periodic-index-precedence": {
        "module": "src/fieldlab/grids.py",
        "bug": [(
            """    if bc == "periodic":
        # The parentheses matter: % binds tighter than +, so `i + 1 % n` is
        # `i + (1 % n)`, which is `i + 1` for every n > 1 and runs off the end.
        return (i - 1) % n, (i + 1) % n""",
            """    if bc == "periodic":
        return i - 1 % n, i + 1 % n""",
        )],
        "variants": {
            "wrong_clamp_instead_of_wrap": [(
                """        return i - 1 % n, i + 1 % n""",
                """        return max(i - 1, 0), min(i + 1, n - 1)""",
            )],
            "wrong_abs_left_neighbour": [(
                """        return i - 1 % n, i + 1 % n""",
                """        return abs(i - 1) % n, (i + 1) % n""",
            )],
        },
    },

    #: 6 ------------------------------------------------------------------
    "silent-unsorted-interpolation": {
        "module": "src/fieldlab/interpolate.py",
        "bug": [(
            """    if not np.all(np.diff(xs) > 0.0):
        raise ValueError("x must be strictly increasing")
""",
            "",
        )],
        "variants": {
            "wrong_sorts_silently": [(
                """    if extrapolate not in ("raise", "clamp", "nan"):""",
                """    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if extrapolate not in ("raise", "clamp", "nan"):""",
            )],
            "wrong_allows_duplicates": [(
                """    if extrapolate not in ("raise", "clamp", "nan"):""",
                """    if not np.all(np.diff(xs) >= 0.0):
        raise ValueError("x must be strictly increasing")
    if extrapolate not in ("raise", "clamp", "nan"):""",
            )],
        },
    },

    #: 7 ------------------------------------------------------------------
    "input-array-mutated": {
        "module": "src/fieldlab/normalize.py",
        "bug": [(
            """    centred = arr - arr.mean()
    scale = float(centred.std(ddof=ddof))""",
            """    # Centre in place: one fewer temporary for a large field.
    arr -= arr.mean()
    centred = arr
    scale = float(centred.std(ddof=ddof))""",
        )],
        "variants": {
            "wrong_restore_after_mutating": [(
                """    arr -= arr.mean()
    centred = arr
    scale = float(centred.std(ddof=ddof))
    if scale == 0.0:
        raise ValueError("cannot z-score a constant field")
    return centred / scale""",
                """    mean = arr.mean()
    arr -= mean
    scale = float(arr.std(ddof=ddof))
    if scale == 0.0:
        arr += mean
        raise ValueError("cannot z-score a constant field")
    out = arr / scale
    arr += mean
    return out""",
            )],
            "wrong_copy_too_late": [(
                """    arr -= arr.mean()
    centred = arr""",
                """    arr -= arr.mean()
    centred = arr.copy()""",
            )],
        },
    },

    #: 8 ------------------------------------------------------------------
    "floor-division-weights": {
        "module": "src/fieldlab/stats.py",
        "bug": [(
            """    total = float(weights.sum())
    if total == 0.0:
        raise ValueError("weights sum to zero")
    return float(np.dot(weights, values) / total)""",
            """    total = weights.sum()
    if total == 0.0:
        raise ValueError("weights sum to zero")
    # Normalise the weights first so the dot product is already an average.
    normalised = weights // total
    return float(np.dot(normalised, values))""",
        )],
        "variants": {
            "wrong_floor_the_result": [(
                """    normalised = weights // total
    return float(np.dot(normalised, values))""",
                """    return float(np.dot(weights, values) // total)""",
            )],
            "wrong_rounded_weights": [(
                """    normalised = weights // total
    return float(np.dot(normalised, values))""",
                """    normalised = np.round(weights / total, 2)
    return float(np.dot(normalised, values))""",
            )],
        },
    },

    #: 9 ------------------------------------------------------------------
    "affine-unit-conversion": {
        "module": "src/fieldlab/units.py",
        "bug": [(
            """    "temperature": {
        "K": (1.0, 0.0),
        "C": (1.0, 273.15),
        "F": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    },""",
            """    "temperature": {
        "K": (1.0, 0.0),
        "C": (274.15, 0.0),
        "F": (5.0 / 9.0, 0.0),
    },""",
        )],
        "variants": {
            "wrong_offset_only": [(
                """        "C": (274.15, 0.0),
        "F": (5.0 / 9.0, 0.0),""",
                """        "C": (1.0, 273.15),
        "F": (1.0, 241.15),""",
            )],
            "wrong_celsius_only": [(
                """        "C": (274.15, 0.0),
        "F": (5.0 / 9.0, 0.0),""",
                """        "C": (1.0, 273.15),
        "F": (5.0 / 9.0, 0.0),""",
            )],
        },
    },
}


def apply_edits(text: str, edits: list[tuple[str, str]], where: str) -> str:
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"{where}: anchor occurs {count} time(s), expected exactly 1:\n"
                f"--- anchor ---\n{old}\n--------------"
            )
        text = text.replace(old, new)
    return text


def make_patch(before: str, after: str, module: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{module}",
        tofile=f"b/{module}",
        n=3,
    )
    return "".join(diff)


def build(check_only: bool) -> int:
    stale: list[str] = []
    written = 0

    for case_id, cfg in CASES.items():
        module = cfg["module"]
        case_dir = REPO_ROOT / "cases" / case_id
        clean = (REPO_ROOT / module).read_text()

        broken = apply_edits(clean, cfg["bug"], f"{case_id}/bug")
        if broken == clean:
            raise SystemExit(f"{case_id}: the bug edits changed nothing")

        artefacts = {
            case_dir / "bug.patch": make_patch(clean, broken, module),
            case_dir / "patches" / "gold.patch": make_patch(broken, clean, module),
        }
        for name, edits in cfg["variants"].items():
            variant = apply_edits(broken, edits, f"{case_id}/{name}")
            if variant == broken:
                raise SystemExit(f"{case_id}/{name}: the variant edits changed nothing")
            artefacts[case_dir / "patches" / f"{name}.patch"] = make_patch(
                broken, variant, module
            )

        for path, content in artefacts.items():
            if check_only:
                current = path.read_text() if path.is_file() else None
                if current != content:
                    stale.append(str(path.relative_to(REPO_ROOT)))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                written += 1

    if check_only:
        if stale:
            print("stale patches (run scripts/build_patches.py):", file=sys.stderr)
            for path in stale:
                print(f"  {path}", file=sys.stderr)
            return 1
        print(f"all patches for {len(CASES)} cases are up to date")
        return 0

    print(f"wrote {written} patch files for {len(CASES)} cases")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the committed patches match this table")
    args = parser.parse_args()
    return build(args.check)


if __name__ == "__main__":
    raise SystemExit(main())
