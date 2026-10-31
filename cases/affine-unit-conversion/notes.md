# Root cause: affine-unit-conversion

> **Seeded bug.** This defect was introduced deliberately, by
> `scripts/build_patches.py`, to exercise the patch-evaluation harness. It was
> never present in a released version of `fieldlab` and was not found in the
> wild. The issue report in `issue.md` is written in the voice of a user who
> hit it, and the numbers quoted there were measured against the broken code,
> but the bug itself is synthetic.

## The defect

```python
# broken
"C": (274.15, 0.0),
"F": (5.0 / 9.0, 0.0),

# correct
"C": (1.0, 273.15),
"F": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
```

The data structure is right: `to_si` applies `value * scale + offset` and never
changed. Only the two temperature rows were collapsed to pure scale factors.

## Why `274.15` is the interesting part

`274.15` is not a typo or a random number. It is the correct kelvin value for
**1 °C**, and it is exactly what you get if you verify a multiplicative
conversion against a single test case and that test case is `1.0`.

That is the whole lesson of this case. An affine map `y = ax + b` has two free
parameters. One data point constrains one combination of them, so any single
anchor can always be matched by a wrong model. `274.15 × 1 = 274.15` is right;
`274.15 × 0 = 0` and `274.15 × 100 = 27415` are not.

Fahrenheit shows the mirror image: slope `5/9` is correct, intercept zero is
not, so it is right for a *difference* of temperatures and wrong for every
absolute temperature. Code that only ever computed `ΔT` would never notice.

## Blast radius

The seeded bug breaks 8 of the 9 parametrised temperature tests in
`tests/test_units.py`, plus the two cross-scale checks. Pressure and length are
untouched, and neither needs an offset: they really are multiplicative, which
is why a single-factor table seemed adequate.

## Candidate review

| candidate | verdict | reasoning |
|---|---|---|
| `gold` | **resolves** | Both rows get their offset back. `273.15 - 32·5/9` is written out rather than pre-computed as `255.372` so the derivation stays visible in the source. |
| `wrong_offset_only` | rejected | Celsius is fixed properly. Fahrenheit becomes `(1.0, 241.15)`, and `241.15` is not arbitrary either: it is `273.15 − 32`, picked so that `32 °F` maps to exactly `273.15 K`. One anchor matched, the other missed: `212 °F` gives `453.15 K` instead of `373.15 K`. The 80 K error is large and completely invisible at the one temperature that was checked. `test_slope_is_five_ninths` reports the span as `180.0` instead of `100.0`, which names the defect precisely. |
| `wrong_celsius_only` | rejected | Celsius fixed, Fahrenheit left multiplicative. Honest and incomplete. Every test in `TestCelsius` passes, so a reviewer who reads the issue title as "Celsius conversion wrong" would approve it. `TestFahrenheit` and `TestScalesAgree` fail. |

## Lesson for the test suite

An affine map needs **two anchors per unit**, and the existing suite had them
for Fahrenheit (32 and 212) but relied on a single-point check for the slope.

The two tests that are worth copying into any unit-conversion suite are
structural rather than tabular:

* **the span test**: `to_si(212, "F") − to_si(32, "F") == 100.0` isolates the
  slope from the intercept, so it fails for exactly one of the two possible
  mistakes and says which;
* **the cross-scale test**: the two scales must agree at `−40`, and a
  parametrised round trip `C → F → K` must agree with `C → K` at every
  temperature. That one needs no reference table at all and catches any
  inconsistency between the rows.

`test_affine_not_linear` is the cheapest of the lot: a pure scale factor sends
0 to 0, so asserting that `0 °C` is *not* `0 K` rejects the entire class of
multiplicative-table bugs in one line.
