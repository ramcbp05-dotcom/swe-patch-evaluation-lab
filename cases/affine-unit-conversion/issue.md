# `to_si` gives impossible temperatures

**fieldlab 0.3.0**

```python
>>> from fieldlab.units import to_si
>>> to_si("temperature", 0.0, "C")
0.0
>>> to_si("temperature", 100.0, "C")
27415.000000000004
>>> to_si("temperature", 32.0, "F")
17.77777777777778
```

Expected `273.15`, `373.15` and `273.15`. Water freezing at 0 K and boiling at
27 415 K are not plausible numbers.

Exactly one input is right:

```python
>>> to_si("temperature", 1.0, "C")
274.15
```

which is the correct value for 1 °C: presumably the case that was checked.

Pressure and length are fine:

```python
>>> to_si("pressure", 1.0, "psi")
6894.757293168361
>>> to_si("length", 12.0, "in")
0.30479999999999996
```

## What we think is going on

`_CONVERSION` stores `(scale, offset)` per unit and `to_si` applies `value *
scale + offset`. The temperature row has `offset = 0.0` for both `C` and `F`,
with `scale = 274.15` for Celsius. A scale factor cannot express `K = C +
273.15`: that map has slope 1 and intercept 273.15, and `value * 274.15`
happens to agree with it only at `value == 1`.

Fahrenheit has the opposite problem: it has the right slope (5/9) and no
intercept.

## Expected behaviour

```
K = C + 273.15
K = (F − 32) × 5/9 + 273.15
```

Both anchors must hold, not one: 0 °C → 273.15 K and 100 °C → 373.15 K; 32 °F →
273.15 K and 212 °F → 373.15 K. A 180 °F span must come out as a 100 K span.
