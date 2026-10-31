# `weighted_mean` returns 0.0 for everything

**fieldlab 0.3.0**

```python
>>> from fieldlab.stats import weighted_mean
>>> weighted_mean([10.0, 20.0], [1, 2])
0.0
```

The answer is `16.666...`. It is zero for essentially every input:

```python
>>> weighted_mean([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
0.0
>>> weighted_mean([5.0, 5.0, 5.0], [0.2, 0.3, 0.5])
0.0
>>> weighted_mean([100.0, 200.0], [7.0, 3.0])
0.0
```

A single weight works, which is the only reason anything in our pipeline still
produces a number:

```python
>>> weighted_mean([42.0], [1.0])
42.0
```

## Why this took a month to surface

Our detector weights are the thing being averaged away, and the quantity we
average is a *residual*: centred data whose true weighted mean is close to zero
anyway. A reported mean of exactly `0.0` looked like a well-calibrated run. We
only caught it when we fed in an uncentred channel and still got `0.0`.

## Expected behaviour

`weighted_mean(x, w)` is `sum(w*x) / sum(w)`:

```python
>>> weighted_mean([10.0, 20.0], [1, 2])
16.666666666666668
>>> weighted_mean([1.0, 11.0], [9.0, 1.0])
2.0
```

Integer weights must behave exactly like the same weights given as floats, and
the result must not change when all the weights are multiplied by a constant.
