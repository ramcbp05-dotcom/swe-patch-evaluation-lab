"""Conversion to SI base units, including the affine temperature maps."""

from __future__ import annotations

__all__ = ["SI_UNIT", "to_si", "supported_units"]

SI_UNIT = {"temperature": "K", "pressure": "Pa", "length": "m"}

# quantity -> unit -> (scale, offset), applied as si = value * scale + offset.
# Temperature needs the offset: degC -> K and degF -> K are affine maps, not
# scale factors, so a table of bare factors cannot express them.
_CONVERSION = {
    "temperature": {
        "K": (1.0, 0.0),
        "C": (1.0, 273.15),
        "F": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    },
    "pressure": {
        "Pa": (1.0, 0.0),
        "kPa": (1000.0, 0.0),
        "MPa": (1.0e6, 0.0),
        "bar": (100000.0, 0.0),
        "atm": (101325.0, 0.0),
        "psi": (6894.757293168361, 0.0),
    },
    "length": {
        "m": (1.0, 0.0),
        "km": (1000.0, 0.0),
        "cm": (0.01, 0.0),
        "mm": (0.001, 0.0),
        "um": (1.0e-6, 0.0),
        "in": (0.0254, 0.0),
        "ft": (0.3048, 0.0),
    },
}


def supported_units(quantity: str) -> tuple[str, ...]:
    """Units accepted for ``quantity``."""
    if quantity not in _CONVERSION:
        raise ValueError(
            f"unknown quantity {quantity!r}; expected one of {tuple(_CONVERSION)}"
        )
    return tuple(_CONVERSION[quantity])


def to_si(quantity: str, value: float, unit: str) -> float:
    """Convert ``value`` in ``unit`` to the SI base unit for ``quantity``."""
    if quantity not in _CONVERSION:
        raise ValueError(
            f"unknown quantity {quantity!r}; expected one of {tuple(_CONVERSION)}"
        )
    table = _CONVERSION[quantity]
    if unit not in table:
        raise ValueError(
            f"unit {unit!r} is not supported for {quantity}; "
            f"expected one of {tuple(table)}"
        )
    scale, offset = table[unit]
    return float(value) * scale + offset
