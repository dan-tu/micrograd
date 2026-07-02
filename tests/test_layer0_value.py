"""Layer 0 — the Value object exists.

This layer is a smoke test: `__init__` and `__repr__` are given to you, so
these pass out of the box. If they DON'T pass, your environment or import
path is wrong — fix that before touching anything else.

    Implement in engine.py: (nothing — provided)
"""

from micrograd.engine import Value


def test_value_stores_data():
    assert Value(2.0).data == 2.0


def test_value_data_can_be_int():
    assert Value(3).data == 3


def test_grad_starts_at_zero():
    assert Value(1.5).grad == 0.0


def test_two_values_are_distinct_objects():
    a, b = Value(1.0), Value(1.0)
    assert a is not b


def test_repr_shows_data_and_grad():
    r = repr(Value(2.0))
    assert "data" in r and "2.0" in r
    assert "grad" in r
