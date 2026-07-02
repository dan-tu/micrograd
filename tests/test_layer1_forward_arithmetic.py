"""Layer 1 — forward arithmetic (values only, gradients come in Layer 3).

Get every arithmetic operation computing the right NUMBER first. You'll also
wire up each op's `_backward` closure here per the engine.py docstrings, but
this layer only checks `.data`, so a wrong closure won't be caught until
Layer 3 — that's fine.

    Implement in engine.py: __add__, __mul__, __pow__, and the reverse/
    convenience ops __neg__, __radd__, __sub__, __rsub__, __rmul__,
    __truediv__, __rtruediv__.
"""

from micrograd.engine import Value


def test_ops_return_value_objects():
    assert isinstance(Value(1.0) + Value(2.0), Value)
    assert isinstance(Value(1.0) * Value(2.0), Value)
    assert isinstance(Value(1.0) ** 2, Value)


def test_add():
    assert (Value(2.0) + Value(3.0)).data == 5.0


def test_add_with_python_number_on_right():
    # exercises __add__ coercing a raw float into a Value
    assert (Value(2.0) + 3.0).data == 5.0


def test_add_with_python_number_on_left():
    # 3.0 + Value(...) dispatches to __radd__
    assert (3.0 + Value(2.0)).data == 5.0


def test_mul():
    assert (Value(2.0) * Value(4.0)).data == 8.0


def test_mul_with_python_number_on_left():
    # dispatches to __rmul__
    assert (3.0 * Value(2.0)).data == 6.0


def test_pow_integer_exponent():
    assert (Value(3.0) ** 2).data == 9.0


def test_pow_fractional_exponent():
    assert abs((Value(4.0) ** 0.5).data - 2.0) < 1e-9


def test_neg():
    assert (-Value(3.0)).data == -3.0


def test_sub():
    assert (Value(5.0) - Value(2.0)).data == 3.0


def test_sub_with_python_number_on_left():
    # dispatches to __rsub__
    assert (10.0 - Value(4.0)).data == 6.0


def test_truediv():
    assert (Value(6.0) / Value(2.0)).data == 3.0


def test_truediv_with_python_number_on_left():
    # dispatches to __rtruediv__
    assert (12.0 / Value(4.0)).data == 3.0


def test_chained_expression():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    assert (a * b + c).data == 4.0
