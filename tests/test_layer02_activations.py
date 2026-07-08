"""Layer 2 — activation functions (forward only).

The nonlinearities that make a network more than a stack of linear maps.
Again, only `.data` is checked here; their gradients are verified in Layer 3.

    Implement in engine.py: relu, tanh, exp
"""

import math

from micrograd.engine import Value


def test_relu_passes_positive_through():
    assert Value(3.0).relu().data == 3.0


def test_relu_clamps_negative_to_zero():
    assert Value(-2.0).relu().data == 0.0


def test_relu_at_zero_is_zero():
    assert Value(0.0).relu().data == 0.0


def test_tanh_matches_math_tanh():
    assert abs(Value(0.5).tanh().data - math.tanh(0.5)) < 1e-9


def test_tanh_of_zero_is_zero():
    assert abs(Value(0.0).tanh().data - 0.0) < 1e-12


def test_tanh_saturates_toward_one():
    assert 0.99 < Value(5.0).tanh().data < 1.0


def test_exp_matches_math_exp():
    assert abs(Value(1.0).exp().data - math.e) < 1e-9


def test_exp_of_zero_is_one():
    assert abs(Value(0.0).exp().data - 1.0) < 1e-12
