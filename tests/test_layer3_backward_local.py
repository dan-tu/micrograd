"""Layer 3 — backward() and each op's LOCAL gradient.

Now the closures you wrote in Layers 1-2 have to be correct. For a single op
`out = f(a, b)`, calling `out.backward()` should:
  - set out.grad = 1.0
  - set a.grad and b.grad to the local partial derivatives.

    Implement in engine.py: backward() (topological sort + reverse pass),
    plus make sure every _backward closure from Layers 1-2 is correct.

Each test below builds ONE operation and checks the analytically-known
derivative. Fresh Value objects per test means grads start at 0.
"""

import math

from micrograd.engine import Value


def test_backward_seeds_output_grad_to_one():
    a = Value(3.0)
    out = a * Value(2.0)
    out.backward()
    assert out.grad == 1.0


def test_add_backward():
    a, b = Value(3.0), Value(4.0)
    out = a + b
    out.backward()
    # d(a+b)/da = 1, d(a+b)/db = 1
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_mul_backward():
    a, b = Value(3.0), Value(4.0)
    out = a * b
    out.backward()
    # d(a*b)/da = b, d(a*b)/db = a
    assert a.grad == 4.0
    assert b.grad == 3.0


def test_pow_backward_square():
    a = Value(3.0)
    out = a ** 2
    out.backward()
    # d(a^2)/da = 2a = 6
    assert a.grad == 6.0


def test_pow_backward_cube():
    a = Value(2.0)
    out = a ** 3
    out.backward()
    # d(a^3)/da = 3a^2 = 12
    assert a.grad == 12.0


def test_sub_backward():
    a, b = Value(5.0), Value(2.0)
    out = a - b
    out.backward()
    assert a.grad == 1.0
    assert b.grad == -1.0


def test_div_backward():
    a, b = Value(6.0), Value(2.0)
    out = a / b
    out.backward()
    # d(a/b)/da = 1/b = 0.5 ; d(a/b)/db = -a/b^2 = -1.5
    assert abs(a.grad - 0.5) < 1e-9
    assert abs(b.grad - (-1.5)) < 1e-9


def test_relu_backward_positive():
    a = Value(2.0)
    out = a.relu()
    out.backward()
    assert a.grad == 1.0


def test_relu_backward_negative():
    a = Value(-2.0)
    out = a.relu()
    out.backward()
    assert a.grad == 0.0


def test_tanh_backward():
    a = Value(0.5)
    out = a.tanh()
    out.backward()
    # d(tanh(x))/dx = 1 - tanh(x)^2
    assert abs(a.grad - (1 - math.tanh(0.5) ** 2)) < 1e-9


def test_exp_backward():
    a = Value(1.0)
    out = a.exp()
    out.backward()
    # d(exp(x))/dx = exp(x)
    assert abs(a.grad - math.e) < 1e-9
