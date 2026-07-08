"""Layer 4 — gradient ACCUMULATION when a Value is reused.

This is the single most common micrograd bug. If a node feeds into the graph
along more than one path, its gradient is the SUM of the contributions from
each path. That only works if every `_backward` closure does

        self.grad += ...        # correct

rather than

        self.grad = ...         # WRONG: later paths clobber earlier ones

If you wrote `=` anywhere, these tests are where it shows up. Go back and make
every closure use `+=`.
"""

from micrograd.engine import Value


def test_value_added_to_itself():
    a = Value(3.0)
    b = a + a          # b = 2a  ->  db/da = 2
    b.backward()
    assert a.grad == 2.0


def test_value_multiplied_by_itself():
    a = Value(3.0)
    b = a * a          # b = a^2 ->  db/da = 2a = 6
    b.backward()
    assert a.grad == 6.0


def test_value_reused_across_two_subexpressions():
    x = Value(4.0)
    y = x * 2          # dy/dx = 2
    z = x * 3          # dz/dx = 3
    out = y + z        # dout/dx = 2 + 3 = 5
    out.backward()
    assert x.grad == 5.0


def test_diamond_graph():
    # f = (a*b) * (a+b), with a reused on both branches.
    a, b = Value(-2.0), Value(3.0)
    d = a * b          # d = -6
    e = a + b          # e =  1
    f = d * e
    f.backward()
    # df/da = b*e + d*1 = 3*1 + (-6) = -3
    # df/db = a*e + d*1 = -2*1 + (-6) = -8
    assert a.grad == -3.0
    assert b.grad == -8.0
