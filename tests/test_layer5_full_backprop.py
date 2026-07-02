"""Layer 5 — full backprop through a deep, tangled graph.

If Layers 0-4 pass, the engine should already handle this — these are
integration tests over graphs too big to differentiate by eye.

`test_karpathy_sanity_check` is the canonical example from the micrograd
README; the expected gradients (138.8338, 645.5773) were produced by PyTorch's
autograd, so matching them means your engine agrees with the real thing.

    Implement in engine.py: nothing new — this verifies everything together.
"""

import math

from micrograd.engine import Value


def test_single_neuron_matches_hand_derivation():
    # A single tanh neuron: o = tanh(x*w + b), chosen so the math is clean.
    x, w, b = Value(2.0), Value(-3.0), Value(6.0)
    n = x * w + b            # = 2*-3 + 6 = 0
    o = n.tanh()             # tanh(0) = 0
    o.backward()
    # do/dn = 1 - tanh(0)^2 = 1
    # dn/dx = w = -3 ; dn/dw = x = 2 ; dn/db = 1
    assert abs(o.data - 0.0) < 1e-12
    assert abs(x.grad - (-3.0)) < 1e-9
    assert abs(w.grad - 2.0) < 1e-9
    assert abs(b.grad - 1.0) < 1e-9


def test_karpathy_sanity_check():
    # Verbatim from the micrograd README. Expected values are PyTorch's.
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g += 10.0 / f

    assert abs(g.data - 24.7041) < 1e-3      # forward pass
    g.backward()
    assert abs(a.grad - 138.8338) < 1e-3     # dg/da (matches torch.autograd)
    assert abs(b.grad - 645.5773) < 1e-3     # dg/db (matches torch.autograd)


def test_exp_tanh_composition():
    # A graph mixing exp and tanh, checked against a direct scalar computation.
    x = Value(0.7)
    y = (x.exp() + x * x).tanh()
    y.backward()

    xv = 0.7
    inner = math.exp(xv) + xv * xv
    yv = math.tanh(inner)
    # dy/dx = (1 - tanh(inner)^2) * (exp(x) + 2x)
    dydx = (1 - math.tanh(inner) ** 2) * (math.exp(xv) + 2 * xv)
    assert abs(y.data - yv) < 1e-9
    assert abs(x.grad - dydx) < 1e-9
