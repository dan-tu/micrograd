"""Layer 6 — numerical gradient check (the capstone).

The study-plan self-test says "gradients match torch.autograd to ~1e-6". This
layer does the dependency-free version of that idea: it compares your analytic
gradients against a central finite-difference estimate

        df/dx  ~=  ( f(x + h) - f(x - h) ) / (2h)

on random expressions built from every op in the engine. If your engine agrees
with finite differences on arbitrary graphs, backprop is correct.

(When you later add PyTorch to the project, the exact same expressions can be
checked against `torch.autograd` — this is the same test, no external deps.)

    Implement in engine.py: nothing new — this cross-checks the whole engine.
"""

import random

from micrograd.engine import Value


def _gradcheck(build, n_inputs, seed, tol=1e-4, h=1e-6):
    """Compare analytic grads (backward) to central differences.

    `build(xs)` takes a list of leaf Values and returns a scalar Value.
    """
    rng = random.Random(seed)
    xs = [Value(rng.uniform(-2.0, 2.0)) for _ in range(n_inputs)]

    out = build(xs)
    out.backward()
    analytic = [x.grad for x in xs]

    for i, x in enumerate(xs):
        original = x.data
        x.data = original + h
        f_plus = build(xs).data
        x.data = original - h
        f_minus = build(xs).data
        x.data = original
        numeric = (f_plus - f_minus) / (2 * h)
        assert abs(numeric - analytic[i]) < tol, (
            f"input {i}: numeric={numeric:.8f} analytic={analytic[i]:.8f}"
        )


def test_gradcheck_polynomial():
    _gradcheck(lambda xs: xs[0] * xs[1] + xs[0] ** 3 - xs[1], n_inputs=2, seed=1)


def test_gradcheck_division_and_neg():
    _gradcheck(lambda xs: (xs[0] - xs[1]) / (xs[0] ** 2 + 1.0), n_inputs=2, seed=2)


def test_gradcheck_with_tanh():
    _gradcheck(lambda xs: (xs[0] * xs[1] + xs[2]).tanh(), n_inputs=3, seed=3)


def test_gradcheck_with_relu_and_exp():
    _gradcheck(lambda xs: (xs[0].relu() + xs[1].exp()) * xs[2], n_inputs=3, seed=4)


def test_gradcheck_deep_chain():
    def build(xs):
        v = xs[0]
        for x in xs[1:]:
            v = (v + x) * x
        return v.tanh()

    _gradcheck(build, n_inputs=4, seed=5)


def test_gradcheck_activation_mid_graph():
    # tanh/exp feeding further ops, so their out.grad != 1 -- guards against a
    # backward closure that forgets the incoming-gradient factor.
    _gradcheck(lambda xs: xs[0].tanh() * xs[1] + xs[2], n_inputs=3, seed=6)
    _gradcheck(lambda xs: (xs[0].exp() + xs[1]) * xs[2].tanh(), n_inputs=3, seed=7)
