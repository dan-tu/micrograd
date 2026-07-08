"""Layer 7 — the Neuron, plus parameters() / zero_grad().

A Neuron is just Values wired together, so its backward pass comes free from
the engine. What's new: exposing a module's trainable Values via parameters(),
and resetting their grads with zero_grad().

    Implement in nn.py: Module.zero_grad, Neuron.__init__/__call__/parameters
"""

from micrograd.engine import Value
from micrograd.nn import Neuron


def test_neuron_has_one_weight_per_input_plus_bias():
    n = Neuron(3)
    assert len(n.parameters()) == 4          # 3 weights + 1 bias


def test_neuron_parameters_are_values():
    n = Neuron(2)
    assert all(isinstance(p, Value) for p in n.parameters())


def test_neuron_forward_returns_single_value():
    n = Neuron(2)
    out = n([1.0, -2.0])
    assert isinstance(out, Value)


def test_linear_neuron_computes_weighted_sum_plus_bias():
    # A non-nonlinear neuron is just w·x + b; pin its params and check the math.
    n = Neuron(2, nonlin=False)
    n.w[0].data, n.w[1].data, n.b.data = 2.0, -3.0, 0.5
    out = n([1.0, 1.0])
    assert abs(out.data - (2.0 * 1.0 + -3.0 * 1.0 + 0.5)) < 1e-9   # = -0.5


def test_nonlinear_neuron_applies_relu():
    n = Neuron(2, nonlin=True)
    n.w[0].data, n.w[1].data, n.b.data = 2.0, -3.0, 0.5
    out = n([1.0, 1.0])          # pre-activation is -0.5 -> relu -> 0
    assert out.data == 0.0


def test_zero_grad_resets_parameter_grads():
    n = Neuron(2, nonlin=False)
    out = n([1.0, 1.0])
    out.backward()
    assert any(p.grad != 0 for p in n.parameters())   # grads exist after backward
    n.zero_grad()
    assert all(p.grad == 0 for p in n.parameters())    # ...and are cleared


def test_zero_grad_prevents_accumulation_across_backwards():
    # Two backward passes WITHOUT zeroing would double the grads (engine L4).
    # zero_grad in between must make the second pass match a single pass.
    n = Neuron(2, nonlin=False)
    x = [1.0, 1.0]
    n([*x]).backward()
    once = [p.grad for p in n.parameters()]
    n.zero_grad()
    n([*x]).backward()
    twice = [p.grad for p in n.parameters()]
    assert all(abs(a - b) < 1e-12 for a, b in zip(once, twice))
