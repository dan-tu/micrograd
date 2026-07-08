"""Layer 8 — the Layer: a row of independent neurons over shared inputs.

    Implement in nn.py: Layer.__init__/__call__/parameters
"""

from micrograd.engine import Value
from micrograd.nn import Layer


def test_layer_with_many_outputs_returns_a_list():
    layer = Layer(3, 4)
    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, list)
    assert len(out) == 4
    assert all(isinstance(o, Value) for o in out)


def test_layer_with_single_output_returns_a_bare_value():
    # The nout == 1 special case, so stacked layers compose without unwrapping.
    layer = Layer(3, 1)
    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, Value)


def test_layer_parameter_count():
    # nout neurons, each with nin weights + 1 bias
    layer = Layer(3, 4)
    assert len(layer.parameters()) == 4 * (3 + 1)


def test_layer_neurons_are_independent():
    # Different neurons should hold different weight objects (not a shared one).
    layer = Layer(2, 3)
    params = layer.parameters()
    assert len({id(p) for p in params}) == len(params)
