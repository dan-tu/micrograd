"""Layer 9 — the MLP: a stack of Layers.

The last test is the real gate: gradients of every parameter checked against
finite differences (the dependency-free form of "match torch.autograd").

    Implement in nn.py: MLP.__init__/__call__/parameters
"""

import random

from micrograd.engine import Value
from micrograd.nn import MLP

from _helpers import param_gradcheck


def test_mlp_builds_the_right_layer_sizes():
    model = MLP(2, [4, 4, 1])
    assert len(model.layers) == 3


def test_mlp_single_output_is_a_value():
    model = MLP(3, [4, 1])
    out = model([1.0, 2.0, 3.0])
    assert isinstance(out, Value)


def test_mlp_multi_output_is_a_list():
    model = MLP(3, [4, 2])
    out = model([1.0, 2.0, 3.0])
    assert isinstance(out, list) and len(out) == 2


def test_mlp_parameter_count():
    # sizes 2 -> 4 -> 4 -> 1 : (2+1)*4 + (4+1)*4 + (4+1)*1 = 12 + 20 + 5 = 37
    model = MLP(2, [4, 4, 1])
    assert len(model.parameters()) == 37


def test_mlp_last_layer_is_linear():
    # The final layer must be linear (not relu), so the output can be negative.
    # MLP(1, [1]) is a single last-layer neuron; pin it to identity and feed a
    # negative input: a linear head returns it unchanged, a relu head clips to 0.
    model = MLP(1, [1])
    w, b = model.parameters()
    w.data, b.data = 1.0, 0.0
    assert model([-5.0]).data == -5.0


def test_mlp_gradients_match_finite_differences():
    # THE GATE: analytic backprop through the whole net vs. numeric gradients.
    random.seed(0)
    model = MLP(2, [4, 4, 1])
    points = [[0.3, -0.7], [-0.4, 0.9]]

    def loss_fn(m):
        total = Value(0.0)
        for row in points:
            total = total + m([Value(row[0]), Value(row[1])])
        return total

    param_gradcheck(model, loss_fn)   # asserts internally to ~1e-4
