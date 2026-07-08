"""Layer 11 — OPTIONAL: gradients match PyTorch's autograd.

The study plan's self-test says "gradients match torch.autograd to ~1e-6".
Layer 9 already proved correctness against finite differences with no
dependencies; this layer does the literal torch comparison for those who have
it installed. The whole module is skipped if torch is missing.

    pip install torch    # to enable this layer
"""

import random

import pytest

torch = pytest.importorskip("torch")

from micrograd.engine import Value
from micrograd.nn import MLP


def test_mlp_grads_match_torch_autograd():
    random.seed(0)
    model = MLP(2, [4, 4, 1])
    xrow = [0.5, -1.2]

    # --- micrograd forward + backward ---
    out_m = model([Value(xrow[0]), Value(xrow[1])])
    out_m.backward()

    # --- identical net in torch, sharing the SAME weight values ---
    tmap = {}                                  # id(param Value) -> torch leaf tensor

    def T(v):
        t = torch.tensor(float(v.data), dtype=torch.double, requires_grad=True)
        tmap[id(v)] = t
        return t

    x = [torch.tensor(float(v), dtype=torch.double) for v in xrow]
    for layer in model.layers:
        new_x = []
        for neuron in layer.neurons:
            act = sum((T(wi) * xi for wi, xi in zip(neuron.w, x)), T(neuron.b))
            new_x.append(act.relu() if neuron.nonlin else act)
        x = new_x
    out_t = x[0]
    out_t.backward()

    # --- compare forward values and every parameter gradient ---
    assert abs(out_m.data - out_t.item()) < 1e-9
    for p in model.parameters():
        assert abs(p.grad - tmap[id(p)].grad.item()) < 1e-6
