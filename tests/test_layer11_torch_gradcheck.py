"""Layer 11 — OPTIONAL: gradients match PyTorch's autograd.

The study plan's self-test says "gradients match torch.autograd to ~1e-6".
Layer 9 already proved correctness against finite differences with no
dependencies; this layer does the literal torch comparison for those who have
it installed. The whole module is SKIPPED if torch is missing, so if this test
"won't run", that's why — install torch to enable it:

    pip install torch          # or:  uv pip install torch

The torch mirror below rebuilds your network from the public interface only
(`model.layers`, `layer.parameters()`, and the MLP "last layer is linear"
convention) so it doesn't depend on what you named the neuron list inside a
Layer.
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

    # --- the same net in torch, sharing the SAME weight values ---
    tmap = {}                                   # id(param Value) -> torch leaf tensor

    def T(v):
        t = torch.tensor(float(v.data), dtype=torch.double, requires_grad=True)
        tmap[id(v)] = t
        return t

    x = [torch.tensor(float(v), dtype=torch.double) for v in xrow]
    n_layers = len(model.layers)
    for li, layer in enumerate(model.layers):
        params = layer.parameters()             # per neuron: [w0..w_{nin-1}, b]
        nin = len(x)
        per_neuron = nin + 1
        nout = len(params) // per_neuron
        is_last = li == n_layers - 1
        new_x = []
        for j in range(nout):
            chunk = params[j * per_neuron:(j + 1) * per_neuron]
            weights, bias = chunk[:nin], chunk[nin]
            act = sum((T(w) * xi for w, xi in zip(weights, x)), T(bias))
            new_x.append(act if is_last else act.relu())
        x = new_x
    out_t = x[0]
    out_t.backward()

    # --- compare forward values and every parameter gradient ---
    assert abs(out_m.data - out_t.item()) < 1e-9
    for p in model.parameters():
        assert abs(p.grad - tmap[id(p)].grad.item()) < 1e-6
