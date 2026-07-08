"""micrograd.nn — a tiny neural-net library on top of the Value engine.

Same guided-stub deal as engine.py: you implement the bodies; the test ladder
(tests/test_layer7..11) drives you one concept at a time.

    Layer 7  test_layer7_neuron.py       Neuron: w·x + b, optional relu; parameters()/zero_grad()
    Layer 8  test_layer8_layer.py        Layer: a row of independent Neurons
    Layer 9  test_layer9_mlp.py          MLP: a stack of Layers (+ finite-diff gradcheck)
    Layer 10 test_layer10_train_moons.py train an MLP on 2D make_moons; loss falls, it separates
    Layer 11 test_layer11_torch_gradcheck.py  OPTIONAL: grads match torch.autograd (skips w/o torch)

Everything here is just Values wired together, so once the engine is correct,
backprop through a whole network is automatic — you never write another
_backward. The only new ideas are (a) collecting a model's trainable Values via
parameters(), and (b) zeroing their grads between steps (recall engine Layer 4:
grads accumulate).

Run `pytest -x` from the repo root and implement whatever the first failing
test asks for.
"""

import random

from micrograd.engine import Value


class Module:
    """Base class: something that owns a list of trainable Values."""

    def zero_grad(self):
        """Reset .grad to 0 on every parameter.

        Needed because grads ACCUMULATE across backward() calls (engine Layer
        4). If you don't clear them, each optimizer step mixes in gradients
        from previous steps. Use parameters().
        """
        for value in self.parameters():
            value.grad = 0.0

    def parameters(self):
        """Return this module's trainable Values. Base has none; subclasses
        override."""
        return []


class Neuron(Module):
    """A single neuron: optionally-nonlinear  f(w · x + b)."""

    def __init__(self, nin, nonlin=True):
        """`nin` inputs. Create one weight Value per input and a single bias
        Value, and remember whether this neuron is nonlinear.

        (A small random weight init and a zero bias is the usual choice.)
        """
        self.nonlin = nonlin
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        """Forward pass. `x` is a length-nin sequence (of Values or numbers).

        Compute the weighted sum of inputs plus the bias, then apply relu if
        this neuron is nonlinear, otherwise return the raw activation.
        """
        total = self.b
        for i, val in enumerate(x):
            total += self.w[i] * val

        if self.nonlin:
            total = total.relu()
        return total

    def parameters(self):
        """The weights and the bias, as one flat list."""
        return self.w + [self.b]


class Layer(Module):
    """A fully-connected layer: `nout` neurons that share the same inputs."""

    def __init__(self, nin, nout, **kwargs):
        """Build `nout` Neurons, each taking `nin` inputs. Pass **kwargs
        (e.g. nonlin) straight through to each Neuron."""
        self.nodes = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """Return the neurons' outputs. Convention: a list of `nout` Values,
        but just the single Value when nout == 1 (so stacked layers compose
        without special-casing)."""
        out = [n(x) for n in self.nodes]
        if len(out) == 1:
            return out[0]
        return out

    def parameters(self):
        """All parameters of all neurons, flattened."""
        all_params = []
        for n in self.nodes:
            all_params += n.parameters()
        return all_params


class MLP(Module):
    """A multi-layer perceptron: a stack of Layers.

    MLP(nin, [n1, n2, ..., nk]) has layer sizes nin -> n1 -> ... -> nk.
    """

    def __init__(self, nin, nouts):
        """Build the Layers from the size list. Convention: every layer is
        nonlinear EXCEPT the last, which is linear (so the output can take any
        real value)."""
        self.layers = []
        prev_layer_size = nin
        for i, size in enumerate(nouts):
            nonlin = i != (len(nouts) - 1)
            layer = Layer(prev_layer_size, size, nonlin=nonlin)
            self.layers.append(layer)
            prev_layer_size = size

    def __call__(self, x):
        """Run the input through each layer in turn and return the final
        output."""
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        """Every parameter in every layer, flattened."""
        all_params = []
        for l in self.layers:
            all_params += l.parameters()
        return all_params
