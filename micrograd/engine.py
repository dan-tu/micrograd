"""micrograd — a tiny scalar-valued autograd engine (guided-stub edition).

You implement this file from scratch. The test ladder in ../tests drives you
through it one concept at a time:

    Layer 0  test_layer0_value.py            the Value object exists (given)
    Layer 1  test_layer1_forward_arithmetic  + - * / ** and reverse ops (forward only)
    Layer 2  test_layer2_activations         relu, tanh, exp (forward only)
    Layer 3  test_layer3_backward_local      each op's LOCAL gradient + backward()
    Layer 4  test_layer4_grad_accumulation   reusing a Value: grads must ACCUMULATE
    Layer 5  test_layer5_full_backprop       chain rule through a whole graph
    Layer 6  test_layer6_numerical_gradcheck analytic grads == finite differences

Run `pytest -x` from the repo root and implement whatever the first failing
test asks for. Everything below is `raise NotImplementedError` except the two
pieces of pure bookkeeping (__init__ and __repr__), which are given so you can
focus on the math rather than the plumbing.

THE ONE BIG IDEA
----------------
Every operation does two things:
  1. FORWARD:  compute `out.data` from its inputs, and record which inputs
     produced it (so backward() can walk the graph).
  2. BACKWARD: define a closure `out._backward` that, ASSUMING `out.grad` is
     already known, adds this op's local gradient contribution onto each
     input's `.grad`. Always use `+=`, never `=` (see Layer 4 for why).
"""

import math


class Value:
    """A single scalar in the computation graph.

    Attributes
    ----------
    data : float
        The scalar value carried forward.
    grad : float
        dL/d(self), i.e. the derivative of the final output w.r.t. this node.
        Starts at 0.0 and is filled in by backward().
    _backward : callable
        Closure installed by whichever op produced this node. Pushes this
        node's grad onto its inputs' grads. No-op for leaf nodes.
    _prev : set[Value]
        The input Values that produced this node (the edges of the graph).
    _op : str
        Label of the producing op, for debugging / graph drawing.
    """

    # ---- Provided infrastructure: you should not need to change __init__ ----
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        # internal autograd state
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ======================================================================
    # Layer 1 — forward arithmetic (also wire up each op's _backward closure;
    # the backward wiring is what Layer 3 will exercise, so do it now)
    # ======================================================================
    def __add__(self, other):
        """self + other.

        FORWARD:  out.data = self.data + other.data
        BACKWARD: addition passes gradient through unchanged to both inputs:
                  self.grad  += out.grad
                  other.grad += out.grad

        Remember to coerce a raw int/float `other` into a Value first, and to
        record (self, other) as out's children so backward() can find them.
        """
        raise NotImplementedError

    def __mul__(self, other):
        """self * other.

        FORWARD:  out.data = self.data * other.data
        BACKWARD: d(a*b)/da = b, d(a*b)/db = a, so
                  self.grad  += other.data * out.grad
                  other.grad += self.data  * out.grad
        """
        raise NotImplementedError

    def __pow__(self, other):
        """self ** other, where `other` is a plain int or float (not a Value).

        FORWARD:  out.data = self.data ** other
        BACKWARD: d(x**n)/dx = n * x**(n-1), so
                  self.grad += (other * self.data ** (other - 1)) * out.grad
        """
        raise NotImplementedError

    # ======================================================================
    # Layer 2 — activations / nonlinearities
    # ======================================================================
    def relu(self):
        """Rectified linear unit: max(0, self).

        FORWARD:  out.data = self.data if self.data > 0 else 0
        BACKWARD: derivative is 1 where the input was positive, else 0:
                  self.grad += (out.data > 0) * out.grad
        """
        raise NotImplementedError

    def tanh(self):
        """Hyperbolic tangent (use math.tanh for the forward pass).

        FORWARD:  t = tanh(self.data);  out.data = t
        BACKWARD: d(tanh(x))/dx = 1 - tanh(x)**2, so
                  self.grad += (1 - t**2) * out.grad
        """
        raise NotImplementedError

    def exp(self):
        """Exponential (use math.exp for the forward pass).

        FORWARD:  e = exp(self.data);  out.data = e
        BACKWARD: d(exp(x))/dx = exp(x) = out.data, so
                  self.grad += e * out.grad
        """
        raise NotImplementedError

    # ======================================================================
    # Layers 3-5 — reverse-mode autodiff over the whole graph
    # ======================================================================
    def backward(self):
        """Compute gradients of `self` w.r.t. every node it depends on.

        Steps:
          1. Build a topological ordering of all nodes reachable from `self`
             via `_prev` (each node appears AFTER all of its inputs).
          2. Seed the output: self.grad = 1.0  (d(self)/d(self) == 1).
          3. Walk the nodes in REVERSE topological order, calling each node's
             `_backward()` so gradient flows from outputs back to inputs.

        (A recursive DFS with a `visited` set is the usual way to build the
        topological order.)
        """
        raise NotImplementedError

    # ======================================================================
    # Layer 1 — convenience / reverse-dispatch ops.
    # Each of these is a one-liner built out of the primitives above; none of
    # them needs its own _backward. Implement them once the primitives work.
    # ======================================================================
    def __neg__(self):
        """-self. Hint: multiply by -1."""
        raise NotImplementedError

    def __radd__(self, other):
        """other + self (Python falls back here when the left operand is a
        raw number). Hint: addition commutes."""
        raise NotImplementedError

    def __sub__(self, other):
        """self - other. Hint: self + (-other)."""
        raise NotImplementedError

    def __rsub__(self, other):
        """other - self. Hint: other + (-self)."""
        raise NotImplementedError

    def __rmul__(self, other):
        """other * self. Hint: multiplication commutes."""
        raise NotImplementedError

    def __truediv__(self, other):
        """self / other. Hint: self * other**-1 (reuse __pow__)."""
        raise NotImplementedError

    def __rtruediv__(self, other):
        """other / self. Hint: other * self**-1."""
        raise NotImplementedError

    # ---- Provided: string form, handy when debugging ----
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
