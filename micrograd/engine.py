"""micrograd — a tiny scalar-valued autograd engine (guided-stub edition).

You implement this file from scratch. The test ladder in ../tests drives you
through it one concept at a time:

    Layer 0  test_layer0_value.py            the Value object exists (given)
    Layer 1  test_layer1_forward_arithmetic  + - * / ** and reverse ops (forward only)
    Layer 2  test_layer2_activations         relu, tanh, exp (forward only)
    Layer 3  test_layer3_backward_local      each op's LOCAL gradient + backward()
    Layer 4  test_layer4_grad_accumulation   reusing a Value in more than one place
    Layer 5  test_layer5_full_backprop       chain rule through a whole graph
    Layer 6  test_layer6_numerical_gradcheck analytic grads == finite differences

Run `pytest -x` from the repo root and implement whatever the first failing
test asks for. The operations below are yours to write; only __init__ and
__repr__ (pure bookkeeping) are given, so you can focus on the math.

THE SHAPE OF AN OP
------------------
Every operation does two things:
  1. FORWARD:  compute out.data from its inputs, and record which inputs
     produced it (so backward() can later walk the graph).
  2. BACKWARD: install a closure out._backward that, given out.grad, sends
     gradient back to each input. What that closure does is different per op —
     that's the part worth deriving yourself.
"""

import math


class Value:
    """A single scalar in the computation graph.

    Attributes
    ----------
    data : float
        The scalar value carried forward.
    grad : float
        dL/d(self), the derivative of the final output w.r.t. this node.
        Starts at 0.0; filled in by backward().
    _backward : callable
        Closure installed by the op that produced this node; run during
        backward(). No-op for leaf nodes.
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
    # Layer 1 — forward arithmetic. Build the output Value now, and while the
    # inputs are in scope, wire up its _backward closure too (Layer 3 checks it).
    # ======================================================================
    def __add__(self, other):
        """self + other.

        Coerce a raw int/float into a Value first. Then think about how a sum
        should route the output's gradient back to each input.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out


    def __mul__(self, other):
        """self * other.

        The backward pass is the product rule — each input's local gradient
        depends on the other input.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """self ** other, where `other` is a plain int or float (not a Value).

        Backward is the power rule.
        """
        assert isinstance(other, (int, float)), "power is only compatible with ints or floats"
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out


    # ======================================================================
    # Layer 2 — activations / nonlinearities
    # ======================================================================
    def relu(self):
        """Rectified linear unit: max(0, self).

        Backward: consider the slope on each side of zero.
        """
        out = Value(0 if self.data < 0 else self.data, (self,), f"ReLU({self.data})")

        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        """Hyperbolic tangent (math.tanh for the forward value).

        Backward: the derivative of tanh is expressible in terms of tanh itself.
        """
        out = Value(math.tanh(self.data), (self,), f"tanh({self.data})")

        def _backward():
            self.grad += (1 - math.tanh(self.data)**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        """Exponential (math.exp for the forward value).

        Backward: exp is its own derivative.
        """
        out = Value(math.exp(self.data), (self,), f'exp({self.data})')

        def _backward():
            self.grad += math.exp(self.data) * out.grad
        out._backward = _backward

        return out

    # ======================================================================
    # Layers 3-5 — reverse-mode autodiff over the whole graph
    # ======================================================================
    def backward(self):
        """Fill in .grad for every node that `self` depends on.

        Sketch: build a topological ordering of the nodes reachable through
        _prev, seed the output node's gradient, then run each node's _backward
        in an order that respects the dependencies. (A DFS with a visited set
        builds the ordering.)
        """
        deps = []
        seen = set()
        def visit(v):
            if v not in seen: 
                seen.add(v)
                for child in v._prev:
                    visit(child)
                deps.append(v)
        visit(self)

        self.grad = 1
        for v in reversed(deps):
            v._backward()

    # ======================================================================
    # Layer 1 — convenience / reverse-dispatch ops. Each is a one-liner built
    # from the primitives above; none needs its own _backward.
    # ======================================================================
    def __neg__(self):
        """-self."""
        return self * -1

    def __radd__(self, other):
        """other + self (Python dispatches here when the left operand is a
        raw number)."""
        return self + other

    def __sub__(self, other):
        """self - other."""
        return self + (-other)

    def __rsub__(self, other):
        """other - self."""
        return other + (-self)

    def __rmul__(self, other):
        """other * self."""
        return self * other

    def __truediv__(self, other):
        """self / other."""
        return self * other**-1

    def __rtruediv__(self, other):
        """other / self."""
        return other * self**-1

    # ---- Provided: string form, handy when debugging ----
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op={self._op})"
