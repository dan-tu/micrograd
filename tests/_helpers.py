"""Shared test helpers for the nn ladder.

Kept dependency-free on purpose. The leading underscore keeps pytest from
collecting this as a test module. (The two-moons dataset lives in
micrograd/data.py so the training script can import it too.)
"""


def param_gradcheck(model, loss_fn, tol=1e-4, h=1e-6):
    """Verify model parameter grads against central finite differences.

    `loss_fn(model)` must build and return a scalar-valued Value using the
    model's current parameters. Runs one analytic backward, then perturbs each
    parameter's .data to estimate the same gradient numerically and asserts
    they agree. This is the dependency-free version of "match torch.autograd".
    """
    loss = loss_fn(model)
    model.zero_grad()
    loss.backward()
    analytic = [p.grad for p in model.parameters()]

    for i, p in enumerate(model.parameters()):
        original = p.data
        p.data = original + h
        f_plus = loss_fn(model).data
        p.data = original - h
        f_minus = loss_fn(model).data
        p.data = original
        numeric = (f_plus - f_minus) / (2 * h)
        assert abs(numeric - analytic[i]) < tol, (
            f"param {i}: numeric={numeric:.8f} analytic={analytic[i]:.8f}"
        )
