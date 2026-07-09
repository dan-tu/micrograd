"""Train an MLP on two-moons — YOUR training loop goes here.

This is the second half of the study plan's Layer 0 milestone. You've built the
autograd engine and the nn library; now write the loop that actually trains a
network with them. The gate is tests/test_layer10_train_moons.py, which imports
train_moons(), runs it, and checks the returned history.

Pieces already on hand:
    make_moons(n_samples, noise, seed) -> (X, y)   # micrograd/data.py
        X: list of [x1, x2] points   y: list of 0/1 labels
    MLP(nin, [h1, ..., 1])                          # micrograd/nn.py
        .parameters(), .zero_grad(), model(x) -> Value

Run it directly to watch it learn:
    python train_moons.py
"""

from micrograd.data import make_moons
from micrograd.nn import MLP


def train_moons(seed=1):
    """Build an MLP and train it to classify the two-moons dataset.

    You design the whole optimization: the loss function, the parameter-update
    rule, the learning-rate schedule, the number of steps, and the network
    shape. Seed the RNG at the start so the run is reproducible.

    Scope — keep it minimal (this proves the engine can train, nothing more;
    it mirrors Karpathy's micrograd demo):
      * Full-batch: each step, run ALL the points through the model, sum the
        loss, one backward(), one update. No minibatches, and no
        train/val/test split -- train and score on the same data. (That
        discipline is the next study-plan milestone, on MNIST.)
      * A FIXED number of steps (~50-100), not "loop until accuracy". With
        full-batch, one step == one epoch over the data. accuracy > 0.80 is an
        end-of-run gate check, not a stopping condition.
      * Keep the net small -- scalar autograd is slow (a Value per operation).
        [8] or [16] hidden units, not hundreds.

    Returns
    -------
    (model, history) where `history` is a dict containing at least:
        "first_loss": float   # dataset loss BEFORE any update
        "final_loss": float   # dataset loss after training
        "accuracy":   float   # fraction of points classified correctly, in [0, 1]

    The gate asserts  history["final_loss"] < history["first_loss"]  and
    history["accuracy"] > 0.80  (a working training run clears that easily).
    """
    N_STEPS = 50
    ALPHA = 0.01

    X, y = make_moons()
    model = MLP(len(X[0]), [16, 32, 1])
    history = {}

    for n in range(N_STEPS):
        loss = 0.0
        correct = 0
        for x_i, y_i in zip(X, y):
            out = model(x_i)
            loss = loss + (y_i - out)**2
            y_hat = out.data >= 0.5
            if y_hat == y_i:
                correct += 1
        accuracy = correct / len(y)

        # Use MSE for loss function
        loss = loss / len(y)
        loss.backward()

        # Update weights
        for param in model.parameters():
            param.data = param.data - param.grad * ALPHA
        model.zero_grad()

        # Report some stats
        print(f"[Step {n}] loss = {loss}, accuracy = {accuracy}")
        if n == 0:
            history["first_loss"] = loss.data
        history["accuracy"] = accuracy
        history["final_loss"] = loss.data
    return (model, history)


if __name__ == "__main__":
    model, history = train_moons()
    print(history)
