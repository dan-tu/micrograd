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

    Returns
    -------
    (model, history) where `history` is a dict containing at least:
        "first_loss": float   # dataset loss BEFORE any update
        "final_loss": float   # dataset loss after training
        "accuracy":   float   # fraction of points classified correctly, in [0, 1]

    The gate asserts  history["final_loss"] < history["first_loss"]  and
    history["accuracy"] > 0.80  (a working training run clears that easily).
    """
    raise NotImplementedError


if __name__ == "__main__":
    model, history = train_moons()
    print(history)
