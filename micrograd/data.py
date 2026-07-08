"""Provided helper: a small 2D dataset for the training milestone.

Pure Python (no numpy/sklearn) so the ladder stays dependency-free.
"""

import math
import random


def make_moons(n_samples=100, noise=0.1, seed=0):
    """A pure-Python version of sklearn's two-moons dataset.

    Returns (X, y) where X is a list of [x1, x2] points and y is a list of
    0/1 labels — two interleaving half-circles with Gaussian noise.
    """
    rng = random.Random(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    X, y = [], []
    for i in range(n_out):                       # outer (upper) moon -> label 0
        t = math.pi * i / max(n_out - 1, 1)
        X.append([math.cos(t) + rng.gauss(0, noise),
                  math.sin(t) + rng.gauss(0, noise)])
        y.append(0)
    for i in range(n_in):                        # inner (lower) moon -> label 1
        t = math.pi * i / max(n_in - 1, 1)
        X.append([1 - math.cos(t) + rng.gauss(0, noise),
                  0.5 - math.sin(t) + rng.gauss(0, noise)])
        y.append(1)
    return X, y
