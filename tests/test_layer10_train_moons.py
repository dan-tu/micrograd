"""Layer 10 — train an MLP on the two-moons dataset (study-plan Layer 0 milestone).

No new library code here: YOU write the training loop in ../train_moons.py.
This test is only the gate — it runs your loop and checks the model actually
learned. See train_moons.py for the contract it expects.
"""

from train_moons import train_moons


def test_training_reduces_loss_and_separates_moons():
    model, history = train_moons()
    assert history["final_loss"] < history["first_loss"]    # optimization worked
    assert history["accuracy"] > 0.80                        # learned a real boundary
