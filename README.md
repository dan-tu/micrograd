# micrograd — from scratch

Reimplement Karpathy's [micrograd](https://github.com/karpathy/micrograd) scalar
autograd engine from a blank file, driven by a ladder of tests. This is the
**Layer 0** build milestone for the World Models study plan: *"reimplement
micrograd from a blank file; gradients match `torch.autograd`."*

You implement everything in `micrograd/engine.py`. The tests define the API and
the expected behavior. Work through the layers in order — each one unlocks the
next.

## Setup

```bash
pip install -r requirements.txt
```

## The workflow

Run the ladder and let the first failing test tell you what to build next:

```bash
pytest -x        # stop at the first failure
```

`-x` is the whole game: implement the one thing the failing test asks for, rerun,
repeat. When everything is green, the engine is done. To see just one layer:

```bash
pytest tests/test_layer3_backward_local.py
```

## The layers

| Layer | File | What you implement |
|------:|------|--------------------|
| 0 | `test_layer0_value.py` | *(given)* — smoke test that `Value` imports and constructs |
| 1 | `test_layer1_forward_arithmetic.py` | `__add__`, `__mul__`, `__pow__` + reverse ops (`-`, `/`, `neg`, r-ops) — forward values only |
| 2 | `test_layer2_activations.py` | `relu`, `tanh`, `exp` — forward values only |
| 3 | `test_layer3_backward_local.py` | `backward()` (topo sort + reverse pass) and each op's local gradient |
| 4 | `test_layer4_grad_accumulation.py` | gradient **accumulation** when a `Value` is reused (`+=`, not `=`) |
| 5 | `test_layer5_full_backprop.py` | *(no new code)* — chain rule through a whole graph; grads match PyTorch |
| 6 | `test_layer6_numerical_gradcheck.py` | *(no new code)* — analytic grads vs. finite differences on random graphs |

Layers 1 and 2 only check `.data`, so a wrong gradient closure slips through
until **Layer 3**, where `backward()` first exercises it. Layer 4 is where a
`self.grad = ...` (instead of `+=`) bug surfaces. Layers 5 and 6 add no new
methods — they're end-to-end proofs that the whole engine is correct.

## What's given vs. what's yours

`engine.py` provides only `__init__` and `__repr__` (pure graph bookkeeping) so
you can focus on the math. Every operation and `backward()` itself is a
`raise NotImplementedError` with a docstring describing the forward formula and
the local gradient. Nothing else is filled in.

## The gate

The study plan's rule: **if you can't reproduce autograd's gradients here, don't
advance** — everything downstream debugs through this. Layer 5 checks against
values PyTorch produced (138.8338, 645.5773); Layer 6 checks against finite
differences. Green on both == you've earned Layer 1.

## Next step (not in this ladder)

Once the engine is green, the full self-test adds a neural-net layer on top:
`Neuron` / `Layer` / `MLP` in `nn.py`, trained on the 2D `make_moons` dataset,
with gradients cross-checked against `torch.autograd`. Ask and I'll scaffold
that ladder too.
