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
pytest tests/test_layer03_backward_local.py
```

(Files are numbered with a leading zero — `test_layer00` … `test_layer11` — so
pytest collects them in true numeric order.)

## The layers

Part A builds the autograd **engine** (`micrograd/engine.py`):

| Layer | File | What you implement |
|------:|------|--------------------|
| 0 | `test_layer00_value.py` | *(given)* — smoke test that `Value` imports and constructs |
| 1 | `test_layer01_forward_arithmetic.py` | `__add__`, `__mul__`, `__pow__` + reverse ops (`-`, `/`, `neg`, r-ops) — forward values only |
| 2 | `test_layer02_activations.py` | `relu`, `tanh`, `exp` — forward values only |
| 3 | `test_layer03_backward_local.py` | `backward()` (topo sort + reverse pass) and each op's local gradient |
| 4 | `test_layer04_grad_accumulation.py` | gradient **accumulation** when a `Value` is reused (`+=`, not `=`) |
| 5 | `test_layer05_full_backprop.py` | *(no new code)* — chain rule through a whole graph; grads match PyTorch |
| 6 | `test_layer06_numerical_gradcheck.py` | *(no new code)* — analytic grads vs. finite differences on random graphs |

Layers 1 and 2 only check `.data`, so a wrong gradient closure slips through
until **Layer 3**, where `backward()` first exercises it. Layer 4 is where a
`self.grad = ...` (instead of `+=`) bug surfaces. Layers 5 and 6 add no new
methods — they're end-to-end proofs that the whole engine is correct.

Part B builds the neural-net library on top (`micrograd/nn.py`):

| Layer | File | What you implement |
|------:|------|--------------------|
| 7 | `test_layer07_neuron.py` | `Neuron` (`w·x + b`, optional relu) + `parameters()` / `zero_grad()` |
| 8 | `test_layer08_layer.py` | `Layer` — a row of independent neurons over shared inputs |
| 9 | `test_layer09_mlp.py` | `MLP` — a stack of layers; **gate:** grads vs. finite differences |
| 10 | `test_layer10_train_moons.py` + `train_moons.py` | **you write the training loop** — train the MLP on 2D `make_moons` until loss falls and it separates (>80% acc) |
| 11 | `test_layer11_torch_gradcheck.py` | *(no new code, optional)* — exact match to `torch.autograd`; skips if torch absent |

Layer 10 is different from the other "no new library code" layers: the loop
itself (loss, SGD update, lr schedule) is yours to write in `train_moons.py`.
`make_moons` is provided in `micrograd/data.py`; the test just runs your
`train_moons()` and checks the returned history. Run `python train_moons.py` to
watch it learn.

Part B is all Values wired together, so you never write another `_backward` —
backprop through a network is automatic once the engine is right. The only new
ideas are collecting `parameters()` and calling `zero_grad()` between steps
(the same accumulation lesson from Layer 4). Layer 9's finite-difference
gradcheck is the real gate; Layer 11 is the literal "match `torch.autograd`"
check for anyone who has torch installed.

## What's given vs. what's yours

`engine.py` and `nn.py` provide only pure bookkeeping — `Value.__init__` /
`__repr__` and the `Module` base's `parameters()` default. Every operation,
`backward()`, and every `Neuron` / `Layer` / `MLP` method is a
`raise NotImplementedError` with a short docstring pointing at the idea (which
rule applies, what the method returns) rather than the code. Nothing else is
filled in.

## The gate

The study plan's rule: **if you can't reproduce autograd's gradients here, don't
advance** — everything downstream debugs through this. Layer 5 checks the engine
against values PyTorch produced (138.8338, 645.5773); Layer 6 and Layer 9 check
against finite differences; Layer 10 shows the whole stack actually learning.
Green through Layer 10 == you've cleared Layer 0 of the study plan.

## After this

Layer 1 of the study plan: an MLP on MNIST in pure NumPy, then dropout +
batchnorm, then a ResNet block. Different repo, same build-it-from-scratch
spirit — ask when you're ready and I'll set up the next scaffold.
