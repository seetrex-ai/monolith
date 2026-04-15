# Monolith

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://github.com/seetrex-ai/monolith)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/seetrex-ai/monolith/actions/workflows/ci.yml/badge.svg)](https://github.com/seetrex-ai/monolith/actions)

**Differentiable EML trees for symbolic regression via gradient descent on a universal operator.**

The EML operator `eml(x, y) = exp(x) - ln(y)` is a universal binary operator for continuous mathematics ([arXiv 2603.21852](https://arxiv.org/abs/2603.21852)) — the continuous analogue of the NAND gate. Combined with the constant 1, it generates all elementary functions from a single grammar: `S → 1 | eml(S, S)`.

Monolith packages EML trees as a reusable PyTorch module for symbolic regression, building on the gradient-based training demonstrated by [Odrzywołek (2026)](https://arxiv.org/abs/2603.21852). It provides a single `fit()` call with hierarchical multi-depth search, symbolic decompilation, and baseline comparisons.

> **Paper:** [Monolith: Differentiable EML Trees for Symbolic Regression via Gradient Descent on a Universal Operator](https://zenodo.org/records/19592926) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19592926.svg)](https://doi.org/10.5281/zenodo.19592926)

## Why Monolith?

Every symbolic regression method (PySR, AI Feynman, PSRN) relies on human-curated operator libraries (`+`, `-`, `×`, `/`, `exp`, `sin`, ...). Monolith uses **one operator and one constant**. This is the minimal possible grammar for elementary function discovery — a lower bound on what gradient descent can achieve without domain-specific engineering.

| Method | Operators | MSE on exp(x) | Output |
|---|---|---|---|
| PySR | +,-,×,/,exp,log,sin,sqrt | 0 (exact) | `exp(x)` |
| MLP (49 params) | tanh, linear | 3×10⁻⁶ | opaque |
| **Monolith** (6 params) | **eml only** | 1.3×10⁻⁵ | `exp(x)` |

Monolith is not a competing tool — it packages EML tree training into a reusable library with honest baselines.

## Installation

```bash
git clone https://github.com/seetrex-ai/monolith.git
cd monolith
pip install -e ".[dev]"
```

## Quick start

```python
import torch
from monolith import EMLTree

# Generate data
x = torch.linspace(-2, 2, 200).unsqueeze(1)
y = torch.exp(x.squeeze(1))

# Train: multi-depth search with restarts
tree = EMLTree.fit(x, y, max_depth=3, n_restarts=10, epochs=10000, verbose=True)

# Evaluate
y_pred = tree(x)

# Decompile to SymPy expression
print(tree.to_symbolic())  # exp(x)
```

### With tau annealing

Tau annealing ([Odrzywołek, 2026](https://arxiv.org/abs/2603.21852)) forces leaves toward discrete assignments during training, enabling cleaner symbolic recovery:

```python
tree = EMLTree.fit(x, y, max_depth=3, n_restarts=10, epochs=10000,
                   tau_search=2.5, tau_hard=0.01, verbose=True)

print(tree.snap_symbolic(x, y))  # clean formula if snap succeeds
```

## Results

### Elementary function recovery

7/7 elementary functions converge with ≤24 parameters at depth 3:

| Function | Domain | MSE | RMSE | Parameters |
|---|---|---|---|---|
| exp(x) | [-2, 2] | 1.3×10⁻⁵ | 0.004 | 6 (depth 1) |
| sqrt(x) | [0.5, 4] | 1.6×10⁻⁵ | 0.004 | 24 |
| ln(x) | [0.5, 3] | 2.0×10⁻⁵ | 0.005 | 24 |
| x³ | [0.5, 2] | 1.1×10⁻³ | 0.033 | 24 |
| 1/x | [0.5, 3] | 1.1×10⁻³ | 0.033 | 24 |
| x² | [0.5, 2.5] | 1.7×10⁻³ | 0.041 | 24 |
| sin(x) | [-3, 3] | 4.1×10⁻³ | 0.064 | 24 |

### Depth barrier and hierarchical training

Random initialization at depth 4 always diverges (MSE ≈ 10¹⁷). We tested 12 initialization strategies — only hierarchical warm start (training depth n-1 first) enables convergence:

| Strategy | MSE (x²) | Converges? |
|---|---|---|
| Random init (any lr/gc) | 2.35×10¹⁷ | No |
| Progressive clamp (3 schedules) | 2,250–9,492 | No |
| Smart init (3 variants) | 2.35×10¹⁷ | No |
| **Warm start: random extend** | **1.9×10⁻⁴** | **Yes** |

Hierarchical training yields **12.9× improvement** for sin(x²) at depth 4 vs depth 3.

### Baselines

| Method | Wins (9 functions) | MSE range | Time/fn | Output |
|---|---|---|---|---|
| PySR | 9/9 | 0 (exact) | ~5s | clean formula |
| MLP (49 params) | 9/9 | <10⁻⁵ | ~30s | opaque |
| **Monolith** | 0/9 | 10⁻⁵ to 10⁻² | ~500s | structured expression |

Both baselines outperform Monolith. This is expected: PySR has the target functions as primitives; the MLP is a universal approximator. Monolith uses a single operator and demonstrates a structural principle, not a competitive tool.

## API

### `EMLTree.fit(x, y, max_depth=3, n_restarts=10, epochs=10000, **kwargs)`

Train an EMLTree via hierarchical multi-depth search with restarts. Returns the best tree found.

- **x** — input data, shape `(n_samples,)` or `(n_samples, n_vars)`
- **y** — target values, shape `(n_samples,)`
- **max_depth** — maximum tree depth to try (default 3)
- **n_restarts** — random seeds per depth (default 10)
- **epochs** — training epochs per run (default 10000)
- **tau_search** — softmax temperature during search phase (default 1.0; set >1.0 to enable tau annealing)
- **tau_hard** — temperature target at end of hardening (default 0.01)
- **hardening_fraction** — fraction of epochs for hardening phase (default 0.25)
- **lam_entropy** — entropy penalty weight during hardening (default 0.02)

### `tree(x, tau=1.0)` — Forward pass

Evaluate the tree. Input `(batch, n_vars)` → output `(batch,)`. Optional `tau` controls softmax temperature.

### `tree.to_symbolic()` — Faithful decompilation

Returns a SymPy expression. Leaves with >95% confidence snap to discrete candidates; others become linear expressions with numeric coefficients.

### `tree.snap_symbolic(x, y, tol=2.0)` — Snap decompilation

Forces argmax on all leaves. Returns a clean SymPy expression if the snapped version fits within tolerance, otherwise `None`.

### `tree.leaf_entropy(tau=1.0)` — Diagnostic

Per-leaf Shannon entropy. Low values = leaf has decided; high = undecided.

## Tests

```bash
pytest tests/ -v                # all tests (41 total, ~16 min)
pytest tests/ -v -m "not slow"  # fast only (28 tests, ~5s)
```

## Development

```bash
git clone https://github.com/seetrex-ai/monolith
cd monolith
pip install -e ".[dev]"
pytest
```

## Roadmap

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Priority areas:

- [x] **Tau annealing** — temperature annealing for discrete leaf assignments, inspired by [Odrzywołek (2026)](https://arxiv.org/abs/2603.21852)
- [ ] **Multi-variable support** — validate f(x, y) and evaluate on Feynman benchmark equations
- [ ] **GPU acceleration** — parallelize multi-restart loop for faster training
- [ ] **Depth 5+ scaling** — extend hierarchical training beyond depth 4

Open a [Discussion](https://github.com/seetrex-ai/monolith/discussions) for questions or ideas.

## Citation

```bibtex
@software{tabares2026monolith,
  title={Monolith: Differentiable EML Trees for Symbolic Regression via Gradient Descent on a Universal Operator},
  author={Tabares Montilla, Jes{\'u}s},
  year={2026},
  doi={10.5281/zenodo.19592926},
  url={https://github.com/seetrex-ai/monolith}
}
```

## License

[MIT](LICENSE)
