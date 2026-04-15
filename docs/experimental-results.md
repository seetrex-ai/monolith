# Monolith — Experimental Results

> All experiments run on 2026-04-14. Python 3.12.5, PyTorch 2.x, CPU only (no GPU). Windows 10 Pro.
> Reproducible via `EMLTree.fit()` with seeds documented per experiment.

## 1. EML Operator Validation

The EML operator `eml(x, y) = exp(x) - ln(y)` (arXiv 2603.21852) was validated against known identities:

| Identity | Expected | Verified | Tolerance |
|----------|----------|----------|-----------|
| `eml(x, 1)` | `exp(x)` | ✓ | atol=1e-6 |
| `eml(0, x)` | `1 - ln(x)` | ✓ | atol=1e-6 |
| `eml(1, 1)` | `e ≈ 2.71828` | ✓ | atol=1e-6 |

Numerical stability verified with hard clamping (exp input `[-20, 20]`, ln input `≥ 1e-7`):
- `safe_eml(1000, 1.0)` → finite (exp clamped to exp(20) ≈ 4.85e8)
- `safe_eml(0, 0)` → finite (ln input clamped to 1e-7)
- `safe_eml(0, -5)` → finite (negative right clamped to 1e-7)
- Gradient flow through `safe_eml`: verified non-None, finite for both inputs

## 2. Inter-Level Clamping: Discovery and Removal

### The problem

The EML operator is inherently expansive. With near-uniform softmax initialization (scale 0.1), intermediate values grow super-exponentially across tree levels:

```
Level 0 (leaves): values ∈ [-0.33, 1.0] (softmax mixture of [0, 1, x])
Level 1:          eml(~0.5, ~0) = exp(0.5) - ln(1e-7) ≈ 17.8
Level 2:          exp(17.8) ≈ 5.3×10⁷ → saturates exp clamp → gradient = 0
```

This makes training impossible at depth ≥ 2 with naive random initialization.

### Initial attempt: fixed inter-level clamping

Added `values.clamp(-10, 10)` between EML levels. Results:

| Function | Depth | MSE with clamp | MSE without clamp | Factor |
|----------|-------|----------------|-------------------|--------|
| exp(x) | 2 | 0.000290 | 0.000244 | 1.2× |
| x² | 2 | 0.154000 | 0.012339 | **12.5×** |
| x² | 3 | 54.670 (diverged) | 0.001650 | **converges** |

### Conclusion

Fixed inter-level clamping creates a representational floor (MSE=0.154 for x²) and completely blocks convergence at depth ≥ 3. **Removed in favor of low learning rate + gradient clipping**, which allows all 7 elementary functions to converge at depth 3.

### Final training recipe

| Depth | Learning rate | Grad clip | Init strategy |
|-------|--------------|-----------|---------------|
| 1-2 | 0.01 | 1.0 | Random × 0.1 |
| 3 | 0.001 | 0.5 | Random × 0.1 |
| 4+ | 0.0005 | 0.1 | Hierarchical warm start (see §4) |

## 3. Elementary Function Recovery

Configuration: `EMLTree.fit(max_depth=3, n_restarts=20, epochs=10000)`, constants=(0, 1), no inter-level clamping.

| Function | Domain | Best depth | MSE | Params | RMSE | Leaf snap? |
|----------|--------|-----------|-----|--------|------|------------|
| exp(x) | [-2, 2] | 1 | 1.34×10⁻⁵ | 6 | 0.0037 | YES — x(1.00), 1(1.00) |
| sqrt(x) | [0.5, 4] | 3 | 1.58×10⁻⁵ | 24 | 0.0040 | no — leaf 4: 0(0.49) |
| ln(x) | [0.5, 3] | 3 | 2.04×10⁻⁵ | 24 | 0.0045 | no — leaf 4: x(0.43) |
| x³ | [0.5, 2] | 3 | 1.07×10⁻³ | 24 | 0.0327 | no — leaf 7: x(0.70) |
| 1/x | [0.5, 3] | 3 | 1.07×10⁻³ | 24 | 0.0327 | no — leaf 6: 0(0.47) |
| x² | [0.5, 2.5] | 3 | 1.65×10⁻³ | 24 | 0.0406 | no — leaf 4: 1(0.66) |
| sin(x) | [-3, 3] | 3 | 4.06×10⁻³ | 24 | 0.0637 | no — leaf 1: 1(0.64) |

**Result: 7/7 elementary functions converge.** Three (exp, sqrt, ln) achieve near-exact recovery (RMSE < 0.005). All targets fit with ≤ 24 parameters.

**Leaf behavior:** Only exp(x) at depth=1 produces discrete leaf assignments (>99% softmax confidence). All depth=3 results use continuous softmax mixtures as an approximation strategy. Leaves do NOT converge to discrete [0, 1, x] assignments — the continuous mixing is functionally necessary, not residual noise.

## 4. Depth 4+ Investigation

### The depth 4 barrier

Random initialization at depth 4 **always** diverges (MSE ≈ 2.35×10¹⁷) regardless of learning rate or gradient clipping. Four levels of exp() create tower-exponential growth that no hyperparameter combination can overcome.

### Systematic search: 12 strategies tested

Target: x² on [0.5, 2.5]. 20 random seeds per strategy, 15,000-20,000 epochs.

| # | Strategy | MSE | Converges? |
|---|----------|-----|------------|
| 1 | Random init, lr=0.001, gc=0.5 | 2.35×10¹⁷ | No |
| 2 | Random init, lr=0.0005, gc=0.1 | 2.35×10¹⁷ | No |
| 3 | Random init, lr=0.0001, gc=0.1 | 2.35×10¹⁷ | No |
| 4 | Progressive clamp linear (3→50) | 2,250 | No |
| 5 | Progressive clamp step (3→10→30→100) | 9,492 | No |
| 6 | Progressive clamp warmup+release (5→100) | 9,490 | No |
| 7 | Smart init: right children = 1, left = random | 2.35×10¹⁷ | No |
| 8 | Smart init: all biased toward 1, + perturbation | 2.35×10¹⁷ | No |
| 9 | Hybrid: alternate_01 init + progressive clamp | 2,250 | No |
| **10** | **Smart init: alternate (even=0, odd=1, +noise σ=0.3)** | **0.0070** | **Yes** |
| **11** | **Warm start: identity_pair (d3 leaf paired with "1")** | **0.00072** | **Yes** |
| **12** | **Warm start: random_extend (d3 logits → first 8 leaves)** | **0.00019** | **Yes** |

**Result: 3/12 strategies converge.** All three share a common property: intermediate values at initialization are bounded (either via `eml(0,1) = 1` or pre-converged subtrees from depth 3). The remaining 9 strategies produce unbounded intermediates → exp saturation → dead gradients.

### Hierarchical training implementation

Based on finding #12, `EMLTree.fit()` now implements hierarchical training:
- Depths 1-3: independent random initialization with adaptive hyperparameters
- Depth 4+: warm start from the best tree at depth N-1 (random_extend strategy — copy N-1 logits to first half of new leaves, random init for rest)

### Depth scaling for x²

| max_depth | Best depth found | MSE | Parameters | Improvement vs previous |
|-----------|-----------------|-----|------------|------------------------|
| 1 | 1 | 5.30×10⁻¹ | 6 | — |
| 2 | 2 | 9.83×10⁻³ | 12 | 54× |
| 3 | 3 | 1.65×10⁻³ | 24 | 6.0× |
| 4 | 4 | 1.12×10⁻³ | 48 | 1.5× |
| 5 | 4 | 1.12×10⁻³ | 48 | — |
| 6 | 4 | 1.12×10⁻³ | 48 | — |

**Result:** For x², depth 4 is the practical ceiling (1.5× improvement over depth 3). Depth 5-6 add parameters but produce no improvement — the hierarchical warm start prevents divergence but does not force deeper trees to find better solutions.

### Elementary functions at max_depth=5

| Function | Best depth (max_depth=5) | MSE | Improved over depth 3? |
|----------|------------------------|-----|----------------------|
| exp(x) | 1 | 1.34×10⁻⁵ | No (already optimal at d=1) |
| ln(x) | 3 | 2.04×10⁻⁵ | No |

**Result:** Elementary functions of one variable saturate at depth 3. Additional depth capacity is not utilized.

## 5. Composite Function Benchmark

The key question: does depth 4+ provide value for functions that are compositions of elementary operations?

### Focused experiment (20 restarts, 10,000 epochs)

| Function | Domain | d3 MSE | d5 best depth | d5 MSE | Improvement |
|----------|--------|--------|---------------|--------|-------------|
| **sin(x²)** | [0, 2.5] | 7.40×10⁻² | **4** | **5.73×10⁻³** | **12.9×** |
| exp(sin(x)) | [-3, 3] | 3.91×10⁻² | 3 | 3.91×10⁻² | none |

### Fast survey (5 restarts, 5,000 epochs)

| Function | Domain | d3 MSE | d5 best depth | d5 MSE | Improvement |
|----------|--------|--------|---------------|--------|-------------|
| exp(sin(x)) | [-3, 3] | 3.06×10⁻¹ | 1 | 3.06×10⁻¹ | none |
| x·exp(x) | [-2, 2] | 4.03×10⁻² | 2 | 4.03×10⁻² | none |
| ln(x²+1) | [-2, 2] | 5.48×10⁻² | 2 | 5.48×10⁻² | none |
| **sin(x²)** | [0, 2.5] | 8.01×10⁻² | **4** | **1.11×10⁻²** | **7.2×** |
| sqrt(exp(x)) | [-2, 1.5] | ~0 | 2 | ~0 | — (already exact) |
| x/(1+x²) | [-3, 3] | 4.63×10⁻¹ | 3 | 4.63×10⁻¹ | none |

**Result:** sin(x²) is the only composite function where depth 4 provides substantial improvement (7.2-12.9× depending on compute budget). sqrt(exp(x)) = exp(x/2) converges exactly at depth 2 (trivial EML representation). Other compositions show no benefit from depth 4+.

**Interpretation:** sin(x²) benefits because squaring x before applying sin creates a frequency-modulated signal that requires more EML composition levels to approximate. Other tested compositions apparently can be approximated equally well at depth 3 via continuous leaf mixing.

## 6. Configurable Leaf Constants

### Motivation

The default leaf candidates `[0, 1, x]` may limit the tree's ability to represent functions involving mathematical constants. Adding π as a candidate could improve trigonometric targets.

### Implementation

`EMLTree` now accepts a `constants` parameter (default `(0.0, 1.0)`). The candidate vector becomes `[c_0, c_1, ..., c_n, x_1, ..., x_k]`. Backward compatible — all prior results use the default.

### Experiment: (0, 1) vs (0, 1, π)

Hypothesis: adding π as a leaf candidate would improve trigonometric composite functions by providing a domain-relevant constant.

Configuration: max_depth=5, 20 restarts, 10,000 epochs. Focused on trigonometric composites.

| Function | (0, 1) depth | (0, 1) MSE | (0, 1, π) depth | (0, 1, π) MSE | Impact |
|----------|-------------|------------|----------------|---------------|--------|
| sin(x²) | 4 | 5.73×10⁻³ | 3 | 1.49×10⁻² | **2.6× worse** |
| exp(sin(x)) | 3 | 3.91×10⁻² | 3 | 6.51×10⁻² | **1.7× worse** |

**Result: Adding π degrades performance.** More candidates per leaf (4 vs 3) increases the parameter count and makes the optimization landscape harder to navigate. The gradient signal is diluted across more options, slowing convergence.

**Conclusion:** The inductive bias of EML trees comes from the operator structure (`exp(x) - ln(y)`), not from the leaf constants. Adding domain-specific constants is not free — it increases ambiguity at each leaf. The default constants `(0, 1)` remain optimal. The configurable constants feature is retained for research flexibility but should not be recommended as a performance improvement.

## 7. Symbolic Decompilation

### Faithful mode: to_symbolic()

Reconstructs the exact expression the tree computes using SymPy:
- Leaves with softmax confidence > 0.95 snap to discrete symbols (0, 1, x, π, ...)
- Leaves with lower confidence → linear expression: `w_1·c_1 + w_2·c_2 + ... + w_k·x`
- EML tree built bottom-up: `exp(left) - log(right)` at each level
- Final expression simplified via `sympy.simplify()`

| Function trained | Depth | Decompiled output | Clean formula? |
|-----------------|-------|-------------------|---------------|
| exp(x) | 1 | `exp(x)` | **Yes** |
| x² | 2 | Numeric EML expression with coefficients | No |
| ln(x) | 3 | Nested EML with mixed coefficients | No |

### Snap mode: snap_symbolic()

Forces argmax on all leaves, builds purely symbolic expression, validates against data:
- Returns expression if `MSE_snap / MSE_original < tolerance` (default 2.0)
- Returns `None` if snap degrades accuracy beyond tolerance

| Function | Snap result | Reason |
|----------|-------------|--------|
| exp(x) depth=1 | `exp(x)` | Leaves are >99% discrete |
| x² depth=2 | `None` | Leaves are mixed (43-66% confidence) |
| ln(x) depth=3 | `None` | Leaves are mixed; forced 0 as right child → log(0) = -∞ |

### Decompilation limitation

Only functions with trivial EML representations (depth=1, 2 leaves) produce clean symbolic formulas via snap. All depth ≥ 2 trained trees use continuous softmax mixtures that cannot be discretized without significant accuracy loss. The faithful decompiler (`to_symbolic`) always produces a correct expression, but it contains numeric coefficients rather than clean symbolic forms.

**This is a fundamental property of the gradient-descent approach**, not a bug. The continuous leaf mixing is how the tree approximates functions that would require much deeper discrete EML trees. The trade-off: fewer parameters and reliable convergence, at the cost of interpretable output.

## 8. Test Suite

32 tests total, all passing:

| Module | Tests | Coverage |
|--------|-------|----------|
| test_eml_ops.py | 7 | EML identities, numerical stability, gradient flow |
| test_eml_tree.py | 18 | Shapes, parameter counts, forward pass, NaN safety, gradient flow, convergence (exp, x², ln), leaf entropy |
| test_symbolic.py | 7+ | Leaf-to-expr snap/mixed, tree building, faithful decompilation, snap decompilation |
| **Total** | **32** | |

Slow tests (convergence + decompilation): ~26 minutes on CPU.

## 9. Architecture

```
EMLTree(depth, n_vars, constants=(0, 1))
├── leaf_logits: Parameter(2^depth × (len(constants) + n_vars))
│   └── softmax → weights over [c_0, c_1, ..., x_1, ..., x_n]
├── forward(x): bottom-up vectorized EML evaluation
│   └── per level: values = safe_eml(left, right)
├── fit(x, y, max_depth, n_restarts, epochs, constants):
│   ├── depth 1-3: independent random init, adaptive lr/gc per depth
│   └── depth 4+: hierarchical warm start from best depth N-1 tree
├── to_symbolic(): faithful SymPy decompilation
│   └── snap threshold 0.95, numeric coefficients for mixed leaves
├── snap_symbolic(x, y, tol): forced discrete decompilation
│   └── argmax all leaves, verify MSE ratio < tolerance, else None
└── leaf_entropy(): per-leaf Shannon entropy diagnostic
```

Total parameters: `2^depth × (len(constants) + n_vars)`.

| depth | constants | n_vars | Parameters |
|-------|-----------|--------|------------|
| 1 | (0, 1) | 1 | 6 |
| 3 | (0, 1) | 1 | 24 |
| 4 | (0, 1) | 1 | 48 |
| 3 | (0, 1, π) | 1 | 32 |
| 4 | (0, 1, π) | 1 | 64 |

## 10. Convergence Rate Analysis

At depth 3 with lr=0.001 and 20 random seeds, convergence threshold MSE < 0.01:

| Function | Converge | Best MSE | Median MSE | Worst MSE |
|----------|----------|----------|------------|-----------|
| ln(x) | 20/20 | 3.8×10⁻⁵ | 5.8×10⁻⁵ | 9.0×10⁻⁵ |
| sqrt(x) | 20/20 | 4.3×10⁻⁴ | 6.0×10⁻⁴ | 8.0×10⁻⁴ |
| x² | 20/20 | 2.8×10⁻³ | 3.0×10⁻³ | 3.2×10⁻³ |
| x³ | 20/20 | 2.7×10⁻³ | 3.2×10⁻³ | 4.1×10⁻³ |
| 1/x | 20/20 | 3.6×10⁻³ | 3.7×10⁻³ | 3.9×10⁻³ |
| exp(x) | 0/20 | 0.377 | 0.889 | 1.089 |
| sin(x) | 0/20 | 0.433 | 0.537 | 0.877 |

**Key finding:** Five functions converge 20/20 with very low variance (best-to-worst ratio < 2×). exp(x) and sin(x) do not converge at depth 3 with lr=0.001 — they require fit()'s adaptive strategy (exp(x) converges at depth 1 with lr=0.01; sin(x) needs the multi-depth search). This reinforces that per-depth hyperparameters are essential.

## 11. Baselines: MLP and PySR

### Important caveat

Both comparisons are **inherently asymmetric**:

- **PySR** has explicit operators (`exp`, `sin`, `log`, `sqrt`, `+`, `-`, `*`, `/`) — the target functions are directly expressible as 1-3 node combinations.
- **MLP** (49 parameters, 1→16→1 with tanh) is a universal function approximator — it can fit any continuous function on a compact domain.
- **Monolith** has ONE operator (`eml`) and the constant 1. All operations must be composed from this single primitive.

Comparing PySR to Monolith is like comparing a calculator to NAND gates. Comparing MLP to Monolith is like comparing a lookup table to a circuit.

### Results (9 functions, 200 data points, best of 20 seeds)

| Function | Monolith MSE | MLP MSE (49p) | PySR MSE | PySR formula |
|----------|-------------|---------------|---------|-------------|
| exp(x) | 1.3×10⁻⁵ | 3×10⁻⁶ | 0 | exp(x) |
| ln(x) | 2.0×10⁻⁵ | <10⁻⁶ | 0 | log(x) |
| sqrt(x) | 1.6×10⁻⁵ | <10⁻⁶ | 0 | sqrt(x) |
| x² | 1.1×10⁻³ | 4×10⁻⁶ | 0 | x*x |
| x³ | 1.1×10⁻³ | 7×10⁻⁶ | 0 | x*x*x |
| 1/x | 2.1×10⁻⁴ | <10⁻⁶ | 0 | 1/x |
| sin(x) | 4.1×10⁻³ | <10⁻⁶ | 0 | sin(x) |
| sin(x²) | 5.7×10⁻³ | 1×10⁻⁶ | 0 | sin(x*x) |
| exp(sin(x)) | 3.9×10⁻² | 1×10⁻⁶ | 0 | exp(sin(x)) |

| Method | Time/fn | Output type | Operators |
|--------|---------|-------------|-----------|
| PySR | ~5s | exact formula | +,-,*,/,exp,log,sin,sqrt |
| MLP | ~30s | opaque | tanh, linear |
| Monolith | ~500s | mixture coefficients | eml only |

**Both baselines win 9/9.** This is expected. Monolith's value is not as a function approximator — it demonstrates that gradient descent can navigate the minimal grammar S → 1 | eml(S, S) without any domain-specific operator engineering.

MLP configuration: `nn.Linear(1,16) → tanh → nn.Linear(16,1)`, Adam lr=0.01, 10k epochs, best of 20 seeds.
PySR configuration: `niterations=40, populations=15, population_size=33, maxsize=20, timeout=60s`.
Monolith configuration: `max_depth=5, n_restarts=20, epochs=10000`, with early stopping (patience=300).

## 12. Summary of Contributions

1. **Reusable EML tree library for PyTorch.** Building on the gradient-based trainer in the original paper's supplementary material (arXiv 2603.21852), Monolith packages EML trees as a pip-installable module with automatic depth selection, multi-restart, and symbolic decompilation.

2. **Recovery of 7/7 elementary functions** via gradient descent with ≤24 parameters. Three functions (exp, ln, sqrt) achieve near-exact recovery (RMSE < 0.005). Five converge 20/20 seeds with low variance.

3. **Systematic characterization of the depth barrier.** 12 initialization/training strategies tested for depth 4. Only 3 work, all requiring bounded intermediate values at initialization.

4. **Hierarchical training** enables depth 4+ convergence where random initialization always diverges. Demonstrated 12.9× improvement for sin(x²) at depth 4 vs depth 3.

5. **Negative result: extended constants degrade performance.** Adding π to leaf candidates worsens convergence 2.6× — more candidates dilute gradient signal.

6. **Two-mode symbolic decompiler**: faithful reconstruction (always correct, may have numeric coefficients) and forced snap (clean formulas when leaf confidence permits).

7. **Honest baselines.** Both MLP (49 params) and PySR outperform Monolith on MSE by orders of magnitude. Documents why the comparisons are structurally asymmetric.

## 13. Known Limitations

1. **Continuous leaf mixing:** Trained trees at depth ≥ 2 use continuous softmax mixtures, not discrete leaf assignments. This means the decompiler produces expressions with numeric coefficients rather than clean symbolic formulas for most targets.

2. **Depth 5+ shows no improvement** for the functions tested (elementary and most composites). Only sin(x²) benefits from depth 4.

3. **Not competitive with PySR or MLP** on standard benchmarks (see §11). This is expected given the asymmetric operator sets.

4. **Single-variable only.** Multi-variable symbolic regression (f(x,y)) has not been validated beyond constructor support.

5. **Training time.** ~8 minutes per function with max_depth=5 on CPU (with early stopping). Dominated by multi-depth search with restarts.
