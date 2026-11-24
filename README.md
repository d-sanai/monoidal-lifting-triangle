# Monoidal Lifting Triangles (MLT)

**Exact Bit-Lossless Neural-Symbolic Computing and Topological Self-Introspection in a 64-Byte Cell**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Standard: C99](https://img.shields.io/badge/Standard-C99-brightgreen.svg)]()
[![Dependencies: None](https://img.shields.io/badge/Dependencies-Zero%20Heap%20%7C%20No%20libm-orange.svg)]()

---

## Overview

Modern artificial intelligence remains split between continuous neural networks—which suffer from floating-point rounding drift, non-associativity, and catastrophic forgetting—and discrete symbolic solvers, which face combinatorial search explosion.

This repository provides the reference C99 implementation and reproducible benchmark suite demonstrating that enforcing a single uncompromising constraint—**exact 0-bit lossless conservation**—causes both paradigms to collapse into a single uniform computing primitive: the **Monoidal Lifting Triangle (MLT)** (`lifting_core_v2.h`).

### Key Properties

* **Single-Header, Zero-Heap, `-lm`-Free C99 Core:** Every primitive is implemented as a cache-line-aligned **64-byte structure** (`LiftingCell` / `LiftingCellLog`) with scalar and ARM64 NEON SIMD backends.
* **Exact Integer Dual Charts:** Maps seamlessly between orthogonal modular arithmetic ($\mathbb{Z}/2^{16}\mathbb{Z}$) and Log-Polar Fourier Holographic Reduced Representations (FHRR) using only **7.5 KB** of L1-resident lookup tables, reproducing the Frobenius axioms of the **ZX-calculus** ($Z$-spider, $X$-spider, Hadamard transitions) without prior theoretical imposition.
* **Backtrack-Free $O(N)$ Causal Abduction:** Dynamic currying via the `0x8000` duality synthesizes Boolean/arithmetic circuits directly into Algebraic Normal Form (ANF), inverting a 64-bit ripple-carry adder ($2^{127}$ states, 127 unknowns) in **127 deterministic steps**.
* **$O(1)$-Memory Backprop-Free Local Learning:** Trains 256-dimensional, 64-layer networks (16,384 weights) without storing intermediate activations, achieving **0-bit deterministic thread parity** (`1T == 10T`) and zero catastrophic forgetting via fixed-pool hysteresis metabolism.
* **Simplicial ($\Delta^2 / \Delta^3$) Self-Introspection & Bidirectional AST:** Unifies deduction, abduction, and learning into inner horn filling ($\Lambda^2_k \to \Delta^2$), and extends to 3D tetrahedral meshes ($\Delta^3$) capable of blind gauge-breaking identification and autonomous **1–4 Pachner subdivision**. The resulting circuit topology serves directly as its own verifiable bidirectional Abstract Syntax Tree (AST).

---

## Architecture & Empirical Highlights

| Layer | Core Mechanism | Benchmark Suite | Verified Result |
| :--- | :--- | :--- | :--- |
| **1. Dual-Chart Substrate** | Pure-integer $\mathbb{Z}/2^{16}\mathbb{Z} \leftrightarrow$ Log-Polar FHRR via 7.5 KB LUT; ZX Frobenius axioms | `test_lifting_core_v2.c`<br>`lifting_fp_vs_z16_ablation.c` | Round-trip error $\le 0\text{ LSB}$ ($\rho$), $\le 0.220^\circ$ ($\theta$); 64-layer ladder $\mu^{64}\circ\Delta^{64}=\mathrm{id}$ **0-bit exact**; **2.54–2.75 G cells/s** (ARM64 NEON) |
| **2. Curried Logic & Abduction** | `0x8000` dynamic currying (ANF), topological degree separation, 1-UIP CDCL | `local_abduction_lifting.c`<br>`reversible_multiplier_8bit_cdcl.c` | 64-bit adder inverted in **127 steps** (0 backtracks); 10-bit multiplier factored with **0 branches**; hard Sudoku solved in a **3 KB fixed pool** |
| **3. $O(1)$ Continual Metabolism** | Dual-stream reverse 2-morphism holonomy updates, $H^1$ OOD detection, slot mitosis/pruning | `lifting_v2_ladder256_ffi.c`<br>`bench_continual_3morph_v2.py` | 256D $\times$ 64L trained with **$O(1)$ activation memory** (`1T == 10T`); **100%** `ctx_id`-free routing; **+0.0% forgetting** across 7 seasons |
| **4. Higher Simplices ($\Delta^2/\Delta^3$)** | Inner horn filling, $\Delta \leftrightarrow \text{Y}$ rewriting, $S^3$ $H^2$ monopole & counterfactual 1–4 Pachner moves | `bench_pachner_simplicial_v2.c`<br>`bench_chart_phase_sep.c`<br>`bench_3d_adaptive_14_learning.c` | $U(1) \to C_4 \to \{e\}$ blind phase recovery $< 1^\circ$; blind localization of hidden `Tet 3` & `Tet 11` subdivisions via 1–4 Pachner refinement |

---

## Quick Start

### 1. Run All C99 Verification Suites (One Command)

No external libraries (`-lm` is not required) and zero dynamic memory allocations (`malloc` is never called):

```bash
make test
```

To compile and run a single suite manually:

```bash
cc -std=c99 -O3 -Wall -Wextra test_lifting_core_v2.c -o test_lifting_v2 && ./test_lifting_v2
```

### 2. Run Python Zero-Copy NumPy & Continual Learning Suites

Requires Python 3.10+ and `numpy`:

```bash
make pytest
```

This builds `liblifting_core_v2.dylib` (macOS) or `liblifting_core_v2.so` (Linux) and executes the 64-byte cache-aligned zero-copy `ndarray` verification (`lifting_core_v2.py`) and multi-season continual learning benchmark (`bench_continual_3morph_v2.py`).

### 3. Build the LaTeX Manuscript

```bash
make paper
```

---

## Repository Layout

```text
.
├── LICENSE                              # MIT License
├── Makefile                             # Cross-platform C99 / Shared Lib / Python / LaTeX build
├── README.md                            # Project overview & reproducibility guide
├── lifting_core_v2.h                    # Single-header 64B MLT core (Scalar + ARM64 NEON)
├── lifting_core_v2_ffi.c                # C ABI shared-library wrapper for core v2
├── lifting_v2_ladder256_ffi.c           # 256D x 64L O(1)-memory backprop-free learner FFI
├── lifting_core_v2.py                   # Zero-copy NumPy structured-array bindings & tests
├── test_lifting_core_v2.c               # 8-module integrated core verification suite
├── lifting_fp_vs_z16_ablation.c         # FP32/FP16/BF16 vs Z/2^16Z exactness ablation
├── local_abduction_lifting.c            # Dynamic currying & 64-bit adder 127-step abduction
├── reversible_multiplier_8bit_cdcl.c    # Multiplier factorization & 1-UIP CDCL in 3 KB pool
├── bench_continual_3morph_v2.py         # Autonomous routing, OOD growth & 7-season metabolism
├── bench_pachner_simplicial_v2.c        # 2D Pachner moves (1-3, 3-1, 2-2) on S^2 and T^2
├── bench_chart_phase_sep.c              # Blind U(1) -> C_4 -> {e} gauge-breaking recovery
├── bench_3d_adaptive_14_learning.c      # 3D S^3 tetrahedral self-introspection & 1-4 Pachner
└── paper/
    └── main.tex                         # Full LaTeX manuscript
```

---

## Citation

If you use this codebase or theoretical framework in your research, please cite:

```bibtex
@misc{mlt2026,
  title        = {Monoidal Lifting Triangles: Exact Bit-Lossless Neural-Symbolic Computing and Topological Self-Introspection in a 64-Byte Cell},
  author       = {The Monoidal Lifting Triangles Team},
  year         = {2026},
  howpublished = {Zenodo / GitHub},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {[https://github.com/](https://github.com/)d-sanai/monoidal-lifting-triangle}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.