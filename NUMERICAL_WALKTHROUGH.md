# Numerical Walkthrough — A Complete Worked Example

This document traces a **single 4×4 image** through every stage of the quantum
image segmentation pipeline, showing the formula and the plugged-in numbers at
each step. It is the numerical companion to [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md).

> **Conventions used here**
> - All numbers are **hand-derived** to show the math explicitly. They follow the
>   exact formulas in `quantum_seg/` but are rounded to 2–4 significant figures.
> - For the GMM unary term we use a **shared, regularized variance σ = 50** for
>   both classes, purely so the log-likelihood numbers stay legible (the real
>   `sklearn` GMM uses per-component covariance; the structure is identical).
> - Grayscale images map to Lab with `a ≈ b ≈ 0`, so the color feature is just
>   the **L (intensity)** channel — consistent with how the pipeline treats
>   grayscale input.

---

## Stage 0 — The Input Image

We take a 4×4 grayscale patch from a downsized `house.tiff`: a **bright house on
the right** (high intensity ≈ 200) against a **dark background on the left**
(low intensity ≈ 55). Pixel values `f(y,x)` are 8-bit integers (0–255):

```
        x=0   x=1   x=2   x=3
 y=0 [   50,   55,  200,  205 ]
 y=1 [   55,   60,  210,  200 ]
 y=2 [   60,   55,  205,  210 ]
 y=3 [   55,   50,  200,  205 ]
```

- Side length = 4 = 2ⁿ  ⟹  **n = 2**
- Bit depth  = 8 bits   ⟹  **q = 8**
- There is a strong vertical edge between columns x=1 and x=2 (dark → bright).

This single image feeds **all four stages** below.

---

## Stage 1 — NEQR Lossless Quantum Encoding

**Source:** `quantum_seg/neqr.py`

### Formula

```
|I⟩ = (1 / 2ⁿ) · Σ_{y=0}^{2ⁿ-1} Σ_{x=0}^{2ⁿ-1}  |f(y,x)⟩ ⊗ |y⟩ ⊗ |x⟩
```

With n = 2 the normalizing factor is `1/2ⁿ = 1/4`, and the position registers
`|y⟩`, `|x⟩` are 2 qubits each (MSB-first).

### Encoding individual pixels

Convert each coordinate and intensity to binary:

| Pixel (y,x) | value f | `|y⟩` | `|x⟩` | color `|f⟩` (8-bit) |
|-------------|---------|-------|-------|----------------------|
| (0,0)       | 50      | `00`  | `00`  | `00110010`           |
| (0,2)       | 200     | `00`  | `10`  | `11001000`           |
| (2,2)       | 205     | `10`  | `10`  | `11001101`           |
| (3,0)       | 55      | `11`  | `00`  | `00110111`           |

> Check: 50 = 32+16+2 = `00110010`; 200 = 128+64+8 = `11001000`;
> 205 = 128+64+8+4+1 = `11001101`. ✓

### The explicit superposition (first few terms)

```
|I⟩ = (1/4) [ |00110010⟩|00⟩|00⟩      ← pixel (0,0) = 50
            + |00110111⟩|00⟩|01⟩      ← pixel (0,1) = 55
            + |11001000⟩|00⟩|10⟩      ← pixel (0,2) = 200
            + |11001101⟩|00⟩|01...    ...
            + ... (16 terms total) ]
```

### Circuit and resources

1. **Hadamard** on all `2n = 4` position qubits → uniform superposition over all
   `2^{2n} = 16` pixel coordinates, each with amplitude `1/4`.
2. **Multi-controlled-X** gates write each pixel's color bits, controlled on the
   position pattern.

| Quantity | Formula | Value |
|----------|---------|-------|
| Total qubits | `2n + q` | `4 + 8 = 12` |
| Pixels encoded | `2^{2n}` | `16` |
| Amplitude per term | `1/2ⁿ` | `0.25` |
| Gate-count order | `O(2^{2n}·q)` | `O(16·8) = O(128)` |

**Normalization check:** 16 terms × (1/4)² = 16 × 1/16 = **1.0** ✓

### Losslessness

Because every register is in the **computational basis**, measuring position
`|10⟩|10⟩` collapses the color register to exactly `|11001101⟩` = 205 — the
original pixel, with **zero error**:

```
MSE = 0   ⟹   PSNR = 10·log₁₀(255² / 0) = ∞      SSIM = 1.0
```

---

## Stage 2 — QHED Quantum Edge Detection

**Source:** `quantum_seg/qsobel.py` → `qhed_1d`, `qhed_edges`

### Core formula (1D)

Amplitude-encode a normalized row, apply a Hadamard to the LSB qubit. For each
adjacent pair `(a, b)`:

```
H on LSB:  (a, b)  →  ( (a+b)/√2 ,  (a−b)/√2 )
```

The odd-indexed outputs `(a−b)/√2`, rescaled by `√2 · ‖row‖`, recover the raw
forward differences. The code does this in two passes (Pass A on the original,
Pass B on the cyclically-shifted signal) to capture every adjacent pair.

### Worked row: y = 0  →  `[50, 55, 200, 205]`

**Norm:**
```
‖row‖ = √(50² + 55² + 200² + 205²) = √(2500+3025+40000+42025) = √87550 ≈ 295.89
```

**Pass A** — pairs `(50,55)` and `(200,205)`:
```
g₀ = (c₀ − c₁) = 50 − 55  = −5
g₂ = (c₂ − c₃) = 200 − 205 = −5
```
(The `/√2` from the Hadamard and the `×√2×‖row‖` rescale cancel, leaving the raw
difference — this is exactly what `diffA = sA[1::2]*√2*norm` computes.)

**Pass B** — cyclic shift `[55, 200, 205, 50]`, pairs `(55,200)` and `(205,50)`:
```
g₁ = c₁ − c₂ = 55 − 200 = −145
g₃ = c₃ − c₀ = 205 − 50 = 155   ← wraparound term, dropped (grad[N−1] = 0)
```

**Assemble** `grad[0::2]=PassA`, `grad[1::2]=PassB`, drop last:
```
grad = [−5, −145, −5, 0]    →   |grad| = [5, 145, 5, 0]
```

The value **145** at index 1 is the dark→bright edge between x=1 and x=2. ✓

### All rows (horizontal gradient `gh`)

Applying the same procedure to each row:

```
gh =
[   5,  145,    5,   0 ]
[   5,  150,   10,   0 ]
[   5,  150,    5,   0 ]
[   5,  150,    5,   0 ]
```

### Worked column: x = 2  →  `[200, 210, 205, 200]` (vertical gradient `gv`)

```
Pass A: g₀ = 200−210 = −10 ,  g₂ = 205−200 = 5
Pass B (shift [210,205,200,200]): g₁ = 210−205 = 5 , g₃ = 200−200 = 0 (dropped)
grad = [−10, 5, 5, 0]  →  |grad| = [10, 5, 5, 0]
```

All columns give:
```
gv =
[   5,    5,   10,    5 ]
[   5,    5,    5,   10 ]
[   5,    5,    5,    5 ]
[   0,    0,    0,    0 ]
```

### 2D edge magnitude

```
E(y,x) = √( gh(y,x)² + gv(y,x)² )      then normalize by max
```

Example at (0,1): `E = √(145² + 5²) = √21050 = 145.09`.
Max over the map is `≈ 150.08`, so dividing through:

```
Edge map E (normalized to [0,1]):
        x=0    x=1    x=2    x=3
 y=0 [ 0.05,  0.97,  0.07,  0.03 ]
 y=1 [ 0.05,  1.00,  0.07,  0.07 ]
 y=2 [ 0.05,  1.00,  0.05,  0.03 ]
 y=3 [ 0.03,  1.00,  0.03,  0.00 ]
```

The bright column at **x = 1** is the quantum-detected boundary between the dark
and bright halves. These values feed the QUBO pairwise term in Stage 3.

---

## Stage 3 — Superpixels and QUBO Construction

**Source:** `quantum_seg/superpixel.py`

### Superpixels (four 2×2 blocks)

```
   SP0 │ SP1          SP0 = rows{0,1}, cols{0,1}   (dark)
   ────┼────          SP1 = rows{0,1}, cols{2,3}   (bright)
   SP2 │ SP3          SP2 = rows{2,3}, cols{0,1}   (dark)
                      SP3 = rows{2,3}, cols{2,3}   (bright)
```

**Mean intensity per superpixel** (L channel):
```
SP0 = (50+55+55+60)/4   = 55.00
SP1 = (200+205+210+200)/4 = 203.75
SP2 = (60+55+55+50)/4   = 55.00
SP3 = (205+210+200+205)/4 = 205.00
```

**4-connected adjacency:** `{SP0–SP1, SP0–SP2, SP1–SP3, SP2–SP3}`
(SP0–SP3 and SP1–SP2 are diagonal, so not adjacent.)

### Seeds (trimap)

We mark the obvious corners as seeds and leave the rest free:
```
SP0 = background seed  (z₀ = 0, fixed)
SP1 = foreground seed  (z₁ = 1, fixed)
free variables = { SP2, SP3 }      ← these two are adjacent ⟹ 2-qubit QAOA
```

### Unary terms  `hᵢ = log P(cᵢ|BG) − log P(cᵢ|FG)`

With a shared variance σ = 50 (σ² = 2500), the Gaussian log-likelihood constant
cancels in the difference, giving the clean form:

```
hᵢ = [ (cᵢ − μ_fg)²  −  (cᵢ − μ_bg)² ] / (2σ²)
```

Seed-derived means: `μ_bg = 55` (from SP0), `μ_fg = 203.75` (from SP1).

**SP2 (c = 55):**
```
h₂ = [ (55 − 203.75)²  −  (55 − 55)² ] / 5000
   = [ 22126.6  −  0 ] / 5000  =  +4.425       (positive → prefers background)
```

**SP3 (c = 205):**
```
h₃ = [ (205 − 203.75)²  −  (205 − 55)² ] / 5000
   = [ 1.56  −  22500 ] / 5000  =  −4.500       (negative → prefers foreground)
```

### Pairwise terms  `Jᵢⱼ = β · exp(−‖cᵢ−cⱼ‖²/σ²) · (1 − qhed_ij)`

Parameters: `β = 10`. The bandwidth σ is the **median** adjacent color distance:
```
distances = { SP0–SP1: 148.75, SP0–SP2: 0, SP1–SP3: 1.25, SP2–SP3: 150 }
sorted = [0, 1.25, 148.75, 150]   →   median σ = (1.25 + 148.75)/2 = 75   (σ² = 5625)
```

`qhed_ij` = mean of the Stage-2 edge map along each shared boundary:

| Edge | ‖cᵢ−cⱼ‖ | boundary pixels (E) | qhed_ij | Jᵢⱼ |
|------|---------|---------------------|---------|------|
| SP0–SP2 | 0 | E[1,0],E[2,0],E[1,1],E[2,1] = .05,.05,1.0,1.0 | 0.525 | `10·e^{0}·(1−0.525)` = **4.75** |
| SP1–SP3 | 1.25 | E[1,2],E[2,2],E[1,3],E[2,3] = .07,.05,.07,.03 | 0.055 | `10·e^{−1.56/5625}·0.945` = **9.45** |
| SP2–SP3 | 150 | E[2,1],E[3,1],E[2,2],E[3,2] = 1.0,1.0,.05,.03 | 0.52 | `10·e^{−22500/5625}·0.48` = **0.088** |

> Note how the strong QHED edge between SP2 and SP3 (qhed = 0.52) **and** their
> large color distance both push `J₂₃` to a tiny 0.088 — the model correctly
> says "these two regions straddle an edge, so do **not** force them to share a
> label."

### Seed hard constraints (fold fixed seeds into free neighbors)

```
SP2 (free) has seed neighbor SP0 (bg):   Δh₂ = +J₀₂ = +4.75
SP3 (free) has seed neighbor SP1 (fg):   Δh₃ = −J₁₃ = −9.45
```

**Effective unary terms:**
```
h₂_eff = h₂ + J₀₂ = 4.425 + 4.75  =  +9.175
h₃_eff = h₃ − J₁₃ = −4.500 − 9.45 =  −13.947
```

### Reduced QUBO (2 free variables z₂, z₃)

```
E(z₂,z₃) = 9.175·z₂  −  13.947·z₃  +  0.088·(z₂ − z₃)²
```

Enumerate all four assignments:

| z₂ z₃ | energy E | meaning |
|-------|----------|---------|
| 0  0  | 0.000    | both bg |
| 1  0  | +9.263   | SP2 fg, SP3 bg |
| **0  1**  | **−13.859** | **SP2 bg, SP3 fg ← minimum** |
| 1  1  | −4.772   | both fg |

The QUBO minimum is `(z₂, z₃) = (0, 1)` — **SP2 background, SP3 foreground**,
exactly matching the image. Now we solve the same problem with QAOA.

---

## Stage 4 — QUBO → Ising → QAOA → Mask

**Source:** `quantum_seg/qaoa_seg.py`

### QUBO → Ising conversion

```
h_ising[i] = −h_eff[i] / 2          J_ising[i,j] = −Jᵢⱼ / 2
```

```
qubit 0 = SP2:  h_ising,0 = −9.175 / 2   = −4.588
qubit 1 = SP3:  h_ising,1 = −(−13.947)/2 = +6.974
coupling:       J_ising   = −0.088 / 2   = −0.044
```

### Cost and Mixer Hamiltonians

```
H_C = −4.588·Z₀  +  6.974·Z₁  −  0.044·Z₀⊗Z₁

H_M =  X₀  +  X₁
```

**Sign check** (spin sᵢ = +1 for z=0, −1 for z=1):

| (z₂,z₃) | (s₀,s₁) | E_Ising = −4.588 s₀ + 6.974 s₁ − 0.044 s₀s₁ |
|---------|---------|----------------------------------------------|
| (0,0)   | (+,+)   | −4.588 + 6.974 − 0.044 = +2.342 |
| (1,0)   | (−,+)   | +4.588 + 6.974 + 0.044 = +11.606 |
| **(0,1)** | **(+,−)** | **−4.588 − 6.974 + 0.044 = −11.518 ← min** |
| (1,1)   | (−,−)   | +4.588 − 6.974 − 0.044 = −2.430 |

Same minimizer as the QUBO: `(0,1)`. ✓

### QAOA circuit (depth p = 1)

```
|ψ(γ,β)⟩ = e^{−iβH_M} · e^{−iγH_C} · |+⟩⊗|+⟩
```

For a single qubit with `H_C = h·Z`, mixer `X`, the p=1 expectation has the
closed form (derived by direct evolution of `|+⟩`):

```
⟨Z⟩(γ,β) = − sin(2β) · sin(2γh)
P(0) = ½(1 − sin2β·sin2γh)      P(1) = ½(1 + sin2β·sin2γh)
```

Because the ZZ coupling is tiny (−0.044), the two qubits are nearly independent,
so the total cost is the sum of two single-qubit terms:

```
⟨H_C⟩(γ,β) ≈ −sin(2β) · [ 4.588·sin(9.176γ)  +  6.974·sin(13.948γ) ]
```

(using `2γh₀ = 2γ·(−4.588)` and `2γh₁ = 2γ·6.974`; signs folded into the bracket).

### Finding the optimal angles

To minimize, set `sin2β = +1 ⟹ β* = π/4 ≈ 0.785`, then maximize
`g(γ) = 4.588·sin(9.176γ) + 6.974·sin(13.948γ)`:

| γ | sin(9.176γ) | sin(13.948γ) | g(γ) |
|------|-------------|--------------|------|
| 0.110 | 0.860 | 0.946 | 10.54 |
| 0.120 | 0.892 | 0.995 | 11.03 |
| **0.125** | **0.911** | **0.986** | **11.06 ← max** |
| 0.140 | 0.959 | 0.928 | 10.87 |

So **γ\* ≈ 0.125, β\* ≈ 0.785**, giving:

```
⟨H_C⟩_min ≈ −11.06
```

Compared to the exact ground energy −11.518, the QAOA p=1 approximation ratio is
`11.06 / 11.518 ≈ 0.96` — it captures **96 %** of the optimum at depth 1, a
textbook illustration of the QAOA approximation gap.

### Output probabilities at (γ*, β*)

```
qubit 1 (SP3, h=+6.974):  P(1) = ½(1 + sin(13.948·0.125)) = ½(1+0.986) = 0.993  → |1⟩ (fg)
qubit 0 (SP2, h=−4.588):  P(0) = ½(1 + sin(9.176·0.125))  = ½(1+0.911) = 0.956  → |0⟩ (bg)
```

Most-probable joint state: `q₀q₁ = "01"`, with joint probability
`0.956 × 0.993 ≈ 0.95`.

### Decoding (MSB-first)

```
best_idx = 1   (binary 01, wire 0 = MSB)
z[i] = (best_idx >> (n_free−1−i)) & 1
  → z₂ = (1 >> 1) & 1 = 0   (SP2 = background)
  → z₃ = (1 >> 0) & 1 = 1   (SP3 = foreground)
```

### Mask reconstruction

Combine seed labels (SP0=bg, SP1=fg) with QAOA results (SP2=bg, SP3=fg) and
expand each superpixel back to its 2×2 pixel block:

```
   SP0=0 │ SP1=1            0 0 │ 1 1
   ──────┼──────           0 0 │ 1 1
   SP2=0 │ SP3=1     →     ────┼────
                           0 0 │ 1 1
                           0 0 │ 1 1
```

**Final foreground/background mask:**
```
0 0 1 1
0 0 1 1
0 0 1 1
0 0 1 1
```

The left half (dark background) is labeled 0 and the right half (the bright
house) is labeled 1 — **the quantum pipeline correctly separated foreground
from background**, matching the input image.

---

## End-to-End Summary

| Stage | Operation | Key formula | Output (this example) |
|-------|-----------|-------------|------------------------|
| 0 | Input | — | 4×4 image, n=2, q=8 |
| 1 | NEQR encode | `\|I⟩ = (1/2ⁿ)Σ\|f⟩\|y⟩\|x⟩` | 12-qubit lossless state, PSNR=∞ |
| 2 | QHED edges | `H on LSB → (a−b)/√2` | edge map, strong column at x=1 |
| 3a | Superpixels | SLIC / 2×2 blocks | SP means: 55, 204, 55, 205 |
| 3b | Unary | `h = [(c−μ_fg)²−(c−μ_bg)²]/2σ²` | h₂=+4.43, h₃=−4.50 |
| 3c | Pairwise | `J = β·e^{−Δc²/σ²}(1−qhed)` | J₂₃=0.088, J₀₂=4.75, J₁₃=9.45 |
| 3d | Seed fold | `Δh ±= J` | h₂_eff=+9.18, h₃_eff=−13.95 |
| 4a | QUBO→Ising | `h_I=−h/2, J_I=−J/2` | H_C = −4.59 Z₀ + 6.97 Z₁ − 0.04 Z₀Z₁ |
| 4b | QAOA p=1 | `⟨Z⟩=−sin2β·sin2γh` | γ*≈0.125, β*≈0.785, ⟨H_C⟩≈−11.06 |
| 4c | Decode | `argmax → bitstring` | "01" → z₂=0, z₃=1 |
| 4d | Mask | SP labels → pixels | left=bg, right=fg ✓ |

Every arrow in this table corresponds to a function in `quantum_seg/`; see
[TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) for the code-level walkthrough and
[QA.md](QA.md) for conceptual questions.
