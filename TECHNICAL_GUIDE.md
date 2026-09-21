# Technical Guide: Quantum Image Segmentation Pipeline

This guide explains every stage of the pipeline in depth — theory, mathematics,
and line-by-line code walkthrough. Read the [README](README.md) first for the
high-level overview.

---

## Part A — Quantum Computing Background

### What is a Qubit?

A classical bit is either 0 or 1. A **qubit** is a two-level quantum system
whose state is a complex superposition:

```
|ψ⟩ = α|0⟩ + β|1⟩      where  |α|² + |β|² = 1
```

- `α` and `β` are complex probability amplitudes.
- Measuring the qubit gives outcome 0 with probability `|α|²` and outcome 1 with
  probability `|β|²`.
- Before measurement, the qubit is simultaneously "0 and 1" — this is superposition.

An **n-qubit register** spans 2ⁿ computational basis states simultaneously,
allowing a single quantum circuit to operate on all 2ⁿ configurations in parallel.

### Quantum Gates Used in This Project

| Gate | Matrix | Effect |
|------|--------|--------|
| **Hadamard (H)** | `[[1,1],[1,-1]]/√2` | `|0⟩ → (|0⟩+|1⟩)/√2`, creates superposition |
| **Pauli-X** | `[[0,1],[1,0]]` | Bit-flip: `|0⟩ → |1⟩`, `|1⟩ → |0⟩` |
| **Pauli-Z** | `[[1,0],[0,-1]]` | Phase-flip: eigenvalues +1 for `|0⟩`, −1 for `|1⟩` |
| **Pauli-X (RX)** | rotation | `RX(θ)|0⟩ = cos(θ/2)|0⟩ − i·sin(θ/2)|1⟩` |
| **RZ(θ)** | diagonal | Phase rotation: `RZ(θ)|0⟩ = |0⟩`, `RZ(θ)|1⟩ = e^{iθ}|1⟩` |
| **CNOT** | 4×4 matrix | Flips target qubit iff control qubit is `|1⟩` |
| **MultiControlledX** | 2ⁿ×2ⁿ | Flips target iff ALL control qubits are `|1⟩` |
| **StatePrep** | – | Directly prepares a given amplitude vector as a quantum state |

### PennyLane Basics

```python
import pennylane as qml
import numpy as np

dev = qml.device("lightning.qubit", wires=3)   # 3-qubit simulator

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)                   # qubit 0 → superposition
    qml.CNOT(wires=[0, 1])                  # entangle qubits 0 and 1
    return qml.state()                       # return full 8-element statevector

state = circuit()
print(state)   # [0.707, 0, 0, 0.707, 0, 0, 0, 0]  (Bell state on qubits 0,1)
```

`lightning.qubit` maintains the full 2ⁿ-element statevector in memory —
exact but exponentially scaling with qubit count.

---

## Part B — Stage 1: NEQR Lossless Quantum Image Encoding

### Theory

The **Novel Enhanced Quantum Representation (NEQR)** (Zhang et al., 2013)
encodes a `2ⁿ × 2ⁿ` grayscale image with `q`-bit pixel intensities as:

```
|I⟩ = (1/2ⁿ) Σ_{y=0}^{2ⁿ-1} Σ_{x=0}^{2ⁿ-1}  |f(y,x)⟩ ⊗ |y⟩ ⊗ |x⟩
```

Where:
- `|y⟩` — n-qubit **row register** (basis-encoded row index)
- `|x⟩` — n-qubit **column register** (basis-encoded column index)
- `|f(y,x)⟩` — q-qubit **color register** (basis-encoded pixel intensity)
- The factor `1/2ⁿ` normalizes the state (uniform superposition over all pixels)

**Total qubits:** `2n + q` (e.g., 14 for an 8×8 image with 8-bit color)

### Why is NEQR Lossless?

All registers use **basis encoding** (computational basis states only):
- No floating-point amplitudes that could be truncated or rounded.
- Each pixel value `f(y,x)` maps to a unique basis state `|f(y,x)⟩`.
- Measuring the state in the standard basis recovers the exact value.

This gives:
- `PSNR = ∞` (zero reconstruction error)
- `SSIM = 1.0` (perfect structural similarity)
- `MSE = 0.0` (zero mean squared error)

### Circuit Construction

**Step 1 — Create uniform superposition over all pixel positions:**
```
H⊗n on all 2n position qubits:
|0⟩^{2n} → (1/2ⁿ) Σ_{y,x} |y⟩|x⟩
```

**Step 2 — Encode pixel intensities:**
For each pixel (y, x) and each color bit i where `f(y,x)` has bit i = 1:
```
MultiControlledX on color qubit i,
  controlled on position register = |y⟩|x⟩
```

This flips color bit i exactly when the quantum registers encode position (y,x).

### Code Walkthrough: `quantum_seg/neqr.py`

```python
# neqr.py — key functions

def neqr_ops(img, q=8):
    """Apply NEQR preparation gates inside a QNode.
    
    img: (2^n, 2^n) uint8 grayscale array
    q:   number of color qubits (default 8 for 8-bit images)
    """
    n = image_dims(img.shape[0])       # n = log2(side length)
    
    # Step 1: Hadamard superposition on all 2n position qubits
    for wire in range(2 * n):
        qml.Hadamard(wires=wire)
    
    # Step 2: For each pixel, encode color bits
    for y in range(2**n):
        for x in range(2**n):
            pixel = int(img[y, x])
            for bit in range(q):       # for each color qubit
                if (pixel >> bit) & 1: # if this bit is 1
                    # Multi-controlled X: flip color bit when position = (y,x)
                    controls = _position_controls(y, x, n)
                    qml.MultiControlledX(wires=controls + [2*n + bit],
                                         control_values=_control_vals(y, x, n))
```

```python
def decode_state(state, n, q):
    """Recover image from NEQR statevector.
    
    For each position (y,x):
      - The coefficient at index encoding (f, y, x) should be ±1/2^n
      - Extract f from which color basis state has non-zero amplitude
    """
    side = 2**n
    img = np.zeros((side, side), np.uint8)
    for y in range(side):
        for x in range(side):
            pos_idx = y * side + x          # position index in 2n-qubit register
            # Find which color state has non-zero amplitude
            for color in range(2**q):
                state_idx = (color << (2*n)) | (y << n) | x
                if abs(state[state_idx]) > 1e-10:
                    img[y, x] = color
                    break
    return img
```

### Resource Report

| Image size | Qubits | Gates | Depth | Sim time |
|-----------|--------|-------|-------|----------|
| 8×8       | 14     | ~302  | ~297  | 0.05s    |
| 16×16     | 16     | ~1246 | ~1239 | 0.10s    |
| 32×32     | 18     | ~4991 | ~4982 | 0.49s    |

Gate count scales as `O(2^{2n} × q)` — exponential. NEQR is practical only
for small patches on a classical simulator.

---

## Part C — Stage 2: Quantum Hadamard Edge Detection (QHED)

### Theory: 1D QHED

Given a row of N = 2ⁿ pixels `c = (c₀, c₁, ..., c_{N-1})`, normalize to unit
norm: `ĉ = c / ||c||`. Amplitude-encode into n qubits:

```
|ψ⟩ = Σᵢ ĉᵢ |i⟩
```

Apply Hadamard to the **Least Significant Bit (LSB)** qubit. The Hadamard
transforms each adjacent pair:

```
H on LSB:  (c_{2i}, c_{2i+1}) → ((c_{2i}+c_{2i+1})/√2, (c_{2i}-c_{2i+1})/√2)
```

The odd-indexed output amplitudes are the **forward differences**:
```
odd output at index 2i+1 = (c_{2i} - c_{2i+1}) / √2 ≈ gradient at pixel i
```

### Two-Pass Recovery

One H-on-LSB pass gives gradients at even positions {0, 2, 4, ...}.
A second pass on the cyclically-shifted signal gives gradients at odd positions
{1, 3, 5, ...}. Together, all N gradients are recovered.

### 2D QHED

```
horizontal gradient:  apply QHED to each row → g_h[y, x]
vertical gradient:    apply QHED to each col → g_v[y, x]
edge magnitude:       E[y,x] = sqrt(g_h[y,x]² + g_v[y,x]²)
normalize:            E → E / max(E)
```

### Bilateral Denoising Refinement

Before QHED, apply OpenCV bilateral filter (σ_space=5, σ_color=40):
- Smooths flat regions (reduces false-edge noise)
- Preserves sharp boundaries (edge-preserving filter)

Result: ODS lifts from 0.539 (raw QHED) to 0.565 (refined QHED).

### Code Walkthrough: `quantum_seg/qsobel.py`

```python
def _hadamard_on_lsb(vec):
    """Apply H to LSB of amplitude-encoded vec. Core quantum step."""
    n = int(round(np.log2(len(vec))))
    dev = qml.device("lightning.qubit", wires=n)

    @qml.qnode(dev)
    def circ():
        qml.StatePrep(vec, wires=range(n))    # amplitude-encode the row
        qml.Hadamard(wires=n - 1)              # H on LSB (last wire = LSB in MSB-first)
        return qml.state()

    return np.real_if_close(np.asarray(circ()))


def qhed_1d(signal, signed=False):
    """Forward-difference gradient of a 1D signal via QHED."""
    sig = np.asarray(signal, float)
    norm = np.linalg.norm(sig)
    c = sig / norm                              # normalize

    # Pass A: gradients at even positions
    sA = _hadamard_on_lsb(c)
    diffA = sA[1::2] * np.sqrt(2.0) * norm     # odd indices = gradients, rescale

    # Pass B: gradients at odd positions (shift by 1)
    cB = np.roll(c, -1)
    sB = _hadamard_on_lsb(cB)
    diffB = sB[1::2] * np.sqrt(2.0) * norm

    grad = np.empty(len(sig))
    grad[0::2] = diffA                          # even-position gradients
    grad[1::2] = diffB                          # odd-position gradients
    grad[-1] = 0.0                              # no wraparound at border
    return grad if signed else np.abs(grad)


def qhed_edges(image, combine="l2"):
    """Full 2D QHED edge map. Requires power-of-two dimensions."""
    img = np.asarray(image, float)
    h, w = img.shape
    gh = np.stack([qhed_1d(img[r, :]) for r in range(h)])   # row-wise
    gv = np.stack([qhed_1d(img[:, c]) for c in range(w)], axis=1)  # col-wise
    mag = np.sqrt(gh**2 + gv**2)
    return mag / mag.max() if mag.max() > 0 else mag
```

### Honest Caveat

The quantum speedup is in the **Hadamard step** (1 gate, O(1) time).
However, `qml.StatePrep(vec, ...)` encodes the amplitude vector classically,
requiring O(N) operations. This bottleneck means current QHED has no
computational advantage over classical Sobel — the value is the proof of
concept and the mathematical equivalence, not speed.

---

## Part D — Stage 4: QUBO Construction

### The Segmentation Problem as QUBO

Assign each superpixel `i` a binary label `zᵢ ∈ {0=bg, 1=fg}` to minimize:

```
E(z) = Σᵢ hᵢ·zᵢ  +  Σ_{(i,j)∈edges} Jᵢⱼ·(zᵢ - zⱼ)²
       ────────────   ─────────────────────────────────
       unary term      pairwise smoothness term
```

**Unary term `hᵢ`**: how much does superpixel i prefer background?
```
hᵢ = log P(cᵢ | BG) − log P(cᵢ | FG)
     ──────────────────────────────────
     from GMM fitted on seed pixels
```
- `hᵢ > 0` → bg-likely (z=0 reduces cost)
- `hᵢ < 0` → fg-likely (z=1 reduces cost)

**Pairwise term `Jᵢⱼ`**: how strongly should neighbors match?
```
Jᵢⱼ = β · exp(−‖cᵢ − cⱼ‖² / σ²) · (1 − qhed_ij)
       ─────────────────────────     ──────────────
       color similarity              QHED edge factor
```
- Similar colors → large J → penalize label mismatch → smooth boundary
- Strong QHED edge → `(1 − qhed_ij) ≈ 0` → small J → allow cut here

### Seed Hard Constraints

Instead of soft constraints, remove seed superpixels from the circuit entirely.
Fold their fixed labels into the effective unary terms of their free neighbors:

**For a fg seed k (zₖ = 1) with free neighbor j:**
```
(zₖ − zⱼ)² = (1 − zⱼ)  →  Δhⱼ += −Jₖⱼ
```

**For a bg seed k (zₖ = 0) with free neighbor j:**
```
(zₖ − zⱼ)² = zⱼ  →  Δhⱼ += +Jₖⱼ
```

After this reduction, only `n_free` (non-seed) qubits enter the QAOA circuit.

### QUBO → Ising Conversion

QAOA operates in the Ising model: spin variables `sᵢ ∈ {−1, +1}` instead
of `zᵢ ∈ {0, 1}`. The substitution `zᵢ = (1 − sᵢ)/2` converts:

```
h_ising[i]   = −h_eff[i] / 2        (coefficient of Zᵢ in H_C)
J_ising[i,j] = −J[i,j]  / 2        (coefficient of Zᵢ⊗Zⱼ in H_C)
```

Minimizing `⟨H_C⟩` in the Ising picture is equivalent to minimizing `E(z)` in
the QUBO picture.

**Sign convention check:**
- `|0⟩` state → Pauli-Z eigenvalue = +1 → sᵢ = +1 → zᵢ = 0 → **background**
- `|1⟩` state → Pauli-Z eigenvalue = −1 → sᵢ = −1 → zᵢ = 1 → **foreground**

For a fg-likely superpixel (`hᵢ < 0` in QUBO):
- `h_ising = −hᵢ/2 > 0`
- Cost `h_ising·Zᵢ` is minimized when `Zᵢ = −1` (state `|1⟩`) → zᵢ = 1 = fg ✓

### Code Walkthrough: `quantum_seg/superpixel.py`

```python
# Complete pipeline for one image:

labels, n_sp   = slic_segments(img_bgr, n_segments=12)
#                 └── cv2.ximgproc SLIC: pixel labels + count

img_lab        = cv2.cvtColor(img_bgr/255.0, cv2.COLOR_BGR2Lab)
sp_colors      = sp_mean_colors(img_lab, labels, n_sp)
#                 └── mean Lab color per superpixel

edges          = sp_adjacency(labels, n_sp)
#                 └── list of (i,j) pairs for 4-connected adjacent SPs

qhed_str       = sp_qhed_edge_strength(labels, edge_map, edges)
#                 └── {(i,j): mean QHED value on shared boundary}

sp_type        = classify_seeds(labels, trimap, n_sp)
#                 └── majority vote: 1=fg_seed, 0=bg_seed, -1=free

gmm_fg         = fit_gmm(img_lab[trimap == 255], n_components=3)
gmm_bg         = fit_gmm(img_lab[trimap == 0  ], n_components=3)

h              = compute_unary(sp_colors, gmm_fg, gmm_bg)
#                 └── h[i] = log P(c_i|BG) - log P(c_i|FG)

J              = compute_pairwise(sp_colors, edges, qhed_str, beta=10.0)
#                 └── J[(i,j)] = β·exp(-dist²/σ²)·(1-qhed)

h_eff, J_free, free_nodes = build_reduced_qubo(h, J, sp_type, n_sp)
#                 └── fold seed constraints, extract free-node QUBO

h_ising, J_ising = qubo_to_ising(h_eff, J_free)
#                 └── h_ising[i] = -h_eff[i]/2
```

---

## Part E — Stage 4: QAOA Circuit and Optimization

### Cost Hamiltonian

```
H_C = Σᵢ h_ising[i] · Zᵢ  +  Σ_{i<j} J_ising[i,j] · Zᵢ⊗Zⱼ
```

In PennyLane:
```python
coeffs = [h_ising[0], h_ising[1], ..., J_ising[(0,1)], ...]
ops    = [qml.PauliZ(0), qml.PauliZ(1), ...,
          qml.PauliZ(0) @ qml.PauliZ(1), ...]
H_C    = qml.Hamiltonian(coeffs, ops)
```

### Mixer Hamiltonian

```
H_M = Σᵢ Xᵢ     (standard X-mixer)
```

In PennyLane: `H_M = qml.qaoa.x_mixer(wires)`

### QAOA Ansatz (depth p = 1)

```
|ψ(γ, β)⟩ = U_M(β) · U_C(γ) · |+⟩^n

|+⟩^n:   apply H to every qubit

U_C(γ):  exp(−i·γ·H_C)
  → For each Zᵢ term:        RZ(2γ·h_i, wire=i)
  → For each Zᵢ⊗Zⱼ term:    CNOT(i,j) → RZ(2γ·J_ij, j) → CNOT(i,j)

U_M(β):  exp(−i·β·H_M)
  → For each Xᵢ term:        RX(2β, wire=i)
```

### COBYLA Optimization

```python
def cost_fn(params):
    return float(cost_qnode(np.asarray(params)))

result = minimize(cost_fn, x0=init_params,
                  method='COBYLA',
                  options={'maxiter': 200, 'rhobeg': 0.5})
```

- **COBYLA** = Constrained Optimization BY Linear Approximations
- Gradient-free — uses only function evaluations, no backpropagation
- Well-suited for noisy quantum circuits (where gradients vanish or are costly)
- 5 random restarts with different initial `(γ₀, β₀)` — best result kept

### Bitstring Decoding

```python
probs = prob_qnode(best_params)     # shape (2^n_free,)
best_idx = np.argmax(probs)         # index of most-probable basis state

# Decode: wire 0 = MSB of best_idx
for i in range(n_free):
    z_free[i] = (best_idx >> (n_free - 1 - i)) & 1
    # z=0 → bg,  z=1 → fg
```

### Pixel Mask Reconstruction

```python
sp_label = np.zeros(n_sp, np.uint8)

# Assign seed labels
sp_label[sp_type == 1] = 1    # fg seeds → fg
sp_label[sp_type == 0] = 0    # bg seeds → bg

# Assign QAOA labels to free superpixels
for local_idx, global_idx in enumerate(free_nodes):
    sp_label[global_idx] = z_free[local_idx]

# Each pixel inherits its superpixel's label
mask = sp_label[labels]        # shape (H, W)
```

---

## Part F — Auto-Trimap Generation

For misc images without human annotations, we generate a synthetic trimap:

```python
def auto_trimap(H, W, fg_radius_frac=0.28, bg_margin_frac=0.08):
    trimap = np.full((H, W), 128, np.uint8)     # all unknown

    # Background seeds: outer border (8% margin)
    mh, mw = int(H * 0.08), int(W * 0.08)
    trimap[:mh, :] = 0
    trimap[-mh:, :] = 0
    trimap[:, :mw] = 0
    trimap[:, -mw:] = 0

    # Foreground seeds: center ellipse (28% radius)
    cy, cx = H // 2, W // 2
    ry, rx = int(H * 0.28), int(W * 0.28)
    cv2.ellipse(trimap, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    return trimap
    # Values: 255=fg seed, 0=bg seed, 128=unknown (→ QAOA)
```

The assumption is that natural images tend to have the subject centered.
This works well for portrait, object, and scene images; less well for aerial
or texture images.

### Running the Demo: `segment_misc.py`

```python
# Complete flow for each image in misc/:

img_bgr = _load_bgr(path)               # load + convert to BGR, resize to SIZE
H, W    = img_bgr.shape[:2]
trimap  = auto_trimap(H, W)             # generate center/border seeds

result  = segment_image(                # full QAOA pipeline
    img_bgr, trimap,
    n_segments=12,
    p=1,
    use_qhed=True,
    maxiter=150,
)

gc_mask = grabcut_baseline(img_bgr, trimap)   # classical GrabCut

misc_fgbg_panel(                        # save 5-panel figure
    name, img_bgr, trimap,
    result["mask"], gc_mask,
    qhed_map=result["qhed_map"],
)
```

---

## Part G — How to Interpret Results

### PSNR = ∞

Peak Signal-to-Noise Ratio is defined as:
```
PSNR = 10 · log₁₀(MAX² / MSE)
```

When MSE = 0 (perfect reconstruction), PSNR → ∞. This confirms NEQR
is lossless — **not** an approximation. A finite PSNR (e.g., 30 dB) would
indicate lossy encoding; ∞ means exactly zero reconstruction error.

### ODS and OIS (BSDS500 Edge Metrics)

Both measure the F-score `F = 2·P·R / (P+R)` between predicted and ground-truth
edge maps, with a 2-pixel matching tolerance:

- **ODS (Optimal Dataset Scale)**: threshold that maximizes F averaged over all images — one global threshold for the entire dataset.
- **OIS (Optimal Image Scale)**: for each image, independently find the best threshold, then average — measures per-image potential.

OIS ≥ ODS always. A large gap means the method is sensitive to threshold choice.

Published anchors for context:
| Method | ODS |
|--------|-----|
| Human  | 0.80 |
| HED (deep learning) | 0.79 |
| gPb    | 0.73 |
| Sobel (literature) | 0.54 |
| QHED (our result) | 0.539 |
| Canny  | 0.50 |

### IoU* and Proxy GT

For misc images without human annotations, we cannot compute real IoU.
Instead, we use the center-ellipse fg region of the auto-trimap as a
**proxy ground truth** and compute IoU* on the **unknown pixels only**.

This means:
- `IoU* = IoU(predicted_mask[unknown], proxy_fg[unknown])`
- The values are only meaningful as a **relative comparison** between QAOA and GrabCut
- Do NOT compare these numbers to dataset benchmarks (GrabCut-50, PASCAL VOC)

### Qubit Scaling

| Configuration | Qubits | Sim feasibility |
|---------------|--------|----------------|
| 8–15 qubits | n_free ≤ 15 | lightning.qubit: seconds |
| 16–20 qubits | 16–20 | lightning.qubit: minutes |
| 21–25 qubits | 21–25 | lightning.qubit: hours |
| 50+ qubits | near-term hardware | IBM Eagle (127 qubits), IonQ Aria |
| 1000+ qubits | fault-tolerant hardware | Future quantum computers |

With `n_segments = 12–16` and typical trimaps removing 30–50% of superpixels
as seeds, practical `n_free ≈ 8–12`, well within the ~22s/image performance
observed in our experiments.

---

## Quick Reference: File-to-Function Map

| Task | File | Key function |
|------|------|-------------|
| Encode image as quantum state | `neqr.py` | `neqr_ops()`, `roundtrip()` |
| QHED 2D edge map | `qsobel.py` | `qhed_edges()`, `qhed_edges_refined()` |
| SLIC superpixels | `superpixel.py` | `slic_segments()` |
| QUBO construction | `superpixel.py` | `compute_unary()`, `compute_pairwise()` |
| Seed constraints | `superpixel.py` | `build_reduced_qubo()` |
| QUBO→Ising | `superpixel.py` | `qubo_to_ising()` |
| QAOA circuit | `qaoa_seg.py` | `make_qaoa_circuit()` |
| QAOA optimization | `qaoa_seg.py` | `optimize_qaoa()` |
| Full pipeline | `qaoa_seg.py` | `segment_image()` |
| Classical baseline | `baselines.py` | `grabcut_baseline()` |
| Auto-trimap | `superpixel.py` | `auto_trimap()` |
| Run all misc images | `segment_misc.py` | `main()` |
| Evaluation metrics | `metrics.py` | `segmentation_report()`, `dataset_ods_ois()` |
| Figures | `figures.py` | `misc_fgbg_panel()`, `bsds_panel()` |
