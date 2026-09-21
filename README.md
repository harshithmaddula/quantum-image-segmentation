# Quantum Image Segmentation

A four-stage quantum image processing pipeline built with [PennyLane](https://pennylane.ai) that combines lossless quantum image encoding, quantum edge detection, and QAOA-based foreground/background segmentation.

---

## Overview

Classical image segmentation methods such as GrabCut and deep-learning approaches (U-Net, SAM) operate entirely in the classical domain. This project demonstrates an end-to-end **quantum-enhanced pipeline** where each stage leverages quantum circuits — from the representation layer through to the combinatorial optimization of region labeling — while maintaining honest benchmarks against classical baselines.

**Key contributions:**
- **NEQR lossless encoding** — basis-encoded quantum image representation with provably perfect reconstruction (PSNR = ∞, SSIM = 1.0)
- **QHED scalable edge detection** — quantum Hadamard circuit computing adjacent-pixel gradients in O(log N) qubits; benchmarked on BSDS500
- **QAOA foreground/background segmentation** — superpixel-level binary labeling via QUBO/Ising optimization with QHED-modulated pairwise terms
- **Honest resource accounting** — every quantum claim is accompanied by gate counts, qubit counts, and the Ruan (2021) state-preparation caveat

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT IMAGE                                                     │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐
│  STAGE 1 · NEQR Encoding │  Basis-encode image into quantum state
│  Qubits: 2n + q          │  |I⟩ = (1/2ⁿ) Σ |f(y,x)⟩⊗|y⟩|x⟩
│  Result: PSNR = ∞        │  Proves spatial information preserved
└──────────────┬───────────┘
               │ (lossless)
               ▼
┌─────────────────────────────┐
│  STAGE 2 · QHED Edge Map    │  Amplitude-encode rows/cols → H on LSB
│  Qubits: log₂(N) per row    │  gradient = c_{2i} - c_{2i+1}
│  Result: ODS 0.539 ≈ Sobel  │  Bilateral denoise → refined edge map
└──────────────┬──────────────┘
               │ (edge map used as feature in Stage 4)
               ▼
┌─────────────────────────────────────────┐
│  STAGE 4 · QUBO + QAOA Fg/Bg           │
│                                         │
│  Image + trimap seeds                   │
│    → SLIC superpixels (n ≈ 12–16)       │
│    → Unary h_i   = log P(BG) - log P(FG)│  (GMM on seed pixels)
│    → Pairwise J_ij = β·exp(...)·(1-qhed)│  (QHED-modulated)
│    → Seed hard constraints              │
│    → QUBO → Ising → QAOA circuit        │
│    → COBYLA optimize ⟨H_C⟩              │
│    → Decode bitstring → pixel mask      │
│  Qubits: n_free ≤ ~12 (simulation)     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  OUTPUT: binary fg/bg mask   │
│  + 5-panel comparison figure │
└──────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```powershell
pip install opencv-contrib-python pennylane pennylane-lightning numpy scipy Pillow matplotlib scikit-learn
```

> **Important:** Use `opencv-contrib-python`, **not** `opencv-python`. The `cv2.ximgproc` module (SLIC superpixels) is only in the contrib package.

### 2. Run the self-tests (no dataset needed)

```powershell
python -m quantum_seg.neqr        # NEQR lossless encode/decode
python -m quantum_seg.qsobel      # QHED vs classical Sobel correlation
python -m quantum_seg.superpixel  # QUBO construction on synthetic image
python -m quantum_seg.qaoa_seg    # Full QAOA pipeline on synthetic circle (~10s)
```

### 3. Segment all misc images (fg/bg)

```powershell
python segment_misc.py
```

Outputs 5-panel figures to `results/figures/misc_fgbg_*.png` (~8–22 s per image).

### 4. Run the full experiment suite

```powershell
# Stages 1 + 2 only (no download required)
python -m quantum_seg.run_experiments

# + Stage 3: BSDS500 supervised edge benchmarks (downloads dataset)
python -m quantum_seg.run_experiments --bsds --bsds-limit 30

# + Stage 5: QAOA fg/bg on misc with auto-trimap
python -m quantum_seg.run_experiments --misc-fgbg
```

Results written to `results/metrics_report.md`.

---

## Project Structure

```
Quantum_Image_Segmentation/
│
├── segment_misc.py            Standalone: run fg/bg on all misc/ images
├── requirements.txt           Python dependencies
├── README.md                  This file
├── TECHNICAL_GUIDE.md         Deep-dive: theory, math, code walkthrough
├── QA.md                      30 Q&A pairs for viva / presentation
│
├── quantum_seg/               Main Python package
│   ├── neqr.py                Stage 1 — NEQR lossless quantum encoding
│   ├── qsobel.py              Stage 2 — QHED edge detection + QSobel ref
│   ├── superpixel.py          Stage 4 — SLIC superpixels + QUBO construction
│   ├── qaoa_seg.py            Stage 4 — QAOA circuit, optimizer, full pipeline
│   ├── baselines.py           Classical: Sobel, Scharr, Laplacian, Canny, GrabCut
│   ├── metrics.py             ODS/OIS, Pratt FOM, Hausdorff, IoU, Dice, PSNR, SSIM
│   ├── data.py                Loaders: misc (USC-SIPI), BSDS500, GrabCut-50
│   ├── figures.py             Visualization panels (misc, BSDS, grabcut, misc_fgbg)
│   └── run_experiments.py     5-stage experiment driver + CLI
│
├── misc/                      USC-SIPI miscellaneous image database (39 TIFFs)
├── data/                      Downloaded datasets (BSDS500, GrabCut-50)
├── results/
│   ├── metrics_report.md      Auto-generated experiment results table
│   └── figures/               PNG output figures
└── paper/
    ├── paper.tex              IEEE Access LaTeX paper
    └── references.bib         Bibliography
```

---

## Stages and Results

| Stage | Method | Dataset | Key Metric |
|-------|--------|---------|------------|
| 1 — NEQR Encoding | Basis-encoded quantum state | USC-SIPI misc (8–32 px crops) | PSNR = ∞, SSIM = 1.0, MSE = 0 |
| 2 — QHED Edge Detection | Quantum Hadamard + bilateral denoise | USC-SIPI misc (128×128) | 86–96% tol-1px agreement with Sobel |
| 3 — BSDS500 Boundary | QHED-raw vs classical | BSDS500 val (256×256) | ODS 0.539 (lit. Sobel: 0.54) |
| 4 — QAOA Fg/Bg | QAOA + QUBO graph cut | Misc images (128×128) | n_free ≈ 11 qubits; ~8–22 s/image |
| 5 — Classical baseline | OpenCV GrabCut | Same as Stage 4 | Side-by-side comparison in figures |

---

## Output Figures

Each run of `segment_misc.py` or `--misc-fgbg` produces a **5-panel PNG** per image:

```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ Original │ Trimap   │ QHED     │  QAOA    │ GrabCut  │
│ image    │ G=fg     │ edges    │  mask    │ mask     │
│          │ R=bg     │ (quantum)│ (quantum)│(classical│
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

Saved to: `results/figures/misc_fgbg_<imagename>.png`

---

## Quantum Limitations (Honest)

This project follows the honest-caveat convention established by Ruan (2021):

- **NEQR gate count** scales as O(2^{2n} · q) — practical only for patches ≤ 32×32 on a simulator.
- **QHED state preparation** is the O(N) bottleneck; the quantum speedup is in the single Hadamard step, not the full circuit.
- **QAOA qubit count** = number of free superpixels ≈ 8–12 in typical runs. The `lightning.qubit` simulator is tractable to ~25 qubits. Full-image QAOA (one qubit per pixel) requires fault-tolerant quantum hardware.
- All experiments run on classical simulation (`lightning.qubit`), not on physical quantum hardware.

---

## Citation

If you use this work, please cite:

```bibtex
@article{quantum_seg_2024,
  title   = {A Four-Stage Quantum Image Segmentation Framework},
  author  = {[Author]},
  journal = {IEEE Access},
  year    = {2024}
}
```

---

## References

1. Y. Zhang et al., "NEQR: A Novel Enhanced Quantum Representation of Digital Images," *Quantum Inf. Process.*, 2013.
2. X.-H. Yao et al., "Quantum Image Processing and Its Application to Edge Detection," *IEEE Trans. Cybern.*, 2017.
3. E. Farhi, J. Goldstone, S. Gutmann, "A Quantum Approximate Optimization Algorithm," *arXiv:1411.4028*, 2014.
4. C. Rother, V. Kolmogorov, A. Blake, "'GrabCut': Interactive Foreground Extraction," *ACM SIGGRAPH*, 2004.
5. Y. Ruan, "Quantum Image Processing: Opportunities and Challenges," *Math. Probl. Eng.*, 2021.
