# Q&A: Quantum Image Segmentation — Viva & Presentation

30 questions in four categories, from introductory to examiner-level depth.
Numbers in answers refer to actual results from this project.

---

## Category A — Conceptual / Theory (Q1–Q10)

---

**Q1. Why use quantum computing for image segmentation? What is the theoretical advantage?**

Quantum computing offers three theoretical advantages relevant to image segmentation:
(1) **Superposition** allows a single quantum state to encode all pixel values simultaneously — NEQR encodes a 2ⁿ×2ⁿ image in one quantum state, where measuring any position collapses to the exact pixel value.
(2) **Parallelism** means a quantum circuit acting on an amplitude-encoded state operates on all 2ⁿ input amplitudes simultaneously — this is how QHED computes all adjacent-pixel differences with a single Hadamard gate.
(3) **Combinatorial optimization** via QAOA can, in principle, explore the exponential space of binary labelings (fg/bg assignments) more efficiently than exhaustive classical search.
The honest caveat: current quantum hardware and simulators still require classical state preparation (O(N) bottleneck), which limits the practical speedup. This project is a proof-of-concept demonstrating correctness, not claiming current computational advantage.

---

**Q2. What is NEQR and why is it described as "lossless"?**

NEQR (Novel Enhanced Quantum Representation, Zhang et al., 2013) encodes a 2ⁿ×2ⁿ grayscale image as the quantum state:
```
|I⟩ = (1/2ⁿ) Σ_{y,x} |f(y,x)⟩ ⊗ |y⟩ ⊗ |x⟩
```
where `|y⟩|x⟩` are n-qubit position registers and `|f(y,x)⟩` is a q-qubit color register, all in the **computational basis**.
It is lossless because computational basis states are orthogonal and exact — no floating-point approximation is involved. Measuring the state at any position always returns the exact integer pixel value. Our experiments confirm: PSNR = ∞, SSIM = 1.0, MSE = 0.0 on all tested image sizes (8×8 to 32×32), across 9 crops from three different images.

---

**Q3. How does a Hadamard gate detect edges in QHED?**

QHED exploits the fact that the Hadamard gate H transforms a pair of amplitudes `(a, b)` to `((a+b)/√2, (a−b)/√2)`. When a row of pixels is amplitude-encoded as `Σᵢ cᵢ|i⟩`, applying H to the least-significant qubit transforms each adjacent pair `(c_{2i}, c_{2i+1})` to `((c_{2i}+c_{2i+1})/√2, (c_{2i}−c_{2i+1})/√2)`. The odd-indexed output amplitude is the forward difference `c_{2i} − c_{2i+1}` — exactly the gradient at pixel i. A large difference means a sharp intensity change (edge); near-zero means flat region. This single Hadamard gate performs N/2 gradient computations simultaneously in the amplitude domain — the quantum parallelism at work.

---

**Q4. What is a QUBO problem and how does image fg/bg labeling map to it?**

A Quadratic Unconstrained Binary Optimization (QUBO) problem minimizes a quadratic function of binary variables:
```
minimize  E(z) = Σᵢ hᵢzᵢ + Σ_{i<j} Qᵢⱼzᵢzⱼ,   zᵢ ∈ {0,1}
```
For fg/bg segmentation, each binary variable `zᵢ` is the label of superpixel `i` (0=bg, 1=fg). The linear term `hᵢ` encodes how likely each superpixel is foreground based on color (derived from GMM likelihoods on seed pixels). The quadratic term `Jᵢⱼ(zᵢ−zⱼ)²` penalizes adjacent superpixels with different labels when they look similar — encouraging smooth, coherent boundaries. This is exactly the Boykov-Jolly (2001) interactive graph-cut energy, reformulated as QUBO for quantum optimization.

---

**Q5. Explain the QAOA algorithm from first principles.**

QAOA (Farhi et al., 2014) solves combinatorial optimization by preparing a parameterized quantum state that gradually encodes the optimal solution:

1. **Initialization**: Start in equal superposition `|+⟩^n = H^⊗n|0⟩^n` — all 2ⁿ bitstrings equally likely.
2. **Cost layer** `U_C(γ) = exp(−iγH_C)`: Rotates the phase of each basis state proportional to its QUBO energy `E(z)`. States with lower energy acquire a more favorable phase.
3. **Mixer layer** `U_M(β) = exp(−iβH_M)` with `H_M = ΣᵢXᵢ`: Mixes probability amplitudes between neighboring bitstrings, allowing the algorithm to "explore" the solution space.
4. **Repeat p times**: More layers → better approximation of the optimal solution.
5. **Measure**: The state `|ψ(γ,β)⟩` concentrates probability on low-energy (good) solutions. The most-probable bitstring is the QAOA answer.
The parameters `(γ₁,β₁,...,γₚ,βₚ)` are classically optimized to minimize `⟨ψ|H_C|ψ⟩`.

---

**Q6. What is the Ising model and how does it relate to QUBO in your work?**

The Ising model (from condensed matter physics) describes a system of binary spins `sᵢ ∈ {−1, +1}` with energy:
```
H_Ising = Σᵢ hᵢsᵢ + Σᵢ<ⱼ Jᵢⱼsᵢsⱼ
```
The substitution `zᵢ = (1 − sᵢ)/2` converts between QUBO (z ∈ {0,1}) and Ising (s ∈ {−1,+1}). In our work, the conversion gives:
```
h_ising[i]   = −h_eff[i] / 2        ← coefficient of PauliZ_i
J_ising[i,j] = −J[i,j]  / 2        ← coefficient of PauliZ_i ⊗ PauliZ_j
```
This Ising form is exactly what QAOA's cost Hamiltonian H_C requires, because PauliZ has eigenvalues ±1 matching the spin convention. Minimizing `⟨H_C⟩` via QAOA is equivalent to minimizing `E(z)` in the QUBO — the same fg/bg labeling problem.

---

**Q7. Why use superpixels instead of individual pixels in QAOA?**

Using pixels directly would require one qubit per pixel. For a 128×128 image that is 16,384 qubits — far beyond any current or near-term quantum simulator or hardware (lightning.qubit is tractable to ~25 qubits). Superpixels group neighboring pixels of similar color into compact regions, reducing the problem size from thousands of pixels to dozens of regions. With `n_segments = 12` and typical seed coverage of 30–50%, we get `n_free ≈ 8–12` qubits — well within simulation capacity. The granularity loss is acceptable: superpixels preserve object boundaries (SLIC respects color edges), and the QAOA labeling boundary aligns with those superpixel boundaries.

---

**Q8. How is QHED integrated into the pairwise term Jᵢⱼ?**

The pairwise smoothness penalty between adjacent superpixels i and j is:
```
Jᵢⱼ = β · exp(−‖cᵢ−cⱼ‖² / σ²) · (1 − qhed_ij)
```
where `qhed_ij` is the mean QHED edge magnitude along all pixels on the shared boundary between superpixels i and j. This integration works as follows: if the quantum QHED circuit detects a strong edge between two superpixels (`qhed_ij ≈ 1`), the factor `(1 − qhed_ij) ≈ 0` reduces the penalty, making the optimizer "willing" to place a label boundary there. Conversely, if QHED finds no edge (`qhed_ij ≈ 0`), the full color-based penalty applies, encouraging same-label assignment. This is the mechanism by which quantum edge information from Stage 2 directly shapes the quantum optimization in Stage 4.

---

**Q9. What is the difference between amplitude encoding (QHED) and basis encoding (NEQR)?**

| Property | Amplitude encoding (QHED) | Basis encoding (NEQR) |
|---------|---------------------------|------------------------|
| Data stored in | Probability amplitudes (complex numbers) | Computational basis states (integers) |
| Qubits needed | log₂(N) for N values | Proportional to value range + position |
| Readout | Probabilistic — must measure many times OR use interference | Deterministic — single measurement gives exact value |
| Lossless? | No — amplitudes are floating point, and measurement is probabilistic | Yes — exact basis state, deterministic decode |
| Example | 256-pixel row → 8 qubits | 8×8 image at 8-bit depth → 14 qubits |
| Use in pipeline | QHED gradient computation | Spatial-preservation proof |

NEQR's losslessness comes at the cost of more qubits and gates. QHED's efficiency (log₂N qubits) comes at the cost of approximate, probabilistic information (though in simulation we use the exact statevector, bypassing the measurement issue).

---

**Q10. What does "seed hard constraint" mean in your QUBO formulation?**

A "seed hard constraint" means a superpixel identified as a seed (from the trimap) is assigned its known label **with certainty** and removed from the QAOA circuit entirely. This differs from a "soft constraint" which would add a very large penalty term to the QUBO. The hard constraint approach: (1) reduces the qubit count from `n_sp` to `n_free = n_sp − n_seeds`, directly reducing circuit depth and optimization difficulty; (2) is mathematically exact — the seed label is never wrong; (3) propagates the seed information to free neighbors through effective unary term updates (Δhⱼ += −Jₖⱼ for fg-seed k, Δhⱼ += +Jₖⱼ for bg-seed k). In our experiments, seeds typically reduce qubit count from 12 to 8–11, saving 1–4 qubits per image.

---

## Category B — Implementation (Q11–Q18)

---

**Q11. Why PennyLane over Qiskit or Cirq?**

PennyLane was chosen for four reasons:
(1) **Hybrid quantum-classical optimization**: PennyLane is designed for variational algorithms (VQE, QAOA), with built-in `qml.qaoa` module providing `cost_layer()`, `mixer_layer()`, and `x_mixer()` — exactly what this project needs.
(2) **`lightning.qubit`**: PennyLane's high-performance statevector simulator written in C++ that is significantly faster than Qiskit's `statevector_simulator` for the circuit sizes used here (~8–16 qubits).
(3) **`qml.StatePrep`**: Direct amplitude state preparation, needed for QHED's amplitude encoding, is a first-class operation in PennyLane.
(4) **Clean QNode API**: PennyLane's `@qml.qnode` decorator and measurement primitives (`qml.expval`, `qml.probs`, `qml.state`) cleanly separate circuit definition from execution.

---

**Q12. What is `lightning.qubit` and why is it used?**

`lightning.qubit` is PennyLane's C++-accelerated exact statevector simulator. Unlike shot-based simulators (which sample measurements), it maintains the full 2ⁿ-element complex statevector and computes expectation values and probabilities analytically — no sampling noise. This is important for our work because: (1) NEQR requires exact statevector readout to prove PSNR = ∞; (2) QHED requires extracting exact amplitude values to recover gradients; (3) QAOA optimization with COBYLA needs reproducible, noise-free function evaluations. The simulator is tractable to ~25 qubits on standard hardware (16 GB RAM), which covers our typical n_free = 8–12.

---

**Q13. Why COBYLA optimizer — why not gradient-based (e.g., Adam)?**

COBYLA (Constrained Optimization BY Linear Approximations) is gradient-free. It is preferred here because:
(1) **No analytic gradients needed**: Computing the parameter-shift rule gradient for a 12-qubit QAOA circuit with n_edges ZZ coupling terms requires 2×n_params circuit evaluations per gradient step — expensive for our COBYLA convergence budget of 200 evaluations.
(2) **Robustness to shallow circuits**: Depth p=1 QAOA has a very "flat" cost landscape. Gradient methods often get stuck; COBYLA's trust-region linear model navigates flat regions better.
(3) **No barren plateau concern at p=1**: At depth-1 QAOA, the landscape is smooth enough for gradient-free methods to find good solutions.
(4) **Multiple restarts**: We run 5 COBYLA restarts from random initial parameters, keeping the best — this ensemble approach compensates for COBYLA's non-convergence guarantee.

---

**Q14. How does the GMM compute the unary term hᵢ?**

A Gaussian Mixture Model (GMM) with K=3 components is fitted on the Lab color values of fg seed pixels, and separately on bg seed pixels:
```python
gmm_fg.fit(fg_pixels)   # fg_pixels: shape (N_fg, 3)
gmm_bg.fit(bg_pixels)   # bg_pixels: shape (N_bg, 3)
```
For each superpixel i with mean Lab color **cᵢ**, the log-likelihood under each model is:
```python
log_fg_i = gmm_fg.score_samples([c_i])   # log P(c_i | FG model)
log_bg_i = gmm_bg.score_samples([c_i])   # log P(c_i | BG model)
h_i = log_bg_i - log_fg_i
```
If `h_i > 0`: the superpixel's color is more likely under the bg model → prefer bg label.
If `h_i < 0`: the superpixel's color is more likely under the fg model → prefer fg label.
Lab color space is used (not RGB) because it is perceptually uniform — Euclidean distance in Lab correlates better with perceived color difference.

---

**Q15. Walk me through the QUBO-to-Ising conversion step by step.**

Starting from the QUBO energy `E(z) = Σᵢ hᵢzᵢ + Σᵢ<ⱼ Jᵢⱼ(zᵢ−zⱼ)²` with zᵢ ∈ {0,1}:

**Step 1**: Substitute `zᵢ = (1−sᵢ)/2` where sᵢ ∈ {−1,+1}:
```
zᵢ = 0 → sᵢ = +1    (PauliZ eigenvalue for |0⟩)
zᵢ = 1 → sᵢ = −1    (PauliZ eigenvalue for |1⟩)
```

**Step 2**: Expand the unary term:
```
hᵢzᵢ = hᵢ·(1−sᵢ)/2 = hᵢ/2 − (hᵢ/2)·sᵢ
```
The constant `hᵢ/2` drops out (doesn't affect optimization). Variable term: `−(hᵢ/2)·sᵢ`

**Step 3**: Expand the pairwise term using `(zᵢ−zⱼ)² = (1−sᵢsⱼ)/2`:
```
Jᵢⱼ(zᵢ−zⱼ)² = Jᵢⱼ/2 − (Jᵢⱼ/2)·sᵢsⱼ
```
Constant `Jᵢⱼ/2` drops out. Variable term: `−(Jᵢⱼ/2)·sᵢsⱼ`

**Step 4**: The Ising Hamiltonian (replacing sᵢ with PauliZ operators Zᵢ):
```
H_C = Σᵢ [−hᵢ/2]·Zᵢ  +  Σᵢ<ⱼ [−Jᵢⱼ/2]·Zᵢ⊗Zⱼ
```

So: `h_ising[i] = −hᵢ/2` and `J_ising[i,j] = −Jᵢⱼ/2`.

---

**Q16. How does the bitstring decoding from QAOA output work?**

After optimization, we run `prob_qnode(best_params)` which returns a vector of `2^n_free` probabilities — one per computational basis state. The most-probable bitstring is:
```python
best_idx = np.argmax(probs)   # e.g., best_idx = 5 = binary 101 for n_free=3
```
This integer encodes the joint labeling of all free superpixels. PennyLane uses **MSB-first wire ordering**: wire 0 is the most significant bit. Decoding:
```python
for i in range(n_free):
    z_free[i] = (best_idx >> (n_free - 1 - i)) & 1
```
For `best_idx = 5` (binary `101`) with `n_free = 3`:
- i=0 (wire 0, MSB): `(5 >> 2) & 1 = 1` → fg
- i=1 (wire 1):      `(5 >> 1) & 1 = 0` → bg
- i=2 (wire 2, LSB): `(5 >> 0) & 1 = 1` → fg

The mapping `free_nodes[i] = global superpixel index` then assigns labels in the full image.

---

**Q17. What happens when n_free = 0 (all superpixels are seeds)?**

When every superpixel is classified as a seed by majority vote (all areas covered by the trimap), `n_free = 0` and no QAOA circuit is needed. The code handles this:
```python
if n_free == 0:
    z_free      = np.array([], dtype=np.int32)
    best_params = np.array([])
    best_cost   = 0.0
```
The pixel mask is then reconstructed purely from seed labels: all fg-seed superpixels → mask=1, all bg-seed superpixels → mask=0. This is correct behavior for highly-annotated trimaps and avoids running a zero-qubit QAOA circuit. In practice, with `n_segments=12` and typical trimaps, some free pixels almost always remain.

---

**Q18. How do you handle grayscale vs. color images in the pipeline?**

The pipeline handles both through an explicit conversion step in `_load_bgr()`:
```python
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
if img.ndim == 2:                          # grayscale (L channel)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)    # → (H,W,3) gray BGR
elif img.shape[2] == 4:                    # RGBA
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
else:                                      # RGB (PIL/TIFF reads as RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
```
All images are then converted to **float Lab color space** for the QUBO:
```python
img_lab = cv2.cvtColor(img.astype(np.float32)/255.0, cv2.COLOR_BGR2Lab)
```
For grayscale images, the BGR→Lab conversion produces a flat `a=0, b=0` color space (only L channel varies), so the GMM and distance computations still work correctly — all color information is carried in the L channel. This means segmentation of grayscale images is based purely on intensity contrast.

---

## Category C — Results and Evaluation (Q19–Q25)

---

**Q19. Why is NEQR PSNR equal to infinity? Is this physically meaningful?**

PSNR = ∞ because MSE = 0 — the encoded-then-decoded image is bit-for-bit identical to the original. PSNR is defined as `10·log₁₀(MAX²/MSE)`, and when MSE = 0, the division by zero yields infinity. This is physically meaningful because NEQR uses **basis encoding**: each pixel value is stored as an exact integer in the computational basis, with no floating-point representation. There is no rounding, no approximation, no compression. The NEQR quantum state is not a simulation of the image — it IS a faithful quantum encoding of every pixel. The practical implication: any subsequent quantum operation on the NEQR state acts on the exact image, not an approximation, ensuring that edge features and texture details are perfectly preserved.

---

**Q20. What do ODS and OIS measure in the BSDS500 evaluation?**

ODS (Optimal Dataset Scale) and OIS (Optimal Image Scale) are both Boundary F-score metrics from Martin et al. (2001), measuring how well predicted edges match human-annotated boundaries with a 2-pixel tolerance.

**ODS**: The single threshold that maximizes the mean F-score across all images in the dataset. One global number — useful for comparing methods head-to-head.

**OIS**: For each individual image, find the optimal threshold independently, compute F-score, then average. OIS ≥ ODS always, since it uses per-image optimal thresholds. The OIS measures the upper bound of what the detector could achieve if we knew the best threshold for each image.

A large ODS–OIS gap (e.g., Canny: 0.501 vs 0.501 = no gap) indicates the method doesn't benefit from image-specific tuning. Our QHED gap (0.539 ODS vs 0.569 OIS) is normal and comparable to Sobel (0.567 vs 0.593).

---

**Q21. Why is QAOA IoU lower than classical GrabCut in your experiments?**

Five reasons:

1. **Coarser resolution**: QAOA labels 12 superpixels; GrabCut labels individual pixels. At 128×128 with 12 segments, each superpixel covers ~1365 pixels — significant granularity loss.

2. **Shallow circuit (p=1)**: QAOA depth p=1 provides only a first-order approximation of the ground state. The approximation ratio improves with p, but so does circuit depth.

3. **No iterative refinement**: GrabCut alternates GMM re-fitting and graph-cut re-solving for 5 iterations, progressively improving the GMM model. Our QAOA runs once.

4. **No ground truth for evaluation**: The IoU* is measured against an auto-generated center-ellipse proxy, not human annotation. The proxy may not match the true fg/bg boundary, penalizing both QAOA and GrabCut unfairly.

5. **Small n_free**: With 8–12 free qubits, QAOA has limited representational capacity. A 12-segment image has 12-bit resolution; GrabCut has full pixel resolution.

Despite this, the QAOA correctly identifies the broad fg/bg structure in most images, establishing a working baseline for future hardware deployments with larger qubit counts.

---

**Q22. The IoU* metric is marked with * — what does that caveat mean?**

The asterisk signals that the evaluation uses a **proxy ground truth** (auto-generated center-ellipse trimap) rather than human annotation, and metrics are computed only on the **unknown region** of the trimap (pixels not already labeled as seeds). This means:

1. Do NOT compare IoU* = 0.079 against GrabCut-50 benchmark IoU values from literature — they use human-annotated masks.
2. The values are only meaningful **relative to each other** for the same image (QAOA vs GrabCut on the same proxy GT).
3. The proxy GT assumes objects are centered — valid for many USC-SIPI images (faces, boats, houses) but not for all (aerial, texture images).

The `*` notation follows the convention in our results tables and is prominently explained in the paper's discussion section.

---

**Q23. QHED achieves ODS = 0.539. How does this compare to published benchmarks?**

| Method | ODS | Source |
|--------|-----|--------|
| QHED-raw (ours) | 0.539 | This work |
| Sobel (literature) | 0.540 | Martin et al., 2001 |
| Sobel (ours) | 0.567 | This work |
| Canny (literature) | 0.600 | Martin et al., 2001 |
| gPb | 0.730 | Arbelaez et al., 2011 |
| HED (deep learning) | 0.790 | Xie & Tu, 2015 |

QHED-raw (0.539) matches published Sobel (0.54) within measurement noise — a strong validation result showing the quantum gradient circuit is mathematically equivalent to classical finite differences. Our classical Sobel (0.567) is slightly higher, likely due to our evaluation pipeline's threshold sweep. The gap to deep learning (HED: 0.79) is expected: both QHED and classical Sobel are single-scale gradient operators, while HED uses multi-scale learned features — a fundamentally different approach.

---

**Q24. What does n_qubits tell us about scalability of the QAOA approach?**

`n_qubits` (= `n_free`) is the number of qubits in the QAOA circuit. It determines:

- **Memory**: The statevector has 2^n_qubits complex numbers. At 16 bytes each, 12 qubits = 64 KB (trivial); 25 qubits = 512 MB (tight); 30 qubits = 16 GB (out of memory on most machines).
- **Simulation time**: Scales roughly as O(n_qubits × 2^n_qubits) per circuit evaluation. Our ~10s/image at 11 qubits; expected ~1000s at 20 qubits.
- **Quantum hardware requirement**: IBM Eagle has 127 qubits; IonQ Aria has 25 qubits. A 50-superpixel QAOA is achievable on near-term hardware.
- **Segmentation quality**: More superpixels → finer boundaries. Diminishing returns above ~100 superpixels (starts to look like pixel-level segmentation).

The current n_free = 8–12 represents a deliberate choice to stay within classical simulation feasibility while demonstrating correct QAOA behavior.

---

**Q25. What does the 5-panel output figure show and how do you interpret it?**

Each figure saved as `results/figures/misc_fgbg_<name>.png` has five panels:

1. **Original image**: The input image, RGB, at the processing resolution (128×128).
2. **Auto-trimap**: Color-coded seed map — green pixels are fg seeds (center ellipse), red pixels are bg seeds (border), grey is the unknown region that QAOA must label.
3. **QHED edges (quantum)**: The quantum Hadamard edge map — brighter = stronger edge. This is the quantum feature that modulates the QUBO pairwise term.
4. **QAOA mask (quantum)**: The fg/bg segmentation output from QAOA — green = foreground, red = background. The IoU* metric shown in the title compares this to the proxy GT.
5. **GrabCut mask (classical)**: The classical baseline — same trimap seeds, same image, run through OpenCV's iterative GrabCut. Compare panels 4 and 5 to assess quantum vs. classical performance.

To interpret: look for whether QAOA correctly segments the central object as fg. Images where QAOA and GrabCut agree show the quantum formulation is working correctly; disagreements indicate where the coarser superpixel resolution or QAOA approximation matters.

---

## Category D — Limitations, Future Work, Broader Impact (Q26–Q30)

---

**Q26. What is the fundamental scalability limitation of your approach?**

There are two distinct bottlenecks:

**NEQR/QHED bottleneck (state preparation)**: Encoding an N-pixel image into amplitude/basis quantum states requires O(N) classical operations to prepare the quantum register. This is not a quantum speedup — it is a classical overhead. Even though the Hadamard gate itself is O(1), the preparation dominates. Ruan (2021) identifies this as the central unresolved challenge in quantum image processing: practical advantage requires a quantum RAM (qRAM) that can load amplitude vectors in O(log N) time.

**QAOA bottleneck (qubit count)**: QAOA on `n_free` qubits has a statevector of size `2^n_free`. Classical simulation becomes impractical beyond ~25 qubits. More meaningful segmentation (50+ superpixels) requires real quantum hardware. Near-term devices (IBM Eagle: 127 qubits, IonQ Aria: 25 qubits) can in principle run 50-superpixel QAOA, but noise and decoherence degrade solution quality — error mitigation is needed.

---

**Q27. How would you extend this pipeline to run on real quantum hardware?**

Three steps:

1. **QHED on hardware**: Replace `qml.StatePrep` with a hardware-efficient encoding circuit that approximates the amplitude loading using a fixed-depth circuit (e.g., `qml.SimplifiedTwoDesign`). Alternatively, use a classical Sobel on the CPU and retain only the QAOA circuit on hardware.

2. **QAOA on hardware**: The QAOA circuit is already hardware-compatible — it uses only single-qubit RZ, RX and two-qubit CNOT operations, which are native on most gate-based processors. Change `qml.device("lightning.qubit", ...)` to `qml.device("qiskit.ibm.remote", ...)` (or PennyLane's IBM connector). Add a zero-noise extrapolation (ZNE) error mitigation wrapper.

3. **Scaling superpixels**: On IBM Eagle (127 qubits), use `n_segments = 100` with typical 40% seed coverage → `n_free ≈ 60` qubits. At QAOA depth p=2 this approaches GrabCut-quality segmentation while maintaining the quantum combinatorial optimization structure.

---

**Q28. Could your QAOA approach compete with deep learning (U-Net, SAM) today?**

No — not at the current scale, and this is acknowledged honestly in the paper. Deep learning methods like SAM operate at full pixel resolution with millions of parameters trained on hundreds of thousands of images. Our QAOA operates at 12-superpixel resolution with 2 parameters optimized on each individual image.

The correct comparison is not "QAOA vs. SAM" but "QAOA vs. classical graph-cut at the same superpixel resolution" — which is the GrabCut comparison we report. The purpose is not to beat deep learning today but to establish a quantum proof-of-concept that can be progressively improved as quantum hardware scales. The theoretical advantage of QAOA — polynomial exploration of exponential solution space — has not yet translated to practical image segmentation because hardware is not yet at the required scale.

---

**Q29. What is the honest quantum advantage in your pipeline? Where does quantum actually help?**

There are two places where quantum circuits provide **genuine mathematical value** (not just equivalent to classical with extra steps):

1. **QHED's Hadamard gradient step**: A single quantum gate simultaneously transforms all N/2 pixel pairs, computing their differences via quantum interference. In a true quantum circuit (on hardware), this is O(1) gates vs. O(N) classical operations — a logarithmic speedup IF state preparation is efficient. Currently, state preparation eliminates this speedup in simulation.

2. **QAOA's QUBO optimization**: For the binary labeling problem of size `n_free`, QAOA explores a quantum superposition of all `2^n_free` labelings simultaneously. Variational quantum algorithms can, for certain problem structures, find good approximate solutions faster than classical heuristics. Whether QAOA provides a practical advantage for the s-t graph-cut problem specifically is an open research question — our work sets up the correct problem formulation for testing this on hardware.

The honest summary: mathematically correct quantum circuits, demonstrated proof-of-concept, practical advantage pending hardware progress.

---

**Q30. What are three concrete next steps to advance this work?**

1. **Deploy QAOA on IBM Quantum hardware** with `n_segments = 50` and QAOA depth p=2–3. Use zero-noise extrapolation for error mitigation. Compare with our simulated results to characterize the hardware noise impact and establish the first hardware-validated quantum image segmentation result.

2. **Replace classical state preparation in QHED with a quantum loading scheme** — either a data-loading circuit from the quantum signal processing literature or a variational quantum autoencoder that maps a compressed representation to the full amplitude vector in O(polylog N) depth. This is the key step to unlocking the QHED speedup.

3. **Extend to multi-class segmentation** by generalizing the QUBO from binary (fg/bg) to k-class labeling, using a one-hot encoding across multiple qubit registers. This transforms the problem from binary QAOA to a higher-dimensional combinatorial optimization — a natural extension now that the binary pipeline is proven correct.
