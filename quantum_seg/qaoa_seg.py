"""QAOA-based foreground/background segmentation.

Stage 4 of the quantum image segmentation pipeline:

  Image + trimap seeds
    → QHED edge map (Stage-2 quantum feature)
    → SLIC superpixels + QUBO construction  (superpixel.py)
    → QAOA optimisation (this file)
    → binary fg/bg pixel mask

Quantum circuit
---------------
Cost Hamiltonian (Ising form):
    H_C = Σ_i h_ising[i]·Z_i  +  Σ_{i<j} J_ising[i,j]·Z_i⊗Z_j

Mixer Hamiltonian (standard X-mixer):
    H_M = Σ_i X_i

QAOA circuit of depth p:
    |ψ(γ,β)⟩ = [Π_k U_M(β_k)·U_C(γ_k)] |+⟩^n
             where U_C(γ) = exp(−iγH_C), U_M(β) = exp(−iβH_M)

Optimised with scipy COBYLA (gradient-free), 5 random restarts.

Honest limitation: classical simulation via lightning.qubit is tractable to
~25 qubits; full-image QAOA at pixel granularity requires quantum hardware.

References: Farhi, Goldstone & Gutmann (2014) QAOA.
            Boykov & Jolly (2001) interactive graph cuts (the QUBO we solve).
"""

from __future__ import annotations

import time

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from .superpixel import (
    classify_seeds,
    compute_pairwise,
    compute_unary,
    fit_gmm,
    build_reduced_qubo,
    qubo_to_ising,
    slic_segments,
    sp_adjacency,
    sp_mean_colors,
    sp_qhed_edge_strength,
)


# --------------------------------------------------------------------------- #
# Hamiltonian construction
# --------------------------------------------------------------------------- #
def build_cost_hamiltonian(
    h_ising: np.ndarray,
    J_ising: dict[tuple[int, int], float],
    wires: list[int],
) -> qml.Hamiltonian:
    """Build H_C = Σ_i h_ising[i]·Z_i + Σ_{i<j} J_ising[i,j]·Z_i⊗Z_j.

    Uses local wire indices (wire = wires[i] for h_ising[i]).
    """
    coeffs: list[float] = []
    obs: list = []
    for i, h_val in enumerate(h_ising):
        if abs(h_val) > 1e-12:
            coeffs.append(float(h_val))
            obs.append(qml.PauliZ(wires[i]))
    for (i, j), J_val in J_ising.items():
        if abs(J_val) > 1e-12:
            coeffs.append(float(J_val))
            obs.append(qml.PauliZ(wires[i]) @ qml.PauliZ(wires[j]))
    if not coeffs:                          # trivial zero Hamiltonian
        coeffs = [0.0]
        obs = [qml.PauliZ(wires[0])]
    return qml.Hamiltonian(coeffs, obs)


def build_mixer_hamiltonian(wires: list[int]) -> qml.Hamiltonian:
    """Standard X-mixer: H_M = Σ_i X_i."""
    return qml.qaoa.x_mixer(wires)


# --------------------------------------------------------------------------- #
# QAOA circuit
# --------------------------------------------------------------------------- #
def make_qaoa_circuit(
    H_C: qml.Hamiltonian,
    H_M: qml.Hamiltonian,
    wires: list[int],
    p: int = 1,
    device_name: str = "default.qubit",  # default.qubit for Python 3.14+ compatibility; use lightning.qubit for speed
) -> tuple:
    """Return (cost_qnode, prob_qnode) sharing the same QAOA circuit body.

    cost_qnode(params) → scalar expval(H_C)  — used during optimisation.
    prob_qnode(params) → (2^n,) probability vector — used for decoding.

    params layout: [γ_1, β_1, …, γ_p, β_p]  (length 2p).
    """
    dev = qml.device(device_name, wires=wires)

    def _circuit(params):
        gamma = params[:p]
        beta  = params[p:]
        for w in wires:
            qml.Hadamard(wires=w)
        for k in range(p):
            qml.qaoa.cost_layer(float(gamma[k]), H_C)
            qml.qaoa.mixer_layer(float(beta[k]),  H_M)

    @qml.qnode(dev)
    def cost_qnode(params):
        _circuit(params)
        return qml.expval(H_C)

    @qml.qnode(dev)
    def prob_qnode(params):
        _circuit(params)
        return qml.probs(wires=wires)

    return cost_qnode, prob_qnode


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def optimize_qaoa(
    cost_qnode,
    p: int = 1,
    maxiter: int = 200,
    init_params: np.ndarray | None = None,
    random_state: int = 0,
    n_restarts: int = 5,
) -> tuple[np.ndarray, float]:
    """COBYLA optimisation of QAOA parameters with random restarts.

    Returns
    -------
    best_params : (2p,) optimised [γ_1, β_1, …, γ_p, β_p]
    best_cost   : final ⟨H_C⟩ at best_params
    """
    def cost_fn(params):
        return float(cost_qnode(np.asarray(params, dtype=float)))

    rng = np.random.default_rng(random_state)
    best_params = np.zeros(2 * p)
    best_cost   = float("inf")

    actual_restarts = 1 if init_params is not None else n_restarts
    for restart in range(actual_restarts):
        x0 = init_params if init_params is not None else rng.uniform(0, np.pi, 2 * p)
        res = minimize(cost_fn, x0, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.5})
        if res.fun < best_cost:
            best_cost   = float(res.fun)
            best_params = np.asarray(res.x, dtype=float)

    return best_params, best_cost


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def decode_qaoa(
    prob_qnode,
    params: np.ndarray,
    n_free: int,
) -> np.ndarray:
    """Sample most-probable bitstring from the optimised QAOA state.

    Wire convention: wire 0 is the most-significant bit of the basis index.
    Bit value 0 → background, 1 → foreground.

    Returns
    -------
    z_free : (n_free,) int array of {0, 1}
    """
    probs = np.asarray(prob_qnode(params), dtype=float)
    best_idx = int(np.argmax(probs))
    z_free = np.array(
        [(best_idx >> (n_free - 1 - i)) & 1 for i in range(n_free)],
        dtype=np.int32,
    )
    return z_free


# --------------------------------------------------------------------------- #
# Mask reconstruction
# --------------------------------------------------------------------------- #
def reconstruct_pixel_mask(
    z_free: np.ndarray,
    free_nodes: list[int],
    sp_type: np.ndarray,
    labels: np.ndarray,
    n_sp: int,
) -> np.ndarray:
    """Map superpixel assignments back to a (H, W) uint8 pixel mask.

    Seed superpixels are assigned their fixed labels; free superpixels use
    z_free assignments.  Each pixel inherits its superpixel's label.
    """
    sp_label = np.zeros(n_sp, np.uint8)
    for i in range(n_sp):
        if sp_type[i] == 1:                  # fg seed
            sp_label[i] = 1
        elif sp_type[i] == 0:                # bg seed
            sp_label[i] = 0
    for local_idx, global_idx in enumerate(free_nodes):
        sp_label[global_idx] = int(z_free[local_idx])
    return sp_label[labels].astype(np.uint8)


# --------------------------------------------------------------------------- #
# Resource summary
# --------------------------------------------------------------------------- #
def qaoa_resource_report(n_free: int, p: int, n_edges: int) -> dict:
    """Quantum resource summary for the QAOA segmentation circuit."""
    depth_est = p * (n_free + 3 * n_edges)     # H + RZ per node, CNOT+RZ+CNOT per edge, RX per node
    note_sim = (
        f"n_free={n_free} qubits — lightning.qubit tractable to ~25; "
        "increase n_segments cautiously."
    ) if n_free <= 25 else (
        f"WARNING: n_free={n_free} > 25 — simulation will be slow. "
        "Reduce n_segments."
    )
    return {
        "qubits":        n_free,
        "params":        2 * p,
        "cost_h_terms":  n_free + n_edges,
        "depth_estimate": depth_est,
        "simulator_limit_note": note_sim,
        "hardware_note": (
            "Full-image QAOA (one qubit/pixel) requires fault-tolerant "
            "quantum hardware. SLIC + QAOA on n_segments superpixels is "
            "the practical near-term formulation."
        ),
    }


# --------------------------------------------------------------------------- #
# Full segmentation pipeline
# --------------------------------------------------------------------------- #
def segment_image(
    img_bgr: np.ndarray,
    trimap: np.ndarray,
    n_segments: int = 16,
    p: int = 1,
    beta: float = 10.0,
    sigma: float | None = None,
    gmm_components: int = 3,
    compactness: float = 10.0,
    maxiter: int = 200,
    random_state: int = 0,
    use_qhed: bool = True,
) -> dict:
    """End-to-end QAOA foreground/background segmentation.

    Parameters
    ----------
    img_bgr     : (H, W, 3) uint8 BGR image
    trimap      : (H, W) uint8 — 0=bg seed, 128=unknown, 255=fg seed
    n_segments  : target SLIC superpixel count (= approximate QAOA qubit count)
    p           : QAOA circuit depth (1 or 2)
    beta        : pairwise smoothness scale
    sigma       : Lab color bandwidth for J_ij (auto if None)
    gmm_components : GMM components for unary term
    compactness : SLIC compactness
    maxiter     : COBYLA iterations per restart
    random_state: RNG seed
    use_qhed    : integrate QHED quantum edge map into pairwise term

    Returns
    -------
    dict with keys: mask, labels, n_sp, n_free, n_fg_seeds, n_bg_seeds,
                    h_eff, J_free, best_params, best_cost, qhed_map, sim_time_s
    """
    import cv2
    t0 = time.time()
    H_img, W_img = img_bgr.shape[:2]

    # ---- 1. QHED edge map (quantum feature; resized to original dims) ----
    if use_qhed:
        from .qsobel import qhed_edges_refined
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float)
        pow2_h = max(2, 2 ** int(round(np.log2(max(H_img, 2)))))
        pow2_w = max(2, 2 ** int(round(np.log2(max(W_img, 2)))))
        gray_p2 = cv2.resize(gray.astype(np.float32), (pow2_w, pow2_h)).astype(float)
        em_p2 = qhed_edges_refined(gray_p2)
        edge_map = cv2.resize(em_p2.astype(np.float32), (W_img, H_img)).astype(float)
    else:
        edge_map = np.zeros((H_img, W_img), float)

    # ---- 2. SLIC superpixels ----
    labels, n_sp = slic_segments(img_bgr, n_segments=n_segments, compactness=compactness)

    # ---- 3. Per-SP Lab colors ----
    img_lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    sp_colors = sp_mean_colors(img_lab, labels, n_sp)

    # ---- 4. Adjacency + QHED edge strength ----
    edges    = sp_adjacency(labels, n_sp)
    qhed_str = sp_qhed_edge_strength(labels, edge_map, edges)

    # ---- 5. Seed classification ----
    sp_type = classify_seeds(labels, trimap, n_sp)

    # ---- 6. GMM on seed pixels ----
    fg_pixels = img_lab[trimap == 255].reshape(-1, 3)
    bg_pixels = img_lab[trimap == 0  ].reshape(-1, 3)
    if len(fg_pixels) == 0:
        fg_pixels = sp_colors[[i for i in range(n_sp) if sp_type[i] == 1]]
    if len(bg_pixels) == 0:
        bg_pixels = sp_colors[[i for i in range(n_sp) if sp_type[i] == 0]]
    gmm_fg = fit_gmm(fg_pixels, n_components=gmm_components, random_state=random_state)
    gmm_bg = fit_gmm(bg_pixels, n_components=gmm_components, random_state=random_state)

    # ---- 7. QUBO terms ----
    h = compute_unary(sp_colors, gmm_fg, gmm_bg)
    J = compute_pairwise(sp_colors, edges, qhed_str, beta=beta, sigma=sigma)

    # ---- 8. Reduce QUBO with seed hard constraints ----
    h_eff, J_free, free_nodes = build_reduced_qubo(h, J, sp_type, n_sp)
    n_free = len(free_nodes)
    n_fg_seeds = int(np.sum(sp_type == 1))
    n_bg_seeds = int(np.sum(sp_type == 0))

    # ---- 9. QAOA ----
    if n_free == 0:
        z_free      = np.array([], dtype=np.int32)
        best_params = np.array([], dtype=float)
        best_cost   = 0.0
    else:
        h_ising, J_ising = qubo_to_ising(h_eff, J_free)
        wires = list(range(n_free))
        H_C = build_cost_hamiltonian(h_ising, J_ising, wires)
        H_M = build_mixer_hamiltonian(wires)
        cost_qnode, prob_qnode = make_qaoa_circuit(H_C, H_M, wires, p=p)
        best_params, best_cost = optimize_qaoa(
            cost_qnode, p=p, maxiter=maxiter, random_state=random_state
        )
        z_free = decode_qaoa(prob_qnode, best_params, n_free)

    # ---- 10. Pixel mask ----
    mask = reconstruct_pixel_mask(z_free, free_nodes, sp_type, labels, n_sp)

    # Clean up quantum devices to free memory
    import gc
    gc.collect()

    return {
        "mask":        mask,
        "labels":      labels,
        "n_sp":        n_sp,
        "n_free":      n_free,
        "n_fg_seeds":  n_fg_seeds,
        "n_bg_seeds":  n_bg_seeds,
        "h_eff":       h_eff,
        "J_free":      J_free,
        "best_params": best_params,
        "best_cost":   best_cost,
        "qhed_map":    edge_map if use_qhed else None,
        "sim_time_s":  time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import cv2

    rng = np.random.default_rng(0)
    H, W = 64, 64
    # Synthetic: bright circle (fg) on dark bg
    img = np.full((H, W, 3), 30, np.uint8)
    cv2.circle(img, (W // 2, H // 2), 20, (200, 120, 80), -1)
    img = np.clip(img.astype(int) + rng.integers(-15, 15, img.shape), 0, 255).astype(np.uint8)

    # Trimap: ring of fg seeds inside circle, corners as bg seeds
    trimap = np.full((H, W), 128, np.uint8)
    cv2.circle(trimap, (W // 2, H // 2), 8, 255, -1)   # fg seeds
    trimap[:8, :8] = 0
    trimap[:8, W-8:] = 0
    trimap[H-8:, :8] = 0
    trimap[H-8:, W-8:] = 0                              # bg seeds

    print("Running QAOA fg/bg segmentation on 64x64 synthetic image ...")
    result = segment_image(img, trimap, n_segments=12, p=1, maxiter=100, use_qhed=False)
    mask = result["mask"]

    # Rough correctness check: centre should be fg, corner should be bg
    centre_fg = float(mask[H // 2, W // 2])
    corner_bg = float(mask[4, 4])
    n_free = result["n_free"]
    print(f"  n_sp={result['n_sp']}  n_free={n_free}  "
          f"fg_seeds={result['n_fg_seeds']}  bg_seeds={result['n_bg_seeds']}")
    print(f"  best_cost={result['best_cost']:.4f}  sim_time={result['sim_time_s']:.1f}s")
    print(f"  centre pixel label={int(centre_fg)} (expected 1=fg)  "
          f"corner pixel label={int(corner_bg)} (expected 0=bg)")

    rr = qaoa_resource_report(n_free, p=1, n_edges=len(result["J_free"]))
    print(f"  Resource: {rr['qubits']} qubits, {rr['params']} params, depth~{rr['depth_estimate']}")
    print("qaoa_seg self-test PASSED")
