"""Superpixel-based QUBO construction for quantum fg/bg segmentation.

Pipeline:
  1. SLIC superpixels (cv2.ximgproc) — reduces pixel count to n_sp
  2. Adjacency graph between spatially neighbouring superpixels
  3. Seed classification from trimap majority vote
  4. Unary terms h_i = log P(c_i|BG) − log P(c_i|FG)  via GMM on seed pixels
  5. Pairwise terms J_ij = β·exp(−‖c_i−c_j‖²/σ²)·(1−qhed_ij)
     — QHED edge strength qhed_ij suppresses the smoothness penalty at quantum-detected edges
  6. Seed hard constraints folded into effective h_i (reduces QUBO to free superpixels only)
  7. QUBO→Ising: h_ising[i] = −h_i/2,  J_ising[i,j] = −J_ij/2

Reference: Boykov & Jolly (2001) interactive graph cuts energy formulation.
"""

from __future__ import annotations

import numpy as np
import cv2


# --------------------------------------------------------------------------- #
# 1. SLIC superpixels
# --------------------------------------------------------------------------- #
def slic_segments(
    img_bgr: np.ndarray,
    n_segments: int = 16,
    compactness: float = 10.0,
    n_iters: int = 10,
) -> tuple[np.ndarray, int]:
    """Run cv2.ximgproc SuperpixelSLIC on a BGR image.

    Returns
    -------
    labels : (H, W) int32, values in [0, n_sp)
    n_sp   : actual superpixel count (may differ slightly from n_segments)
    """
    img_u8 = img_bgr if img_bgr.dtype == np.uint8 else np.clip(img_bgr, 0, 255).astype(np.uint8)
    img_lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2Lab)
    H, W = img_lab.shape[:2]
    region_size = max(4, int(np.sqrt(H * W / max(1, n_segments))))
    slic = cv2.ximgproc.createSuperpixelSLIC(
        img_lab,
        algorithm=cv2.ximgproc.SLIC,
        region_size=region_size,
        ruler=float(compactness),
    )
    slic.iterate(n_iters)
    labels = slic.getLabels().astype(np.int32)
    n_sp = int(slic.getNumberOfSuperpixels())
    return labels, n_sp


# --------------------------------------------------------------------------- #
# 2. Per-superpixel mean Lab colors
# --------------------------------------------------------------------------- #
def sp_mean_colors(
    img_lab: np.ndarray,
    labels: np.ndarray,
    n_sp: int,
) -> np.ndarray:
    """Mean Lab color per superpixel.

    Parameters
    ----------
    img_lab : (H, W, 3) float32 Lab image (L∈[0,100], a,b∈[−127,127])
    labels  : (H, W) int32 from slic_segments
    n_sp    : number of superpixels

    Returns
    -------
    colors : (n_sp, 3) float64
    """
    flat_lab = img_lab.reshape(-1, 3).astype(np.float64)
    flat_lbl = labels.ravel()
    colors = np.zeros((n_sp, 3), np.float64)
    counts = np.bincount(flat_lbl, minlength=n_sp).astype(float)
    for c in range(3):
        colors[:, c] = np.bincount(flat_lbl, weights=flat_lab[:, c], minlength=n_sp)
    counts = np.where(counts > 0, counts, 1.0)
    return colors / counts[:, None]


# --------------------------------------------------------------------------- #
# 3. Superpixel adjacency graph
# --------------------------------------------------------------------------- #
def sp_adjacency(labels: np.ndarray, n_sp: int) -> list[tuple[int, int]]:
    """4-connected adjacency: pairs (i, j) with i < j sharing a pixel boundary."""
    lab = np.asarray(labels, np.int32)
    edge_set: set[tuple[int, int]] = set()
    # Horizontal neighbours
    lr = lab[:, :-1]
    rr = lab[:, 1:]
    rows, cols = np.where(lr != rr)
    for r, c in zip(rows.tolist(), cols.tolist()):
        a, b = int(lr[r, c]), int(rr[r, c])
        edge_set.add((min(a, b), max(a, b)))
    # Vertical neighbours
    tr = lab[:-1, :]
    br = lab[1:, :]
    rows, cols = np.where(tr != br)
    for r, c in zip(rows.tolist(), cols.tolist()):
        a, b = int(tr[r, c]), int(br[r, c])
        edge_set.add((min(a, b), max(a, b)))
    return sorted(edge_set)


# --------------------------------------------------------------------------- #
# 4. QHED edge strength along shared boundaries
# --------------------------------------------------------------------------- #
def sp_qhed_edge_strength(
    labels: np.ndarray,
    edge_map: np.ndarray,
    edges: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Mean QHED edge value along the shared pixel boundary for each SP edge.

    Returns
    -------
    qhed_strength : dict (i, j) → float in [0, 1]
    """
    edge_set = set(edges)
    accum: dict[tuple[int, int], list[float]] = {e: [] for e in edge_set}
    lab = np.asarray(labels, np.int32)
    em  = np.asarray(edge_map, float)
    # Horizontal boundaries
    lr, rr = lab[:, :-1], lab[:, 1:]
    rows, cols = np.where(lr != rr)
    for r, c in zip(rows.tolist(), cols.tolist()):
        key = (min(int(lr[r, c]), int(rr[r, c])), max(int(lr[r, c]), int(rr[r, c])))
        if key in accum:
            accum[key].append(float(em[r, c]))
            accum[key].append(float(em[r, c + 1]))
    # Vertical boundaries
    tr, br = lab[:-1, :], lab[1:, :]
    rows, cols = np.where(tr != br)
    for r, c in zip(rows.tolist(), cols.tolist()):
        key = (min(int(tr[r, c]), int(br[r, c])), max(int(tr[r, c]), int(br[r, c])))
        if key in accum:
            accum[key].append(float(em[r, c]))
            accum[key].append(float(em[r + 1, c]))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in accum.items()}


# --------------------------------------------------------------------------- #
# 5a. GMM fitting on seed pixels
# --------------------------------------------------------------------------- #
def fit_gmm(pixels: np.ndarray, n_components: int = 3, random_state: int = 42):
    """Fit a GaussianMixture on pixel Lab colors (N, 3).

    Returns a fitted sklearn.mixture.GaussianMixture.
    """
    from sklearn.mixture import GaussianMixture
    n_comp = max(1, min(n_components, len(pixels) // 2)) if len(pixels) >= 2 else 1
    gmm = GaussianMixture(n_components=n_comp, covariance_type="full",
                          max_iter=100, random_state=random_state)
    gmm.fit(pixels.reshape(-1, 3))
    return gmm


# --------------------------------------------------------------------------- #
# 5b. Unary terms
# --------------------------------------------------------------------------- #
def compute_unary(
    sp_colors: np.ndarray,
    gmm_fg,
    gmm_bg,
) -> np.ndarray:
    """h_i = log P(c_i|BG) − log P(c_i|FG).

    Positive h_i → bg-likely (z_i=0 preferred).
    Negative h_i → fg-likely (z_i=1 preferred).
    """
    log_fg = gmm_fg.score_samples(sp_colors)   # (n_sp,)
    log_bg = gmm_bg.score_samples(sp_colors)   # (n_sp,)
    return log_bg - log_fg


# --------------------------------------------------------------------------- #
# 5c. Pairwise terms
# --------------------------------------------------------------------------- #
def compute_pairwise(
    sp_colors: np.ndarray,
    edges: list[tuple[int, int]],
    qhed_strength: dict[tuple[int, int], float],
    beta: float = 10.0,
    sigma: float | None = None,
) -> dict[tuple[int, int], float]:
    """J_ij = β·exp(−‖c_i−c_j‖²/σ²)·(1−qhed_ij).

    sigma defaults to the median pairwise Lab distance between adjacent superpixels.
    """
    if not edges:
        return {}
    dists = [float(np.linalg.norm(sp_colors[i] - sp_colors[j])) for i, j in edges]
    if sigma is None:
        sigma = max(1e-3, float(np.median(dists)))
    sigma2 = sigma ** 2
    J: dict[tuple[int, int], float] = {}
    for (i, j), d in zip(edges, dists):
        qe = qhed_strength.get((i, j), 0.0)
        J[(i, j)] = max(1e-6, beta * np.exp(-d * d / sigma2) * (1.0 - qe))
    return J


# --------------------------------------------------------------------------- #
# 6. Seed classification from trimap
# --------------------------------------------------------------------------- #
def classify_seeds(
    labels: np.ndarray,
    trimap: np.ndarray,
    n_sp: int,
    fg_thr: float = 0.5,
    bg_thr: float = 0.5,
) -> np.ndarray:
    """Assign each superpixel as fg seed (1), bg seed (0), or free (−1).

    A superpixel is fg/bg seed if the majority of its pixels are in that seed region.
    Trimap convention: 255=fg seed, 0=bg seed, 128=unknown.
    """
    flat_lbl = labels.ravel()
    flat_tri = np.asarray(trimap, np.int32).ravel()
    counts   = np.bincount(flat_lbl, minlength=n_sp).astype(float)
    fg_counts = np.bincount(flat_lbl, weights=(flat_tri == 255).astype(float), minlength=n_sp)
    bg_counts = np.bincount(flat_lbl, weights=(flat_tri == 0  ).astype(float), minlength=n_sp)
    safe = np.where(counts > 0, counts, 1.0)
    sp_type = np.full(n_sp, -1, np.int8)
    sp_type[fg_counts / safe > fg_thr] = 1
    sp_type[bg_counts / safe > bg_thr] = 0
    return sp_type


# --------------------------------------------------------------------------- #
# 7. Reduce QUBO with seed hard constraints
# --------------------------------------------------------------------------- #
def build_reduced_qubo(
    h: np.ndarray,
    J: dict[tuple[int, int], float],
    sp_type: np.ndarray,
    n_sp: int,
) -> tuple[np.ndarray, dict[tuple[int, int], float], list[int]]:
    """Fold seed constraints into h_eff, return QUBO for free superpixels only.

    For fg seed k (z_k=1) and free neighbour j:  Δh_j += −J_kj
    For bg seed k (z_k=0) and free neighbour j:  Δh_j += +J_kj

    Returns
    -------
    h_eff      : (n_free,) effective unary terms
    J_free     : pairwise dict with local indices (i, j), i < j
    free_nodes : list of global SP indices for each free node (length n_free)
    """
    free_nodes = [i for i in range(n_sp) if sp_type[i] == -1]
    n_free = len(free_nodes)
    global_to_local = {g: l for l, g in enumerate(free_nodes)}

    h_eff = h[free_nodes].copy()

    # Accumulate seed constraint offsets
    for (gi, gj), Jij in J.items():
        ti, tj = int(sp_type[gi]), int(sp_type[gj])
        if ti != -1 and tj == -1:          # gi is seed, gj is free
            lj = global_to_local[gj]
            if ti == 1:    # fg seed → Δh_j += -J
                h_eff[lj] -= Jij
            else:          # bg seed → Δh_j += +J
                h_eff[lj] += Jij
        elif ti == -1 and tj != -1:        # gi is free, gj is seed
            li = global_to_local[gi]
            if tj == 1:    # fg seed
                h_eff[li] -= Jij
            else:          # bg seed
                h_eff[li] += Jij

    # Build pairwise among free nodes only
    J_free: dict[tuple[int, int], float] = {}
    for (gi, gj), Jij in J.items():
        if sp_type[gi] == -1 and sp_type[gj] == -1:
            li, lj = global_to_local[gi], global_to_local[gj]
            key = (min(li, lj), max(li, lj))
            J_free[key] = Jij

    return h_eff, J_free, free_nodes


# --------------------------------------------------------------------------- #
# 8. QUBO → Ising conversion
# --------------------------------------------------------------------------- #
def qubo_to_ising(
    h_eff: np.ndarray,
    J_free: dict[tuple[int, int], float],
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Convert QUBO coefficients to Ising (PennyLane Z-operator) coefficients.

    QUBO z∈{0,1} → Ising s∈{+1,−1} via z=(1−s)/2.

    h_ising[i]   = −h_eff[i] / 2   (coefficient of Z_i in H_C)
    J_ising[i,j] = −J_ij / 2       (coefficient of Z_i⊗Z_j)

    The mapping ensures: minimising ⟨H_C⟩ is equivalent to minimising E(z).
    """
    h_ising = -h_eff / 2.0
    J_ising = {k: -v / 2.0 for k, v in J_free.items()}
    return h_ising, J_ising


# --------------------------------------------------------------------------- #
# Automatic trimap generator (for images without annotations)
# --------------------------------------------------------------------------- #
def auto_trimap(
    H: int,
    W: int,
    fg_radius_frac: float = 0.28,
    bg_margin_frac: float = 0.08,
) -> np.ndarray:
    """Generate a center-ellipse fg / border bg trimap for unannotated images.

    Parameters
    ----------
    H, W            : image height and width
    fg_radius_frac  : fg ellipse radius as fraction of each dimension
    bg_margin_frac  : bg border margin as fraction of each dimension

    Returns
    -------
    trimap : (H, W) uint8 — 0=bg seed, 128=unknown, 255=fg seed
    """
    trimap = np.full((H, W), 128, np.uint8)
    # Bg seeds: outer border
    mh = max(1, int(H * bg_margin_frac))
    mw = max(1, int(W * bg_margin_frac))
    trimap[:mh, :]  = 0
    trimap[-mh:, :] = 0
    trimap[:, :mw]  = 0
    trimap[:, -mw:] = 0
    # Fg seeds: center ellipse
    cy, cx = H // 2, W // 2
    ry = max(2, int(H * fg_radius_frac))
    rx = max(2, int(W * fg_radius_frac))
    cv2.ellipse(trimap, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return trimap


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    H, W = 64, 64
    # Synthetic BGR image: fg circle on dark background
    img = np.zeros((H, W, 3), np.uint8)
    cv2.circle(img, (W // 2, H // 2), 20, (180, 100, 60), -1)
    img += rng.integers(0, 20, img.shape, dtype=np.uint8)

    # Synthetic trimap
    trimap = np.full((H, W), 128, np.uint8)
    trimap[35:45, 27:37] = 255   # fg seeds (inside circle)
    trimap[:10, :10] = 0         # bg seeds (corner)

    # Synthetic edge map (zeros)
    edge_map = np.zeros((H, W), float)

    labels, n_sp = slic_segments(img, n_segments=16)
    print(f"SLIC: n_sp={n_sp}")

    img_lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    sp_colors = sp_mean_colors(img_lab, labels, n_sp)
    edges = sp_adjacency(labels, n_sp)
    print(f"Adjacency: {len(edges)} edges")

    qhed_str = sp_qhed_edge_strength(labels, edge_map, edges)

    sp_type = classify_seeds(labels, trimap, n_sp)
    n_fg = int(np.sum(sp_type == 1))
    n_bg = int(np.sum(sp_type == 0))
    n_free = int(np.sum(sp_type == -1))
    print(f"Seeds: fg={n_fg} bg={n_bg} free={n_free}")

    fg_pix = img_lab[(trimap == 255)]
    bg_pix = img_lab[(trimap == 0)]
    gmm_fg = fit_gmm(fg_pix, n_components=1)
    gmm_bg = fit_gmm(bg_pix, n_components=1)

    h = compute_unary(sp_colors, gmm_fg, gmm_bg)
    J = compute_pairwise(sp_colors, edges, qhed_str)
    print(f"h range: [{h.min():.3f}, {h.max():.3f}]   J entries: {len(J)}")

    h_eff, J_free, free_nodes = build_reduced_qubo(h, J, sp_type, n_sp)
    h_ising, J_ising = qubo_to_ising(h_eff, J_free)
    print(f"Reduced QUBO: n_free={len(free_nodes)}  J_free entries={len(J_free)}")
    print(f"Ising h range: [{h_ising.min():.3f}, {h_ising.max():.3f}]")
    print("superpixel self-test PASSED")
