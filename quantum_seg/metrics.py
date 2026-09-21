"""The mandatory metric battery for comparing against SOTA.

Grouped exactly as in the approved plan:

  Boundary (need GT)      : Boundary-F1 (+ dataset ODS / OIS), Hausdorff / 95-HD, Pratt FOM
  Fidelity (encode->decode): PSNR, MSE, MAE, SSIM   -- proves NEQR preserves spatial info
  No-reference (no GT)     : edge density, QHED-vs-baseline tolerance agreement
  Region (if closed)       : IoU / Dice / pixel-accuracy helpers
  Resource (quantum claim) : pass-through dict from neqr / qsobel

Implemented without scikit-image / scikit-learn (only numpy + scipy) so the pipeline
runs in this environment as-is.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


# --------------------------------------------------------------------------- #
# fidelity (reconstruction / representation)
# --------------------------------------------------------------------------- #
def mse(a, b):
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def psnr(a, b, peak: float = 255.0):
    e = mse(a, b)
    return float("inf") if e == 0 else float(10 * np.log10(peak ** 2 / e))


def ssim(a, b, peak: float = 255.0, sigma: float = 1.5):
    """Gaussian-window SSIM (Wang et al. 2004), global mean over the map."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    C1 = (0.01 * peak) ** 2
    C2 = (0.03 * peak) ** 2
    mu_a = gaussian_filter(a, sigma)
    mu_b = gaussian_filter(b, sigma)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    va = gaussian_filter(a * a, sigma) - mu_a2
    vb = gaussian_filter(b * b, sigma) - mu_b2
    vab = gaussian_filter(a * b, sigma) - mu_ab
    s = ((2 * mu_ab + C1) * (2 * vab + C2)) / ((mu_a2 + mu_b2 + C1) * (va + vb + C2))
    return float(np.mean(s))


def epi(orig, proc):
    """Edge-Preservation Index: Pearson correlation of Laplacian-high-passed images."""
    from scipy.ndimage import laplace
    lo = laplace(np.asarray(orig, float)).ravel()
    lp = laplace(np.asarray(proc, float)).ravel()
    if lo.std() == 0 or lp.std() == 0:
        return 1.0 if np.allclose(lo, lp) else 0.0
    return float(np.corrcoef(lo, lp)[0, 1])


def fidelity_report(orig, recon) -> dict:
    return {"PSNR": psnr(orig, recon), "MSE": mse(orig, recon),
            "MAE": mae(orig, recon), "SSIM": ssim(orig, recon), "EPI": epi(orig, recon)}


# --------------------------------------------------------------------------- #
# boundary metrics (need ground-truth edge map)
# --------------------------------------------------------------------------- #
def _binarize(edge_map, thr):
    return np.asarray(edge_map, float) >= thr


def boundary_pr_f(pred_bin, gt_bin, tol: int = 2):
    """Precision/Recall/F with a matching tolerance of ``tol`` pixels (BSDS style).

    A predicted edge pixel is a hit if a GT edge lies within ``tol`` px (and vice-versa
    for recall), measured via Euclidean distance transforms.
    """
    pred = np.asarray(pred_bin, bool)
    gt = np.asarray(gt_bin, bool)
    if pred.sum() == 0 or gt.sum() == 0:
        p = 1.0 if pred.sum() == 0 else 0.0
        r = 1.0 if gt.sum() == 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f
    dt_gt = distance_transform_edt(~gt)        # dist from every pixel to nearest GT edge
    dt_pred = distance_transform_edt(~pred)
    precision = np.mean(dt_gt[pred] <= tol)    # predicted edges close to a GT edge
    recall = np.mean(dt_pred[gt] <= tol)       # GT edges covered by a prediction
    f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f)


def boundary_f_sweep(pred_map, gt_bin, tol: int = 2, n_thr: int = 20):
    """Sweep thresholds -> list of (thr, P, R, F). Best-F over this is the image's OIS."""
    pred_map = np.asarray(pred_map, float)
    hi = pred_map.max()
    if hi <= 0:
        return [(0.0, 0.0, 0.0, 0.0)]
    out = []
    for thr in np.linspace(hi / n_thr, hi, n_thr):
        p, r, f = boundary_pr_f(pred_map >= thr, gt_bin, tol)
        out.append((float(thr), p, r, f))
    return out


def best_f(sweep):
    return max(sweep, key=lambda t: t[3])      # (thr, P, R, F) with max F


def pratt_fom(pred_bin, gt_bin, alpha: float = 1.0 / 9.0):
    """Pratt's Figure of Merit -- edge-localisation accuracy in [0,1] (1 = perfect)."""
    pred = np.asarray(pred_bin, bool)
    gt = np.asarray(gt_bin, bool)
    n_pred, n_gt = int(pred.sum()), int(gt.sum())
    if n_pred == 0 or n_gt == 0:
        return 0.0
    d = distance_transform_edt(~gt)[pred]      # distance of each predicted edge to GT
    fom = np.sum(1.0 / (1.0 + alpha * d ** 2)) / max(n_pred, n_gt)
    return float(fom)


def hausdorff(pred_bin, gt_bin, pct: float = 95.0):
    """(Directed-symmetric) Hausdorff distance between edge pixel sets; pct-percentile."""
    pred = np.asarray(pred_bin, bool)
    gt = np.asarray(gt_bin, bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")
    d_p2g = distance_transform_edt(~gt)[pred]
    d_g2p = distance_transform_edt(~pred)[gt]
    return float(max(np.percentile(d_p2g, pct), np.percentile(d_g2p, pct)))


def boundary_report(pred_map, gt_bin, tol: int = 2) -> dict:
    """Full boundary battery for one image (best-F over a threshold sweep)."""
    sweep = boundary_f_sweep(pred_map, gt_bin, tol=tol)
    thr, p, r, f = best_f(sweep)
    pred_bin = np.asarray(pred_map, float) >= thr
    return {"BF_bestF": f, "precision": p, "recall": r, "thr": thr,
            "PrattFOM": pratt_fom(pred_bin, gt_bin),
            "Hausdorff95": hausdorff(pred_bin, gt_bin), "_sweep": sweep}


# --------------------------------------------------------------------------- #
# dataset-level ODS / OIS (the headline BSDS SOTA numbers)
# --------------------------------------------------------------------------- #
def dataset_ods_ois(sweeps: list) -> dict:
    """ODS = best F at a single global threshold; OIS = mean of per-image best F.

    ``sweeps`` is a list of per-image sweeps (each a list of (thr,P,R,F)); all sweeps
    must share the same threshold grid length.
    """
    if not sweeps:
        return {"ODS": 0.0, "OIS": 0.0}
    k = len(sweeps[0])
    mean_f_per_thr = [np.mean([s[i][3] for s in sweeps]) for i in range(k)]
    ods = float(max(mean_f_per_thr))
    ois = float(np.mean([best_f(s)[3] for s in sweeps]))
    return {"ODS": ods, "OIS": ois}


# --------------------------------------------------------------------------- #
# no-reference (no GT available -- e.g. on misc)
# --------------------------------------------------------------------------- #
def edge_density(edge_bin):
    return float(np.mean(np.asarray(edge_bin, bool)))


def tolerance_agreement(map_a, map_b, tol: int = 1, q: float = 0.85):
    """No-GT sanity: tolerance-aware overlap of two edge maps (top-(1-q) quantile)."""
    a = np.asarray(map_a, float) >= np.quantile(map_a, q)
    b = np.asarray(map_b, float) >= np.quantile(map_b, q)
    return boundary_pr_f(a, b, tol)[2]


# --------------------------------------------------------------------------- #
# region metrics (only if edges are closed into regions)
# --------------------------------------------------------------------------- #
def iou(pred_mask, gt_mask):
    p, g = np.asarray(pred_mask, bool), np.asarray(gt_mask, bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union else 1.0


def dice(pred_mask, gt_mask):
    p, g = np.asarray(pred_mask, bool), np.asarray(gt_mask, bool)
    s = p.sum() + g.sum()
    return float(2 * np.logical_and(p, g).sum() / s) if s else 1.0


def pixel_accuracy(pred_mask, gt_mask) -> float:
    p, g = np.asarray(pred_mask, bool), np.asarray(gt_mask, bool)
    return float(np.mean(p == g))


def segmentation_report(pred_mask, gt_mask) -> dict:
    return {
        "IoU": iou(pred_mask, gt_mask),
        "Dice": dice(pred_mask, gt_mask),
        "PixelAcc": pixel_accuracy(pred_mask, gt_mask),
    }
