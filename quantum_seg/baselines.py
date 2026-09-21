"""Classical edge-detection baselines (OpenCV) -- the fair same-scale comparators.

Each function takes a grayscale float/uint image and returns an edge-magnitude map
normalised to [0, 1] (Canny returns a {0,1} map).  These are run on the *identical*
inputs as QHED/QSobel so the metric comparison is apples-to-apples.
"""

from __future__ import annotations

import cv2
import numpy as np


def _prep(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.max() > 1.5:           # already 0..255
        return a
    return a * 255.0


def _norm(m: np.ndarray) -> np.ndarray:
    m = np.abs(m).astype(np.float32)
    return m / m.max() if m.max() > 0 else m


def sobel(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    a = _prep(img)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=ksize)
    return _norm(np.sqrt(gx ** 2 + gy ** 2))


def scharr(img: np.ndarray) -> np.ndarray:
    a = _prep(img)
    gx = cv2.Scharr(a, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(a, cv2.CV_32F, 0, 1)
    return _norm(np.sqrt(gx ** 2 + gy ** 2))


def laplacian(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    a = _prep(img)
    return _norm(cv2.Laplacian(a, cv2.CV_32F, ksize=ksize))


def canny(img: np.ndarray, lo: int = 100, hi: int = 200) -> np.ndarray:
    a = _prep(img).astype(np.uint8)
    return (cv2.Canny(a, lo, hi) > 0).astype(np.float32)


ALL = {"sobel": sobel, "scharr": scharr, "laplacian": laplacian, "canny": canny}


def grabcut_baseline(
    img_bgr: np.ndarray,
    trimap: np.ndarray,
    n_iters: int = 5,
) -> np.ndarray:
    """Classical OpenCV GrabCut fg/bg segmentation seeded from trimap.

    Trimap convention: 255=fg seed, 0=bg seed, 128=unknown.

    Returns
    -------
    mask : (H, W) uint8, 1=foreground, 0=background
    """
    h, w = img_bgr.shape[:2]
    # Map trimap to cv2 GrabCut mask constants
    gc_mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)  # unknown → probable fg
    gc_mask[trimap == 0]   = cv2.GC_BGD   # definite background
    gc_mask[trimap == 255] = cv2.GC_FGD   # definite foreground
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    img_u8 = img_bgr if img_bgr.dtype == np.uint8 else (img_bgr * 255).astype(np.uint8)
    try:
        cv2.grabCut(img_u8, gc_mask, None, bgd_model, fgd_model, n_iters,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # Fallback: if GrabCut fails (e.g. no seeds), return trimap fg as mask
        return (trimap == 255).astype(np.uint8)
    return np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
