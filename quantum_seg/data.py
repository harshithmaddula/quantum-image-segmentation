"""Data loaders: USC-SIPI ``misc`` + BSDS500 (download, extract, read human GT).

misc has no segmentation masks -> used for encoding fidelity + no-reference metrics.
BSDS500 ships multiple human boundary annotations -> the supervised SOTA comparison
(ODS / OIS / Boundary-F1).  BSDS images are 321x481, so for the quantum engine we
crop/resize to a power-of-two (scale-matched) tile.
"""

from __future__ import annotations

import glob
import io
import os
import tarfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MISC_DIR = os.path.join(ROOT, "misc")
BSDS_DIR = os.path.join(ROOT, "data", "bsds500")

# Known BSDS500 mirrors (tried in order). Berkeley removed the original .tgz (404),
# so the BIDS GitHub mirror (zip) is primary.
BSDS_URLS = [
    "https://codeload.github.com/BIDS/BSDS500/zip/refs/heads/master",
    "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip",
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
]


# --------------------------------------------------------------------------- #
# size helpers
# --------------------------------------------------------------------------- #
def to_pow2(img: np.ndarray, size: int, mode: str = "crop") -> np.ndarray:
    """Return a size x size (power-of-two) version of a grayscale image."""
    if mode == "resize":
        return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((size, size),
                          Image.BILINEAR), dtype=float)
    h, w = img.shape                       # center crop
    y, x = max(0, (h - size) // 2), max(0, (w - size) // 2)
    crop = img[y:y + size, x:x + size]
    if crop.shape != (size, size):         # too small -> pad
        out = np.zeros((size, size), float)
        out[:crop.shape[0], :crop.shape[1]] = crop
        return out
    return crop.astype(float)


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def misc_paths(pattern: str = "*.tiff") -> list[str]:
    return sorted(glob.glob(os.path.join(MISC_DIR, pattern)))


def load_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=float)


def misc_gray(name: str, size: int | None = None, mode: str = "crop") -> np.ndarray:
    img = load_gray(os.path.join(MISC_DIR, name))
    return to_pow2(img, size, mode) if size else img


# --------------------------------------------------------------------------- #
# BSDS500
# --------------------------------------------------------------------------- #
def _find_bsds_root(base: str) -> str | None:
    """Return dir X such that X/BSDS500/data/images exists (handles any wrapper folder)."""
    if not os.path.isdir(base):
        return None
    for dp, dirs, _ in os.walk(base):
        if os.path.isdir(os.path.join(dp, "BSDS500", "data", "images")):
            return dp
    return None


def fetch_bsds500(dest: str = BSDS_DIR) -> str | None:
    """Download + extract BSDS500 once. Returns the dir containing BSDS500/, or None."""
    root = _find_bsds_root(dest)
    if root:
        return root
    os.makedirs(dest, exist_ok=True)
    for url in BSDS_URLS:
        try:
            print(f"[bsds] downloading {url} ...")
            with urllib.request.urlopen(url, timeout=120) as resp:
                blob = resp.read()
            if url.endswith(".tgz") or url.endswith(".tar.gz"):
                with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                    tar.extractall(dest)
            else:                                    # zip (GitHub mirrors)
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    zf.extractall(dest)
            root = _find_bsds_root(dest)
            if root:
                print(f"[bsds] extracted; root = {root}")
                return root
            print("[bsds] extracted but BSDS500/data/images not found")
        except Exception as e:                       # noqa: BLE001
            print(f"[bsds] failed: {e}")
    return None


def _read_gt_boundaries(mat_path: str) -> list[np.ndarray]:
    """Read all human boundary maps from a BSDS groundTruth .mat file."""
    from scipy.io import loadmat
    m = loadmat(mat_path)
    gt = m["groundTruth"]
    out = []
    for k in range(gt.shape[1]):
        try:
            cell = gt[0, k][0, 0]
            out.append(np.asarray(cell["Boundaries"]).astype(bool))
        except Exception:                           # noqa: BLE001
            continue
    return out


def bsds_pairs(root: str, split: str = "val", limit: int | None = None):
    """Yield (name, gray_image, [human_boundary_maps]) for a BSDS split."""
    img_dir = os.path.join(root, "BSDS500", "data", "images", split)
    gt_dir = os.path.join(root, "BSDS500", "data", "groundTruth", split)
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if limit:
        imgs = imgs[:limit]
    for p in imgs:
        name = os.path.splitext(os.path.basename(p))[0]
        gt_path = os.path.join(gt_dir, name + ".mat")
        if not os.path.exists(gt_path):
            continue
        gray = np.asarray(Image.open(p).convert("L"), dtype=float)
        yield name, gray, _read_gt_boundaries(gt_path)


def combine_gt(boundaries: list[np.ndarray]) -> np.ndarray:
    """Union of human annotators -> a single GT boundary map."""
    if not boundaries:
        return np.zeros((1, 1), bool)
    acc = np.zeros_like(boundaries[0], bool)
    for b in boundaries:
        acc |= b
    return acc


# --------------------------------------------------------------------------- #
# GrabCut-50
# --------------------------------------------------------------------------- #
GRABCUT_DIR = os.path.join(ROOT, "data", "grabcut50")

# Community mirrors — tried in order; manual fallback printed if all fail.
GRABCUT_URLS = [
    "https://github.com/saic-vul/GrabCut/archive/refs/heads/master.zip",
]


def _find_grabcut_dirs(base: str):
    """Walk base to locate (images_dir, trimaps_dir, gt_dir) by directory name."""
    img_dir = tri_dir = gt_dir = None
    for dp, _dirs, files in os.walk(base):
        if not any(f.lower().endswith((".jpg", ".jpeg", ".bmp", ".png")) for f in files):
            continue
        bn = os.path.basename(dp).lower()
        if bn in ("images", "imgs", "data_iscale") and img_dir is None:
            img_dir = dp
        elif bn in ("trimaps", "trimap", "data_trimap") and tri_dir is None:
            tri_dir = dp
        elif bn in ("gt", "groundtruth", "data_gt", "ground_truth", "masks") and gt_dir is None:
            gt_dir = dp
    return img_dir, tri_dir, gt_dir


def fetch_grabcut50(dest: str = GRABCUT_DIR) -> str | None:
    """Download + extract the GrabCut-50 dataset once (idempotent).

    Expected final layout under ``dest``:
        images/   — 50 images (.jpg / .bmp)
        trimaps/  — grayscale trimaps: 0=bg seed, 128=unknown, 255=fg seed
        gt/       — grayscale GT masks: 0=background, 255=foreground

    Returns ``dest`` on success, ``None`` if all downloads fail.
    """
    if (os.path.isdir(os.path.join(dest, "images")) and
            os.path.isdir(os.path.join(dest, "trimaps")) and
            os.path.isdir(os.path.join(dest, "gt"))):
        return dest
    os.makedirs(dest, exist_ok=True)
    for url in GRABCUT_URLS:
        try:
            print(f"[grabcut50] downloading {url} ...")
            with urllib.request.urlopen(url, timeout=120) as resp:
                blob = resp.read()
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                zf.extractall(dest)
            img_d, tri_d, gt_d = _find_grabcut_dirs(dest)
            if img_d and tri_d and gt_d:
                for name, src in [("images", img_d), ("trimaps", tri_d), ("gt", gt_d)]:
                    tgt = os.path.join(dest, name)
                    if os.path.abspath(src) != os.path.abspath(tgt):
                        if not os.path.exists(tgt):
                            os.rename(src, tgt)
                print(f"[grabcut50] extracted; dest = {dest}")
                return dest
            print("[grabcut50] extracted but could not locate images/trimaps/gt")
        except Exception as exc:                        # noqa: BLE001
            print(f"[grabcut50] download failed: {exc}")
    print(
        "\n[grabcut50] Automatic download failed. Place files manually:\n"
        f"  {dest}/images/   — original images (.jpg or .bmp)\n"
        f"  {dest}/trimaps/  — trimap masks (grayscale: 0=bg, 128=unknown, 255=fg)\n"
        f"  {dest}/gt/       — ground-truth fg masks (grayscale: 0=bg, 255=fg)\n"
    )
    return None


def grabcut_triplets(dest: str = GRABCUT_DIR, limit: int | None = None):
    """Yield (name, img_bgr, trimap, gt_mask) for each GrabCut-50 image.

    Yields
    ------
    name    : str image stem
    img_bgr : (H, W, 3) uint8 BGR
    trimap  : (H, W) uint8  — 0=bg seed, 128=unknown, 255=fg seed
    gt_mask : (H, W) bool   — True = foreground
    """
    import cv2 as _cv2
    img_dir = os.path.join(dest, "images")
    tri_dir = os.path.join(dest, "trimaps")
    gt_dir  = os.path.join(dest, "gt")
    if not os.path.isdir(img_dir):
        return
    all_imgs = sorted(
        p for pat in ("*.jpg", "*.jpeg", "*.bmp", "*.png")
        for p in glob.glob(os.path.join(img_dir, pat))
    )
    if limit:
        all_imgs = all_imgs[:limit]
    for ip in all_imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        tri_path = gt_path = None
        for ext in (".bmp", ".png", ".jpg", ".jpeg"):
            if tri_path is None and os.path.exists(os.path.join(tri_dir, stem + ext)):
                tri_path = os.path.join(tri_dir, stem + ext)
            if gt_path is None and os.path.exists(os.path.join(gt_dir, stem + ext)):
                gt_path = os.path.join(gt_dir, stem + ext)
        if tri_path is None or gt_path is None:
            continue
        img_bgr = _cv2.imread(ip)
        trimap  = _cv2.imread(tri_path, _cv2.IMREAD_GRAYSCALE)
        gt_raw  = _cv2.imread(gt_path,  _cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or trimap is None or gt_raw is None:
            continue
        yield stem, img_bgr, trimap, gt_raw > 127
