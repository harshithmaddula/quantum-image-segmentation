"""Standalone fg/bg segmentation for all images in misc/.

Run:     python segment_misc.py
Outputs: results/figures/misc_fgbg_<name>.png  for every TIFF in misc/
         5-panel figure per image: Original | Trimap | QHED edges | QAOA mask | GrabCut mask
"""

import sys
import os
import glob
import numpy as np
import gc  # garbage collection for memory management

# --------------------------------------------------------------------------- #
# User-configurable settings  (edit these lines to change behaviour)
# --------------------------------------------------------------------------- #
SIZE       = 128   # resize each image to SIZE×SIZE before processing
               # 128 = fast (~8s/image),  256 = higher quality (~30s/image)
N_SEGMENTS = 12    # SLIC superpixel count ≈ QAOA qubit count  (keep ≤ 20)
USE_QHED   = True  # True: quantum QHED edges feed into pairwise term
               # False: skip QHED for faster classical-only comparison
QAOA_DEPTH = 1     # QAOA circuit depth p  (1 = fast, 2 = better approx ratio)

# --------------------------------------------------------------------------- #
# Dependency check — must run before importing quantum_seg (which also needs cv2)
# --------------------------------------------------------------------------- #
try:
    import cv2
    _ = cv2.ximgproc          # SLIC superpixels live in ximgproc
except (ModuleNotFoundError, AttributeError):
    print("\nERROR: opencv-contrib-python is not installed in this Python environment.\n")
    print("Fix — choose one option:")
    print("  Option 1 (fastest):  deactivate the wrong venv and use system Python")
    print("    > deactivate")
    print("    > python segment_misc.py\n")
    print("  Option 2: install in the current environment")
    print("    > pip install opencv-contrib-python")
    print("    NOTE: do NOT install 'opencv-python' — it lacks cv2.ximgproc for SLIC\n")
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Import pipeline components
# --------------------------------------------------------------------------- #
from quantum_seg.qaoa_seg   import segment_image
from quantum_seg.baselines  import grabcut_baseline
from quantum_seg.superpixel import auto_trimap
from quantum_seg.figures    import misc_fgbg_panel
from quantum_seg.metrics    import segmentation_report
import quantum_seg.data as D

FIGDIR   = os.path.join(D.ROOT, "results", "figures")
MISC_DIR = D.MISC_DIR


# --------------------------------------------------------------------------- #
# Image loader (grayscale + color + RGBA all handled)
# --------------------------------------------------------------------------- #
def _load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:                          # grayscale → BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                    # RGBA → BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:                                      # RGB (PIL/TIFF) → BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (SIZE, SIZE))


# --------------------------------------------------------------------------- #
# Proxy ground truth for misc (no human annotations available)
# --------------------------------------------------------------------------- #
def _proxy_gt(H: int, W: int) -> np.ndarray:
    """Center ellipse region used as proxy fg GT (matches auto_trimap fg zone)."""
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    return (((yy - cy) / (H * 0.28 + 1)) ** 2 +
            ((xx - cx) / (W * 0.28 + 1)) ** 2) < 1.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(FIGDIR, exist_ok=True)

    images = sorted(glob.glob(os.path.join(MISC_DIR, "*.tiff")))
    if not images:
        print(f"No TIFF files found in {MISC_DIR}")
        sys.exit(0)

    N = len(images)
    print(f"\nQuantum fg/bg segmentation on {N} misc images")
    print(f"Settings: SIZE={SIZE}  N_SEGMENTS={N_SEGMENTS}  "
          f"USE_QHED={USE_QHED}  QAOA_DEPTH={QAOA_DEPTH}")
    print(f"Figures will be saved to: {FIGDIR}\n")

    rows = []
    for idx, path in enumerate(images, 1):
        name = os.path.basename(path)
        print(f"[{idx:2d}/{N}] {name:<24s}", end=" ", flush=True)
        gc.collect()  # Free memory before processing new image

        try:
            img_bgr = _load_bgr(path)
        except FileNotFoundError:
            print("SKIP (file not readable)")
            continue
        except Exception as e:
            print(f"SKIP ({type(e).__name__})")
            gc.collect()  # Free memory before continuing
            continue
        H, W   = img_bgr.shape[:2]
        trimap = auto_trimap(H, W)

        # QAOA quantum segmentation
        result = segment_image(
            img_bgr, trimap,
            n_segments=N_SEGMENTS,
            p=QAOA_DEPTH,
            use_qhed=USE_QHED,
            maxiter=150,
        )

        # Classical GrabCut baseline
        gc_mask = grabcut_baseline(img_bgr, trimap)

        # Metrics on the unknown region vs proxy GT
        unknown  = (trimap == 128)
        proxy    = _proxy_gt(H, W)
        m_q  = segmentation_report(result["mask"][unknown], proxy[unknown])
        m_gc = segmentation_report(gc_mask[unknown],        proxy[unknown])

        # Save 5-panel figure
        fig_path = misc_fgbg_panel(
            name, img_bgr, trimap,
            result["mask"], gc_mask,
            qhed_map=result["qhed_map"],
            metrics_qaoa=m_q,
            metrics_gc=m_gc,
        )

        t = result["sim_time_s"]
        print(f"QAOA IoU*={m_q['IoU']:.3f}  GC IoU*={m_gc['IoU']:.3f}  "
              f"qubits={result['n_free']}  {t:.1f}s")
        rows.append((name, m_q["IoU"], m_gc["IoU"], result["n_free"], t))
        gc.collect()  # Free quantum simulator and matplotlib memory

    if not rows:
        print("No images processed.")
        return

    # Summary table
    w = 26
    print(f"\n{'─' * 72}")
    print(f"  {'image':<{w}}  {'QAOA IoU*':>9}  {'GC IoU*':>7}  {'n_qubits':>8}  {'time':>6}")
    print(f"{'─' * 72}")
    for name, iq, igc, nq, t in rows:
        print(f"  {name:<{w}}  {iq:>9.3f}  {igc:>7.3f}  {nq:>8}  {t:>5.1f}s")
    print(f"{'─' * 72}")

    mean_q  = float(np.mean([r[1] for r in rows]))
    mean_gc = float(np.mean([r[2] for r in rows]))
    print(f"  {'MEAN':<{w}}  {mean_q:>9.3f}  {mean_gc:>7.3f}")
    print(f"\n* metrics vs auto-generated center-ellipse proxy — not human GT.")
    print(f"  Use as QAOA-vs-GrabCut relative comparison only.\n")
    print(f"Figures saved → {FIGDIR}")
    print(f"Open any PNG to inspect: Original | Trimap | QHED | QAOA | GrabCut\n")


if __name__ == "__main__":
    main()
