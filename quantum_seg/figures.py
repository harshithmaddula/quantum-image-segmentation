"""Boundary-overlay figures for the report.

misc  (no GT)  : [ original | QHED edges | Sobel edges ]
BSDS500 (GT)   : [ original | QHED edges | human GT | QHED(green)+GT(red) overlay ]

Saved as PNGs under results/figures/.  Uses the Agg backend (no display needed).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import baselines as B
from . import data as D
from . import metrics as M
from .qsobel import qhed_edges, qhed_edges_refined

FIGDIR = os.path.join(D.ROOT, "results", "figures")


def _bin(edge_map, q=0.85):
    m = np.asarray(edge_map, float)
    return m >= np.quantile(m, q) if m.max() > 0 else m.astype(bool)


def _overlay(gray, qhed_bin, gt_bin):
    """Grayscale with QHED edges in green and GT boundaries in red (overlap=yellow)."""
    g = np.asarray(gray, float)
    g = (g - g.min()) / (np.ptp(g) + 1e-9)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[gt_bin] = [1, 0, 0]          # GT red
    rgb[qhed_bin] = [0, 1, 0]        # QHED green
    rgb[qhed_bin & gt_bin] = [1, 1, 0]  # overlap yellow
    return rgb


def misc_panel(name: str, size: int = 128):
    img = D.misc_gray(name, size=size, mode="crop")
    q = qhed_edges(img)
    qn = qhed_edges_refined(img)
    s = B.sobel(img)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title(f"{name}  ({size}x{size})")
    ax[1].imshow(_bin(q), cmap="gray"); ax[1].set_title("QHED raw (quantum)")
    ax[2].imshow(_bin(qn), cmap="gray"); ax[2].set_title("QHED+denoise (refined)")
    ax[3].imshow(_bin(s), cmap="gray"); ax[3].set_title("Sobel (classical)")
    for a in ax: a.axis("off")
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"misc_{os.path.splitext(name)[0]}.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def bsds_panel(name, gray, gts, size: int = 256, tol: int = 2):
    g = D.to_pow2(gray, size, mode="resize")
    gt = D.to_pow2(D.combine_gt(gts).astype(np.uint8) * 255, size, mode="resize") > 64
    q = qhed_edges_refined(g)                          # refined (NMS) -- the headline method
    rep = M.boundary_report(q, gt, tol=tol)            # use best-F threshold for the visual
    qbin = q >= rep["thr"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ax[0].imshow(g, cmap="gray"); ax[0].set_title(f"BSDS {name}")
    ax[1].imshow(qbin, cmap="gray"); ax[1].set_title(f"QHED+denoise  (F={rep['BF_bestF']:.3f})")
    ax[2].imshow(gt, cmap="gray"); ax[2].set_title("human GT (union)")
    ax[3].imshow(_overlay(g, qbin, gt)); ax[3].set_title("QHED=green  GT=red  hit=yellow")
    for a in ax: a.axis("off")
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"bsds_{name}.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return out


def _trimap_rgb(trimap: np.ndarray) -> np.ndarray:
    """Visualise trimap: green=fg seeds, red=bg seeds, grey=unknown."""
    out = np.stack([np.full_like(trimap, 0.5, float)] * 3, axis=-1)
    out[trimap == 255] = [0, 1, 0]   # fg seed → green
    out[trimap == 0]   = [1, 0, 0]   # bg seed → red
    return out


def grabcut_panel(
    name: str,
    img_bgr: np.ndarray,
    trimap: np.ndarray,
    qaoa_mask: np.ndarray,
    gc_mask: np.ndarray,
    gt_mask: np.ndarray,
    qhed_map: np.ndarray | None = None,
    metrics_qaoa: dict | None = None,
    metrics_gc: dict | None = None,
) -> str:
    """Six-panel comparison: Original | Trimap | QHED edges | QAOA mask | GrabCut | GT.

    Saves to results/figures/grabcut_<name>.png and returns the path.
    """
    os.makedirs(FIGDIR, exist_ok=True)
    img_rgb = img_bgr[:, :, ::-1]          # BGR → RGB for matplotlib
    qhed_show = qhed_map if qhed_map is not None else np.zeros(img_rgb.shape[:2])

    def _mask_title(label, metrics):
        if metrics is None:
            return label
        return f"{label}  IoU={metrics['IoU']:.3f}  Dice={metrics['Dice']:.3f}"

    fig, ax = plt.subplots(1, 6, figsize=(24, 4))
    ax[0].imshow(img_rgb);                  ax[0].set_title(name)
    ax[1].imshow(_trimap_rgb(trimap));      ax[1].set_title("Trimap (G=fg, R=bg)")
    ax[2].imshow(qhed_show, cmap="hot");    ax[2].set_title("QHED edges (quantum)")
    ax[3].imshow(qaoa_mask, cmap="RdYlGn"); ax[3].set_title(_mask_title("QAOA (quantum)", metrics_qaoa))
    ax[4].imshow(gc_mask, cmap="RdYlGn");   ax[4].set_title(_mask_title("GrabCut (classical)", metrics_gc))
    ax[5].imshow(np.asarray(gt_mask, np.uint8), cmap="RdYlGn"); ax[5].set_title("Ground truth")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    out = os.path.join(FIGDIR, f"grabcut_{name}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def misc_fgbg_panel(
    name: str,
    img_bgr: np.ndarray,
    trimap: np.ndarray,
    qaoa_mask: np.ndarray,
    gc_mask: np.ndarray,
    qhed_map: np.ndarray | None = None,
    metrics_qaoa: dict | None = None,
    metrics_gc: dict | None = None,
) -> str:
    """Five-panel figure for misc (no ground truth): Original | Trimap | QHED | QAOA | GrabCut.

    Saves to results/figures/misc_fgbg_<name>.png and returns the path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    img_rgb  = img_bgr[:, :, ::-1]
    qhed_show = qhed_map if qhed_map is not None else np.zeros(img_rgb.shape[:2])

    def _title(label, metrics):
        if not metrics:
            return label
        return f"{label}\nIoU*={metrics['IoU']:.3f}  Dice*={metrics['Dice']:.3f}"

    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    ax[0].imshow(img_rgb);                         ax[0].set_title(name)
    ax[1].imshow(_trimap_rgb(trimap));             ax[1].set_title("Auto-trimap\n(G=fg seed, R=bg seed)")
    ax[2].imshow(qhed_show, cmap="hot");           ax[2].set_title("QHED edges\n(quantum)")
    ax[3].imshow(qaoa_mask, cmap="RdYlGn");        ax[3].set_title(_title("QAOA (quantum)", metrics_qaoa))
    ax[4].imshow(gc_mask,   cmap="RdYlGn");        ax[4].set_title(_title("GrabCut (classical)", metrics_gc))
    for a in ax:
        a.axis("off")
    fig.suptitle("* metrics vs auto-trimap fg/bg (no human GT)", fontsize=8, style="italic")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05, hspace=0.0)
    out = os.path.join(FIGDIR, f"misc_fgbg_{os.path.splitext(name)[0]}.png")
    fig.savefig(out, dpi=60, bbox_inches="tight")  # Further reduced: figsize, DPI, and bbox_inches for memory efficiency
    plt.close(fig)
    plt.close("all")  # Close all figures
    del fig, ax
    import gc
    gc.collect()  # Force garbage collection
    return out


def main(misc_names=("5.1.12.tiff", "boat.512.tiff", "4.2.03.tiff", "house.tiff"),
         n_bsds: int = 4):
    os.makedirs(FIGDIR, exist_ok=True)
    made = []
    for nm in misc_names:
        made.append(misc_panel(nm))
    root = D.fetch_bsds500()
    if root:
        for name, gray, gts in D.bsds_pairs(root, split="val", limit=n_bsds):
            if gts:
                made.append(bsds_panel(name, gray, gts))
    with open(os.path.join(FIGDIR, "_index.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(made))
    print(f"wrote {len(made)} figures to {FIGDIR}")
    return made


if __name__ == "__main__":
    main()
