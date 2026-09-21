"""Experiment driver -> the mandatory-metric tables vs SOTA.

Stages (each independent, so you get results even if BSDS download fails):
  1. NEQR encoding fidelity (PSNR/SSIM/MSE/MAE/EPI) + quantum resource counts on misc crops
     -> proves spatial information is preserved by the quantum representation.
  2. QHED vs classical baselines on misc (no ground truth): edge density + tolerance agreement.
  3. BSDS500 supervised boundary SOTA: ODS / OIS / Boundary-F1 / Pratt FOM / Hausdorff for
     QHED and each classical baseline, beside published literature anchors.

Usage:
  python3 -m quantum_seg.run_experiments                      # stages 1-2 (no network)
  python3 -m quantum_seg.run_experiments --bsds --bsds-limit 20   # + stage 3
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from . import baselines as B
from . import data as D
from . import metrics as M
from . import neqr
from .qsobel import qhed_edges, qhed_edges_refined
from . import qaoa_seg as QS
from .figures import grabcut_panel, misc_fgbg_panel
from .superpixel import auto_trimap

RESULTS = os.path.join(D.ROOT, "results")

# Published BSDS500 boundary ODS anchors (approx, for context only).
LIT_ANCHORS = {"Human": 0.80, "Sobel(lit)": 0.54, "Canny(lit)": 0.60,
               "gPb": 0.73, "StructuredEdges": 0.75, "HED(deep)": 0.79}


def _fmt(x):
    if isinstance(x, float):
        return "inf" if x == float("inf") else f"{x:.4f}"
    return str(x)


def _table(headers, rows):
    cols = [headers] + [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(cols[r][c]) for r in range(len(cols))) for c in range(len(headers))]
    line = lambda cells: "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), sep] + [line(r) for r in cols[1:]])


# --------------------------------------------------------------------------- #
# stage 1: NEQR fidelity + resources
# --------------------------------------------------------------------------- #
def stage_fidelity(images, sizes=(8, 16, 32)) -> str:
    rows = []
    for name in images:
        for s in sizes:
            patch = D.misc_gray(name, size=s, mode="crop").astype(np.uint8)
            t0 = time.time()
            rec = neqr.roundtrip(patch, q=8)
            dt = time.time() - t0
            fid = M.fidelity_report(patch, rec)
            res = neqr.resource_report(patch)
            rows.append([name, f"{s}x{s}", res["num_qubits"], res.get("gate_count", "-"),
                         res.get("depth", "-"), fid["PSNR"], fid["SSIM"], fid["MSE"],
                         fid["EPI"], f"{dt:.2f}s"])
    return _table(["image", "size", "qubits", "gates", "depth",
                   "PSNR", "SSIM", "MSE", "EPI", "sim_time"], rows)


# --------------------------------------------------------------------------- #
# stage 2: QHED vs baselines on misc (no GT)
# --------------------------------------------------------------------------- #
def stage_misc_edges(images, size=128) -> str:
    rows = []
    for name in images:
        img = D.misc_gray(name, size=size, mode="crop")
        q = qhed_edges(img)
        for bname, fn in B.ALL.items():
            be = fn(img)
            agree = M.tolerance_agreement(q, be, tol=1)
            rows.append([name, f"{size}x{size}", "QHED-vs-" + bname,
                         M.edge_density(q >= np.quantile(q, 0.85)),
                         M.edge_density(be >= (be.max() * 0.5 if bname != "canny" else 0.5)),
                         agree])
    return _table(["image", "size", "comparison", "QHED_edge_density",
                   "baseline_edge_density", "tol1_agreement"], rows)


# --------------------------------------------------------------------------- #
# stage 3: BSDS500 supervised boundary SOTA
# --------------------------------------------------------------------------- #
def stage_bsds(limit=20, size=256, split="val", tol=2) -> str:
    root = D.fetch_bsds500()
    if root is None:
        return "_BSDS500 unavailable (download failed) -- skipped. Re-run with network access._"

    methods = {"QHED-raw": lambda g: qhed_edges(g),
               "QHED-denoise": lambda g: qhed_edges_refined(g)}   # bilateral->QHED
    methods.update({k: (lambda g, fn=fn: fn(g)) for k, fn in B.ALL.items()})
    sweeps = {m: [] for m in methods}
    extra = {m: {"PrattFOM": [], "Hausdorff95": []} for m in methods}

    n = 0
    for name, gray, gts in D.bsds_pairs(root, split=split, limit=limit):
        if not gts:
            continue
        g = D.to_pow2(gray, size, mode="resize")
        gt = D.to_pow2(D.combine_gt(gts).astype(np.uint8) * 255, size, mode="resize") > 64
        if gt.sum() == 0:
            continue
        for m, fn in methods.items():
            emap = fn(g)
            sw = M.boundary_f_sweep(emap, gt, tol=tol)
            sweeps[m].append(sw)
            rep = M.boundary_report(emap, gt, tol=tol)
            extra[m]["PrattFOM"].append(rep["PrattFOM"])
            extra[m]["Hausdorff95"].append(rep["Hausdorff95"])
        n += 1

    rows = []
    for m in methods:
        oo = M.dataset_ods_ois(sweeps[m])
        rows.append([m, oo["ODS"], oo["OIS"],
                     float(np.mean(extra[m]["PrattFOM"])),
                     float(np.median([h for h in extra[m]["Hausdorff95"] if np.isfinite(h)] or [np.inf]))])
    tbl = _table(["method", "ODS", "OIS", "PrattFOM", "Hausdorff95(med)"], rows)
    anchors = _table(["literature_anchor", "ODS(approx)"],
                     [[k, v] for k, v in LIT_ANCHORS.items()])
    return (f"Evaluated on {n} BSDS500 '{split}' images at {size}x{size} (resize), tol={tol}px.\n\n"
            + tbl + "\n\n**Published SOTA anchors (context):**\n\n" + anchors)


# --------------------------------------------------------------------------- #
# stage 4: QAOA fg/bg segmentation on GrabCut-50
# --------------------------------------------------------------------------- #
_QAOA_CAVEAT = """\
**Quantum limitations and honest context**

The QAOA circuit operates on `n_free` qubits, one per free (non-seed) superpixel.
With `n_segments=16` and typical GrabCut-50 trimaps having ~30-50% seed pixels,
`n_free ≈ 8-12` qubits — well within the `lightning.qubit` exact statevector
simulator (tractable to ~25 qubits).  Full-image QAOA (one qubit/pixel) requires
near-future fault-tolerant quantum hardware.

QAOA depth p=1 provides a constant-factor approximation guarantee; p=2 improves
the approximation ratio at the cost of 4 variational parameters instead of 2.

Quantum integration: QHED edge strengths computed by the quantum Hadamard circuit
(Stage 2) modulate J_ij pairwise terms — this is the genuine quantum feature of
Stage 4.  SLIC superpixels and GMM fitting remain classical preprocessing steps.
"""


def stage_grabcut(
    limit: int = 10,
    n_segments: int = 16,
    p: int = 1,
    beta: float = 10.0,
    use_qhed: bool = True,
    maxiter: int = 200,
) -> str:
    """Stage 4: QAOA fg/bg segmentation vs classical GrabCut on GrabCut-50."""
    dest = D.fetch_grabcut50()
    if dest is None:
        return (
            "_GrabCut-50 unavailable (download failed). "
            "Place files in `data/grabcut50/` and re-run with `--grabcut`._"
        )

    rows = []
    for name, img_bgr, trimap, gt_mask in D.grabcut_triplets(dest, limit=limit):
        # QAOA segmentation
        result = QS.segment_image(
            img_bgr, trimap,
            n_segments=n_segments, p=p, beta=beta,
            maxiter=maxiter, use_qhed=use_qhed,
        )
        qaoa_mask = result["mask"]

        # Classical GrabCut baseline
        gc_mask = B.grabcut_baseline(img_bgr, trimap)

        # Metrics on unknown region (standard GrabCut evaluation protocol)
        unknown = (trimap == 128)
        if unknown.sum() > 0:
            m_qaoa = M.segmentation_report(qaoa_mask[unknown], gt_mask[unknown])
            m_gc   = M.segmentation_report(gc_mask[unknown],   gt_mask[unknown])
        else:
            m_qaoa = M.segmentation_report(qaoa_mask, gt_mask)
            m_gc   = M.segmentation_report(gc_mask,   gt_mask)

        # Save 6-panel figure
        grabcut_panel(
            name, img_bgr, trimap, qaoa_mask, gc_mask, gt_mask,
            qhed_map=result["qhed_map"],
            metrics_qaoa=m_qaoa, metrics_gc=m_gc,
        )

        n_free = result["n_free"]
        rows.append(["QAOA (quantum)",   name, m_qaoa["IoU"], m_qaoa["Dice"],
                     m_qaoa["PixelAcc"], n_free, f"{result['sim_time_s']:.1f}s"])
        rows.append(["GrabCut (classical)", name, m_gc["IoU"], m_gc["Dice"],
                     m_gc["PixelAcc"], "-", "-"])

    if not rows:
        return "_No GrabCut-50 images found. Check data/grabcut50/ layout._"

    tbl = _table(
        ["method", "image", "IoU", "Dice", "PixelAcc", "n_qubits", "sim_time"],
        rows,
    )

    # Summary stats
    qaoa_rows = [r for r in rows if "QAOA" in r[0]]
    gc_rows   = [r for r in rows if "GrabCut" in r[0]]
    def _mean_iou(rs):
        vals = [r[2] for r in rs if isinstance(r[2], float)]
        return float(np.mean(vals)) if vals else 0.0

    summary = (
        f"\nMean IoU (unknown region, {len(qaoa_rows)} images): "
        f"QAOA={_mean_iou(qaoa_rows):.4f}  GrabCut={_mean_iou(gc_rows):.4f}\n"
    )

    return f"Evaluated on {len(qaoa_rows)} GrabCut-50 images.\n\n{tbl}{summary}\n{_QAOA_CAVEAT}"


# --------------------------------------------------------------------------- #
# stage 5: QAOA fg/bg on misc (auto-trimap, no ground truth)
# --------------------------------------------------------------------------- #
def _load_misc_bgr(name: str, size: int) -> np.ndarray:
    """Load a misc image as (H, W, 3) uint8 BGR, resized to size×size."""
    import cv2 as _cv2
    path = os.path.join(D.MISC_DIR, name)
    img  = _cv2.imread(path, _cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    # Grayscale → BGR
    if img.ndim == 2:
        img = _cv2.cvtColor(img, _cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:           # RGBA → BGR
        img = _cv2.cvtColor(img, _cv2.COLOR_BGRA2BGR)
    else:
        img = _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR)   # PIL/TIFF loads RGB
    return _cv2.resize(img, (size, size))


def stage_misc_fgbg(
    images: list[str] | None = None,
    size: int = 128,
    n_segments: int = 12,
    p: int = 1,
    use_qhed: bool = True,
    maxiter: int = 150,
) -> str:
    """Stage 5: QAOA fg/bg on misc images using an auto-generated center/border trimap.

    No ground truth is available -- metrics are computed against the auto-trimap
    fg region itself and are marked with * to indicate this caveat.

    Parameters
    ----------
    images     : list of misc filenames; defaults to all TIFF files in misc/
    size       : resize target (power-of-two recommended; 128 is fast, 256 is higher quality)
    n_segments : SLIC superpixel count (≈ QAOA qubit count)
    p          : QAOA circuit depth
    use_qhed   : integrate QHED quantum edge map into pairwise term
    """
    import glob as _glob
    if images is None:
        images = sorted(os.path.basename(p)
                        for p in _glob.glob(os.path.join(D.MISC_DIR, "*.tiff")))
    if not images:
        return "_No misc TIFF images found in misc/._"

    rows = []
    for name in images:
        try:
            img_bgr = _load_misc_bgr(name, size)
        except FileNotFoundError:
            continue
        H, W = img_bgr.shape[:2]
        trimap = auto_trimap(H, W)

        result  = QS.segment_image(img_bgr, trimap, n_segments=n_segments,
                                   p=p, maxiter=maxiter, use_qhed=use_qhed)
        gc_mask = B.grabcut_baseline(img_bgr, trimap)

        # Metrics on unknown region vs auto-trimap fg (no human GT)
        unknown   = (trimap == 128)
        gt_auto   = (trimap == 255)       # auto fg seed region used as proxy GT
        # Extend gt_auto to unknown region based on distance from center
        # (pixels closer to center ellipse → fg label for metric purposes)
        import cv2 as _cv2
        gt_full = _cv2.resize(
            (gt_auto.astype(np.uint8) * 255),
            (W, H),
        ) > 127
        # Use thresholded distance as proxy GT for unknown pixels
        cy, cx = H // 2, W // 2
        yy, xx = np.ogrid[:H, :W]
        dist_fg = np.sqrt(((yy - cy) / (H * 0.28 + 1)) ** 2 + ((xx - cx) / (W * 0.28 + 1)) ** 2)
        proxy_gt = dist_fg < 1.0          # inside the fg ellipse radius

        m_qaoa = M.segmentation_report(result["mask"][unknown], proxy_gt[unknown])
        m_gc   = M.segmentation_report(gc_mask[unknown],        proxy_gt[unknown])

        misc_fgbg_panel(name, img_bgr, trimap, result["mask"], gc_mask,
                        qhed_map=result["qhed_map"],
                        metrics_qaoa=m_qaoa, metrics_gc=m_gc)

        rows.append(["QAOA", name, m_qaoa["IoU"], m_qaoa["Dice"],
                     m_qaoa["PixelAcc"], result["n_free"], f"{result['sim_time_s']:.1f}s"])
        rows.append(["GrabCut", name, m_gc["IoU"], m_gc["Dice"],
                     m_gc["PixelAcc"], "-", "-"])

    if not rows:
        return "_No misc images processed._"

    tbl = _table(["method", "image", "IoU*", "Dice*", "PixelAcc*", "n_qubits", "sim_time"], rows)
    note = (
        "\n_\\* metrics are vs the auto-generated center-ellipse trimap, NOT human GT. "
        "Use as a relative comparison between QAOA and GrabCut only._\n"
        f"\nFigures saved to `results/figures/misc_fgbg_*.png`\n"
    )
    return f"Evaluated on {len(rows)//2} misc images at {size}×{size}, n_segments={n_segments}.\n\n{tbl}{note}"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsds", action="store_true", help="download + evaluate BSDS500")
    ap.add_argument("--bsds-limit", type=int, default=20)
    ap.add_argument("--bsds-size", type=int, default=256)
    ap.add_argument("--misc-size", type=int, default=128)
    ap.add_argument("--grabcut", action="store_true",
                    help="download GrabCut-50 and run QAOA fg/bg segmentation (Stage 4)")
    ap.add_argument("--grabcut-limit", type=int, default=10,
                    help="number of GrabCut-50 images to process")
    ap.add_argument("--grabcut-segments", type=int, default=16,
                    help="SLIC superpixel count (≈ QAOA qubit count)")
    ap.add_argument("--grabcut-depth", type=int, default=1,
                    help="QAOA circuit depth p (1 or 2)")
    ap.add_argument("--grabcut-no-qhed", action="store_true",
                    help="disable QHED edge integration (ablation)")
    ap.add_argument("--misc-fgbg", action="store_true",
                    help="run QAOA fg/bg on misc images with auto-trimap (Stage 5)")
    ap.add_argument("--misc-fgbg-size", type=int, default=128,
                    help="resize misc images to this square size before processing")
    ap.add_argument("--misc-fgbg-segments", type=int, default=12,
                    help="SLIC superpixel count for misc fg/bg stage")
    ap.add_argument("--misc-fgbg-names", nargs="*", default=None,
                    help="specific misc filenames to process (default: all TIFFs)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "metrics_report.md"))
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    fid_imgs = ["5.1.12.tiff", "4.2.03.tiff", "boat.512.tiff"]   # incl. texture (mandrill)
    edge_imgs = ["5.1.12.tiff", "boat.512.tiff", "4.2.03.tiff", "house.tiff"]

    parts = ["# Quantum Image Segmentation -- Mandatory Metrics vs SOTA\n",
             "## 1. NEQR encoding fidelity + quantum resources (spatial-preservation gate)\n",
             "_Lossless representation: exact reconstruction => PSNR=inf, SSIM=1, MSE=0 "
             "(every boundary/texture pixel preserved). Resource columns substantiate the "
             "'quantum' claim (per Ruan 2021)._\n",
             stage_fidelity(fid_imgs)]
    print("[stage 1] fidelity done")

    parts += ["\n## 2. QHED vs classical baselines on misc (no ground truth)\n",
              "_misc has no masks -> tolerance-aware agreement with classical edges + edge "
              "density. Supervised SOTA numbers are in stage 3 (BSDS)._\n",
              stage_misc_edges(edge_imgs, size=args.misc_size)]
    print("[stage 2] misc edges done")

    parts += ["\n## 3. BSDS500 supervised boundary SOTA (ODS / OIS / Boundary-F1 / Pratt / Hausdorff)\n"]
    if args.bsds:
        parts.append(stage_bsds(limit=args.bsds_limit, size=args.bsds_size))
        print("[stage 3] bsds done")
    else:
        parts.append("_Run with `--bsds` to download BSDS500 and compute supervised SOTA metrics._")

    parts += ["\n## 4. QAOA Fg/Bg Segmentation on GrabCut-50\n"]
    if args.grabcut:
        parts.append(stage_grabcut(
            limit=args.grabcut_limit,
            n_segments=args.grabcut_segments,
            p=args.grabcut_depth,
            use_qhed=not args.grabcut_no_qhed,
        ))
        print("[stage 4] grabcut done")
    else:
        parts.append(
            "_Run with `--grabcut` to download GrabCut-50 and evaluate QAOA fg/bg segmentation._"
        )

    parts += ["\n## 5. QAOA Fg/Bg on Misc Images (auto-trimap, no human GT)\n"]
    if args.misc_fgbg:
        parts.append(stage_misc_fgbg(
            images=args.misc_fgbg_names,
            size=args.misc_fgbg_size,
            n_segments=args.misc_fgbg_segments,
            use_qhed=not args.grabcut_no_qhed,
        ))
        print("[stage 5] misc fgbg done")
    else:
        parts.append(
            "_Run with `--misc-fgbg` to test QAOA fg/bg segmentation on misc images._"
        )

    report = "\n".join(parts) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nwrote {args.out}\n")
    print(report)


if __name__ == "__main__":
    main()
