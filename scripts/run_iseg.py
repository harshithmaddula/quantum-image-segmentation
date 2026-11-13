"""Run the graph-cut pipeline on the local iSeg image set without scoring."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantum_seg import qaoa_seg
from quantum_seg.qubo_solvers import solve_graph_cut
from quantum_seg.superpixel import auto_trimap, classify_seeds


DEFAULT_INPUT = ROOT / "data" / "iseg" / "images"
DEFAULT_OUTPUT = ROOT / "results" / "iseg"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"}
EVALUATION_NOTE = "no ground truth, ellipse trimap: robustness and speed only"

CSV_FIELDS = [
    "image_id", "filename", "original_width", "original_height", "status",
    "error_type", "error_message", "n_superpixels", "n_free_nodes",
    "total_latency_s", "edge_map_s", "slic_s", "hybrid_features_s",
    "gmm_s", "qubo_build_s", "solver_s", "max_abs_h_eff", "median_abs_h_eff",
    "max_J", "max_abs_h_over_max_J", "free_labels_changed_with_J_zero",
    "foreground_fraction_all_pixels", "foreground_fraction_free_pixels",
    "free_region_class", "collapse_flag", "foreground_seed_pixels",
    "background_seed_pixels", "unknown_pixels", "sklearn_convergence_warnings",
    "evaluation_note",
]


def load_bgr(path: Path) -> tuple[np.ndarray, int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path.name}")
    height, width = image.shape[:2]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.resize(image, (128, 128)), width, height


def _blank_row(path: Path, evaluation_note: str) -> dict:
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "image_id": path.stem,
        "filename": path.name,
        "status": "error",
        "evaluation_note": evaluation_note,
    })
    return row


def _write_overlay(path: Path, image_bgr: np.ndarray, trimap: np.ndarray,
                   edge_map: np.ndarray, mask: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    trimap_rgb = np.full((*trimap.shape, 3), 0.5, dtype=float)
    trimap_rgb[trimap == 255] = (0.0, 0.8, 0.2)
    trimap_rgb[trimap == 0] = (0.9, 0.15, 0.1)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(rgb); axes[0].set_title("Input 128x128")
    axes[1].imshow(trimap_rgb); axes[1].set_title("Ellipse trimap")
    axes[2].imshow(edge_map, cmap="magma"); axes[2].set_title("QHED/Sobel feature")
    axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[3].set_title("Graph-cut mask")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{path.stem} | {EVALUATION_NOTE}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _run_one(path: Path, out_dir: Path, overlay_path: Path | None,
             note: str) -> dict:
    row = _blank_row(path, note)
    try:
        image, original_width, original_height = load_bgr(path)
        row["original_width"] = original_width
        row["original_height"] = original_height
        trimap = auto_trimap(128, 128)
        row["foreground_seed_pixels"] = int(np.count_nonzero(trimap == 255))
        row["background_seed_pixels"] = int(np.count_nonzero(trimap == 0))
        row["unknown_pixels"] = int(np.count_nonzero(trimap == 128))

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            start = time.perf_counter()
            result = qaoa_seg.segment_image(
                image, trimap, n_segments=12, p=1, use_qhed=True,
                solver="graphcut",
            )
            measured_latency = time.perf_counter() - start

        labels = result["labels"]
        free_nodes = np.flatnonzero(classify_seeds(labels, trimap, result["n_sp"]) == -1)
        free_pixels = np.isin(labels, free_nodes)
        free_labels = np.asarray(result["free_assignment"], dtype=np.int32)
        zeroed_assignment, _ = solve_graph_cut(
            result["h_eff"], {edge: 0.0 for edge in result["J_free"]}
        )

        h_values = np.asarray(result["h_eff"], dtype=float)
        max_abs_h = float(np.max(np.abs(h_values))) if h_values.size else 0.0
        median_abs_h = float(np.median(np.abs(h_values))) if h_values.size else 0.0
        j_values = [abs(float(value)) for value in result["J_free"].values()]
        max_j = max(j_values, default=0.0)
        free_mask_values = result["mask"][free_pixels]
        free_fg_fraction = (float(np.mean(free_mask_values))
                            if free_mask_values.size else float("nan"))
        if not free_mask_values.size:
            free_class = "no_free_pixels"
        elif np.all(free_mask_values == 1):
            free_class = "all_foreground"
        elif np.all(free_mask_values == 0):
            free_class = "all_background"
        else:
            free_class = "mixed"

        stage_times = result.get("stage_times_s", {})
        row.update({
            "status": "ok",
            "n_superpixels": result["n_sp"],
            "n_free_nodes": result["n_free"],
            "total_latency_s": f"{measured_latency:.6f}",
            "edge_map_s": f"{stage_times.get('edge_map_s', 0.0):.6f}",
            "slic_s": f"{stage_times.get('slic_s', 0.0):.6f}",
            "hybrid_features_s": f"{stage_times.get('hybrid_features_s', 0.0):.6f}",
            "gmm_s": f"{stage_times.get('gmm_s', 0.0):.6f}",
            "qubo_build_s": f"{stage_times.get('qubo_build_s', 0.0):.6f}",
            "solver_s": f"{stage_times.get('solver_s', 0.0):.6f}",
            "max_abs_h_eff": f"{max_abs_h:.12g}",
            "median_abs_h_eff": f"{median_abs_h:.12g}",
            "max_J": f"{max_j:.12g}",
            "max_abs_h_over_max_J": (f"{max_abs_h / max_j:.12g}" if max_j else "inf"),
            "free_labels_changed_with_J_zero": int(np.count_nonzero(free_labels != zeroed_assignment)),
            "foreground_fraction_all_pixels": f"{float(np.mean(result['mask'])):.8f}",
            "foreground_fraction_free_pixels": (f"{free_fg_fraction:.8f}"
                                                if np.isfinite(free_fg_fraction) else ""),
            "free_region_class": free_class,
            "collapse_flag": free_class in {"all_foreground", "all_background"},
            "sklearn_convergence_warnings": sum(
                warning.category.__name__ == "ConvergenceWarning"
                and warning.category.__module__.startswith("sklearn")
                for warning in captured
            ),
        })

        mask_path = out_dir / "masks" / f"{path.stem}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), result["mask"].astype(np.uint8) * 255):
            raise OSError(f"Failed to write mask: {mask_path}")
        if overlay_path is not None:
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            _write_overlay(overlay_path, image, trimap, result["qhed_map"], result["mask"])
    except Exception as exc:  # Continue through all files; error is recorded per image.
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc).replace("\r", " ").replace("\n", " ")
    return row


def _load_existing(csv_path: Path) -> tuple[set[str], list[dict]]:
    if not csv_path.exists():
        return set(), []
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(f"Existing CSV schema does not match: {csv_path}")
        rows = list(reader)
    return {row["image_id"] for row in rows}, rows


def _write_summary(path: Path, rows: list[dict], image_count: int,
                   warmup_id: str | None, threadpool_info: list[dict] | None) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] != "ok"]
    latencies = [float(row["total_latency_s"]) for row in ok_rows]
    ratios = np.asarray([float(row["max_abs_h_over_max_J"]) for row in ok_rows], dtype=float)
    collapsed = [row for row in ok_rows if row["collapse_flag"] in (True, "True", "true", "1")]
    failures_text = ("\n".join(
        f"- `{row['filename']}`: {row['error_type']}: {row['error_message']}"
        for row in failed
    ) if failed else "- None")

    if latencies:
        median_latency = statistics.median(latencies)
        p90_latency = float(np.percentile(latencies, 90))
        latency_summary = f"- Median total latency: {median_latency:.4f} s\n- 90th percentile total latency: {p90_latency:.4f} s"
    else:
        latency_summary = "- No successful images to time."

    if ratios.size:
        ratio_summary = (
            f"min/median/p90/max: {np.min(ratios):.6g} / {np.median(ratios):.6g} / "
            f"{np.percentile(ratios, 90):.6g} / {np.max(ratios):.6g}"
        )
    else:
        ratio_summary = "No successful image ratios."

    thread_lines = [f"- OpenCV threads: {cv2.getNumThreads()}"]
    if threadpool_info is None:
        thread_lines.append("- Native BLAS/OpenMP thread limit: threadpoolctl unavailable")
    else:
        thread_lines.append("- Native BLAS/OpenMP thread limit: 1 via threadpoolctl")
        seen_threadpools = set()
        for item in threadpool_info:
            identity = (item.get("internal_api"), item.get("prefix"), item.get("num_threads"))
            if identity in seen_threadpools:
                continue
            seen_threadpools.add(identity)
            thread_lines.append(
                f"- Threadpool: {item.get('internal_api', 'unknown')} / "
                f"{item.get('prefix', 'unknown')}: {item.get('num_threads', 'unknown')} thread(s)"
            )

    content = f"""# iSeg Phase 1: Graph-Cut Robustness and Speed

**Evaluation label:** {EVALUATION_NOTE}. No ground truth was available; no IoU or Dice is reported.

## Settings

- Input images: {image_count} local files from `data/iseg/images/`
- Input size: 128x128 using OpenCV resize, matching `segment_misc.py`
- Superpixels: `n_segments=12`; solver: `graphcut`; `use_qhed=True`; `p=1`
- Trimap: auto-generated center ellipse and border, values 255=foreground seed, 0=background seed, 128=unknown
- Warm-up: one unreported call on `{warmup_id or 'none'}`
{"\n".join(thread_lines)}
- Command: `python scripts/run_iseg.py`
- Resumability: completed image IDs in `per_image.csv` are skipped on subsequent runs.

## Run Summary

- Images found: {image_count}
- Successful: {len(ok_rows)}
- Failed: {len(failed)}
- Collapsed free regions: {len(collapsed)} / {len(ok_rows)} successful images
{latency_summary}
- `max|h_eff| / max J` distribution: {ratio_summary}

## Failures

{failures_text}

## Outputs and Limitations

- Per-image metrics and status: `results/iseg/per_image.csv`
- Predicted masks (not committed): `results/iseg/masks/`
- Sample overlays for the first 12 sorted images (not committed): `results/iseg/samples/`
- This run measures execution, timing, term scales, and mask collapse only. With no iSeg ground-truth masks or user strokes present, it cannot establish segmentation accuracy.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retry-errors", action="store_true",
                        help="retry IDs that already have an error row in the CSV")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    cv2.setNumThreads(args.threads)
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
        thread_limit = threadpool_limits(limits=args.threads)
        active_threadpools = threadpool_info()
    except ImportError:
        thread_limit = None
        active_threadpools = None

    images = sorted(
        (path for path in args.input_dir.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )
    if not images:
        raise SystemExit(f"No supported images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_image.csv"
    completed, existing_rows = _load_existing(csv_path)
    if args.retry_errors:
        completed = {row["image_id"] for row in existing_rows if row["status"] == "ok"}

    pending = [path for path in images if path.stem not in completed]
    warmup_id = None
    if pending:
        warmup_id = pending[0].stem
        warmup_image, _, _ = load_bgr(pending[0])
        qaoa_seg.segment_image(
            warmup_image, auto_trimap(128, 128), n_segments=12, p=1,
            use_qhed=True, solver="graphcut",
        )

    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
            stream.flush()
        for index, image_path in enumerate(pending, start=1):
            overlay_path = (args.output_dir / "samples" / f"{image_path.stem}.png"
                            if image_path in images[:12] else None)
            row = _run_one(image_path, args.output_dir, overlay_path, EVALUATION_NOTE)
            writer.writerow(row)
            stream.flush()
            print(f"[{index}/{len(pending)}] {image_path.name}: {row['status']}", flush=True)

    _, all_rows = _load_existing(csv_path)
    _write_summary(args.output_dir / "SUMMARY.md", all_rows, len(images),
                   warmup_id, active_threadpools)
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_dir / 'SUMMARY.md'}")
    if thread_limit is not None:
        thread_limit.unregister()


if __name__ == "__main__":
    main()