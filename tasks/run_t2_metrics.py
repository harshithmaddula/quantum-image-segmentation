"""Resumable Step B: BSDS500 QHED/Sobel/Canny metrics and timing."""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
import json
import os
import sys
import time
from pathlib import Path

THREADS = 6
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = str(THREADS)

import cv2
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.bsds_metrics import (
    boundary_counts,
    h95,
    localization_multi_annotator,
    non_maximum_suppression,
    pratt_fom,
    threshold_values,
)
from tasks.run_t2_qhed import BSDS, OUT, THREADS, load_gray_center_crop
METHODS = ("QHED", "Sobel", "Canny")
TOL_FRACTION = 0.0075
N_THRESHOLDS = 99
SOBEL_GRID = (3, 5, 7)
CANNY_LOW_GRID = (25, 50, 75)
CANNY_HIGH_GRID = (100, 150, 200, 225)
SEED = 20261115


def load_image_and_annotations(image_path: Path):
    gray, (width, height) = load_gray_center_crop(image_path)
    mat_path = BSDS / "groundTruth" / "val" / f"{image_path.stem}.mat"
    mat = loadmat(mat_path)
    cells = mat["groundTruth"]
    annotations = []
    for index in range(cells.shape[1]):
        cell = cells[0, index][0, 0]
        boundary = np.asarray(cell["Boundaries"], dtype=bool)
        top, left = (boundary.shape[0] - 256) // 2, (boundary.shape[1] - 256) // 2
        cropped = boundary[top:top + 256, left:left + 256]
        if cropped.shape != (256, 256):
            raise ValueError(f"GT crop has unexpected shape {cropped.shape}: {mat_path.name}")
        annotations.append(cropped)
    if not annotations:
        raise ValueError(f"No usable human boundary maps in {mat_path}")
    return gray, annotations, (width, height)


def load_training_records(limit: int | None):
    image_dir = BSDS / "images" / "train"
    paths = sorted(image_dir.glob("*.jpg"), key=lambda path: path.name)
    if limit is not None:
        paths = paths[:limit]
    records = []
    for path in paths:
        gray, _ = load_gray_center_crop(path)
        mat_path = BSDS / "groundTruth" / "train" / f"{path.stem}.mat"
        mat = loadmat(mat_path)
        cells = mat["groundTruth"]
        anns = []
        for index in range(cells.shape[1]):
            boundary = np.asarray(cells[0, index][0, 0]["Boundaries"], dtype=bool)
            top, left = (boundary.shape[0] - 256) // 2, (boundary.shape[1] - 256) // 2
            anns.append(boundary[top:top + 256, left:left + 256])
        if not anns or any(a.shape != (256, 256) for a in anns):
            continue
        records.append((path.stem, gray, anns))
    if not records:
        raise ValueError("No usable BSDS500 training examples for baseline tuning")
    return records


def _sobel_edges(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel_size)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel_size)
    magnitude = cv2.magnitude(gx, gy)
    peak = float(magnitude.max())
    if peak:
        magnitude /= peak
    return non_maximum_suppression(magnitude, gx, gy)


def _canny_edges(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.Canny(gray, threshold1=int(low), threshold2=int(high)).astype(np.float64) / 255.0


def _aggregate_f(rows: list[dict], threshold_index: int) -> float:
    matched_pred = n_pred = matched_gt = n_gt = 0
    for row in rows:
        counts = row["sweep_counts"][threshold_index]
        matched_pred += counts[0]
        n_pred += counts[1]
        matched_gt += counts[2]
        n_gt += counts[3]
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    precision, recall = matched_pred / n_pred, matched_gt / n_gt
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _sweep_one(edge_map: np.ndarray, annotations: list[np.ndarray], tolerance: float,
               thresholds: np.ndarray) -> tuple[list[list[int]], list[float]]:
    counts_by_threshold = []
    f_by_threshold = []
    for threshold in thresholds:
        result = boundary_counts(edge_map >= threshold, annotations, tolerance)
        counts_by_threshold.append([
            result["matched_pred"], result["n_pred"], result["matched_gt"], result["n_gt"]
        ])
        f_by_threshold.append(result["F"])
    return counts_by_threshold, f_by_threshold


def tune_baselines(records, output_path: Path) -> dict:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    thresholds = threshold_values([], count=N_THRESHOLDS)
    tolerance = 0.0075 * np.hypot(256, 256)

    best_sobel = None
    for kernel_size in SOBEL_GRID:
        maps = [_sobel_edges(gray, kernel_size) for _, gray, _ in records]
        result = _evaluate_maps(maps, [anns for _, _, anns in records], tolerance, thresholds)
        candidate = {"kernel_size": kernel_size, "threshold": result["ods_threshold"],
                     "train_ods": result["ods"], "train_ois": result["ois"]}
        if best_sobel is None or candidate["train_ods"] > best_sobel["train_ods"]:
            best_sobel = candidate

    best_canny = None
    for low in CANNY_LOW_GRID:
        for high in CANNY_HIGH_GRID:
            if low >= high:
                continue
            maps = [_canny_edges(gray, low, high) for _, gray, _ in records]
            result = _evaluate_maps(maps, [anns for _, _, anns in records], tolerance,
                                   np.asarray([0.5]))
            candidate = {"low": low, "high": high,
                         "train_f": result["ods"], "train_ois": result["ois"]}
            if best_canny is None or candidate["train_f"] > best_canny["train_f"]:
                best_canny = candidate

    payload = {
        "tuning_split": "BSDS500 train only",
        "training_images_used": len(records),
        "tolerance_pixels": float(tolerance),
        "n_thresholds_for_sobel": N_THRESHOLDS,
        "sobel_grid_kernel_size": list(SOBEL_GRID),
        "canny_grid_low": list(CANNY_LOW_GRID),
        "canny_grid_high": list(CANNY_HIGH_GRID),
        "sobel": best_sobel,
        "canny": best_canny,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, output_path)
    return payload


def _evaluate_maps(maps, annotation_sets, tolerance, thresholds):
    per_image = []
    all_f = []
    for edge_map, annotations in zip(maps, annotation_sets):
        counts, f_values = _sweep_one(edge_map, annotations, tolerance, thresholds)
        per_image.append({"sweep_counts": counts, "f_values": f_values})
        all_f.append(f_values)
    aggregate = [_aggregate_f(per_image, index) for index in range(len(thresholds))]
    ods_index = int(np.argmax(aggregate))
    ois = float(np.mean([max(values) for values in all_f]))
    return {"ods": float(aggregate[ods_index]),
            "ods_threshold": float(thresholds[ods_index]),
            "ois": ois, "per_image": per_image}


def _method_edge_map(method: str, gray: np.ndarray, tuning: dict,
                     edge_path: Path | None, gradient_path: Path | None) -> np.ndarray:
    if method == "QHED":
        if edge_path is None or gradient_path is None or not edge_path.exists() or not gradient_path.exists():
            raise FileNotFoundError("QHED edge map/gradient cache missing; run Step A first")
        edge_map = np.load(edge_path)
        with np.load(gradient_path) as data:
            gx, gy = data["gx"], data["gy"]
        return non_maximum_suppression(edge_map, gx, gy)
    if method == "Sobel":
        return _sobel_edges(gray, int(tuning["sobel"]["kernel_size"]))
    canny = tuning["canny"]
    return _canny_edges(gray, int(canny["low"]), int(canny["high"]))


def _image_metric_rows(image_id: str, method_maps: dict, annotations: list[np.ndarray],
                       per_method_seconds: dict, tolerance: float, thresholds: np.ndarray,
                       tuning: dict):
    rows = {}
    for method, edge_map in method_maps.items():
        measure_start = time.perf_counter()
        counts, f_values = _sweep_one(edge_map, annotations, tolerance, thresholds)
        rows[method] = {
            "image_id": image_id,
            "method": method,
            "warmup_protocol": "one unreported warm-up per method on first selected validation image",
            "sweep_counts_json": json.dumps(counts, separators=(",", ":")),
            "ois_f": float(max(f_values)),
            "edge_and_sweep_seconds": 0.0,
            "localization_seconds": "",
            "total_seconds": "",
            "f_at_ods": "",
            "pratt_fom_mean_annotator": "",
            "h95_median_annotator": "",
            "external_fixed_setting_f": "",
            "status": "ok",
            "error_message": "",
        }
        if method == "Sobel":
            fixed_threshold = float(tuning["sobel"]["threshold"])
            rows[method]["external_fixed_setting_f"] = boundary_counts(
                edge_map >= fixed_threshold, annotations, tolerance
            )["F"]
        elif method == "Canny":
            rows[method]["external_fixed_setting_f"] = boundary_counts(
                edge_map >= 0.5, annotations, tolerance
            )["F"]
        rows[method]["edge_and_sweep_seconds"] = (
            float(per_method_seconds[method]) + time.perf_counter() - measure_start
        )
    return rows


def _csv_path(pilot: bool) -> Path:
    return OUT / ("pilot_per_image_metrics.csv" if pilot else "per_image_metrics.csv")


def _write_csv(path: Path, rows_by_key: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows_by_key.values():
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="time Step B on one validation image")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--tune-limit", type=int, default=None,
                        help="training images for baseline tuning; default all training images")
    args = parser.parse_args()

    try:
        import threadpoolctl
        thread_context = threadpoolctl.threadpool_limits(limits=THREADS)
    except ImportError:
        thread_context = nullcontext()
    with thread_context:
        cv2.setNumThreads(THREADS)
        val_paths = sorted((BSDS / "images" / "val").glob("*.jpg"), key=lambda p: p.name)
        selected = val_paths[:1] if args.pilot else val_paths[:args.limit]
        if len(selected) != (1 if args.pilot else args.limit):
            raise SystemExit(f"Not enough validation images for requested count: {len(selected)}")

        # One unreported warm-up per method before measuring tuning or test timing.
        warm_gray, warm_annotations, _ = load_image_and_annotations(selected[0])
        warm_id = selected[0].stem
        warm_specs = (
            ("QHED", _method_edge_map("QHED", warm_gray, {},
                                      OUT / "edge_maps" / f"{warm_id}.npy",
                                      OUT / "gradients" / f"{warm_id}.npz")),
            ("Sobel", _sobel_edges(warm_gray, 3)),
            ("Canny", _canny_edges(warm_gray, 50, 150)),
        )
        warm_thresholds = threshold_values([], count=N_THRESHOLDS)
        for _method, _edge_map in warm_specs:
            _sweep_one(_edge_map, warm_annotations,
                       0.0075 * np.hypot(256, 256), warm_thresholds)

        tune_limit = args.tune_limit if args.tune_limit is not None else (10 if args.pilot else None)
        train_records = load_training_records(tune_limit)
        tuning_name = "baseline_tuning_pilot.json" if args.pilot else "baseline_tuning.json"
        tuning_path = OUT / tuning_name
        tuning_start = time.perf_counter()
        tuning = tune_baselines(train_records, tuning_path)
        tuning_seconds = time.perf_counter() - tuning_start

        csv_path = _csv_path(args.pilot)
        fields = ["image_id", "method", "warmup_protocol", "sweep_counts_json", "ois_f",
                  "edge_and_sweep_seconds", "localization_seconds", "total_seconds",
                  "f_at_ods", "pratt_fom_mean_annotator", "h95_median_annotator",
              "external_fixed_setting_f", "ods_threshold", "status", "error_message"]
        rows_by_key = {}
        if csv_path.exists():
            with csv_path.open("r", newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames == fields:
                    for row in reader:
                        rows_by_key[(row["image_id"], row["method"])] = row

        thresholds = threshold_values([], count=N_THRESHOLDS)
        tolerance = float(0.0075 * np.hypot(256, 256))
        for image_path in selected:
            image_id = image_path.stem
            if all(
                (image_id, method) in rows_by_key
                and rows_by_key[(image_id, method)]["status"] == "ok"
                and rows_by_key[(image_id, method)]["sweep_counts_json"]
                for method in METHODS
            ):
                print(f"SKIP {image_id}: existing Step B rows")
                continue
            method_maps = {}
            seconds = {}
            gray = annotations = None
            try:
                gray, annotations, _ = load_image_and_annotations(image_path)
                for method in METHODS:
                    started = time.perf_counter()
                    edge_path = OUT / "edge_maps" / f"{image_id}.npy"
                    gradient_path = OUT / "gradients" / f"{image_id}.npz"
                    edge_map = _method_edge_map(method, gray, tuning, edge_path, gradient_path)
                    seconds[method] = time.perf_counter() - started
                    method_maps[method] = edge_map
            except Exception as exc:
                for method in METHODS:
                    key = (image_id, method)
                    if key in rows_by_key and rows_by_key[key]["status"] == "ok":
                        continue
                    rows_by_key[key] = {
                        "image_id": image_id, "method": method, "status": "error",
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                _write_csv(csv_path, rows_by_key, fields)
                continue

            per_image_rows = _image_metric_rows(image_id, method_maps, annotations,
                                                seconds, tolerance, thresholds, tuning)
            rows_by_key.update({(image_id, method): row for method, row in per_image_rows.items()})
            _write_csv(csv_path, rows_by_key, fields)

        completed_ids = [path.stem for path in selected
                         if all((path.stem, method) in rows_by_key
                                and rows_by_key[(path.stem, method)]["status"] == "ok"
                                for method in METHODS)]
        summary_rows = {}
        for method in METHODS:
            method_rows = [rows_by_key[(image_id, method)] for image_id in completed_ids]
            if not method_rows:
                continue
            sweeps = [{"sweep_counts": json.loads(row["sweep_counts_json"])} for row in method_rows]
            aggregate = [_aggregate_f(sweeps, index) for index in range(N_THRESHOLDS)]
            ods_index = int(np.argmax(aggregate))
            ods_threshold = float(thresholds[ods_index])
            localization_start = time.perf_counter()
            for row, image_id in zip(method_rows, completed_ids):
                image_path = next(p for p in selected if p.stem == image_id)
                gray, annotations, _ = load_image_and_annotations(image_path)
                edge_path = OUT / "edge_maps" / f"{image_id}.npy"
                gradient_path = OUT / "gradients" / f"{image_id}.npz"
                edge_map = _method_edge_map(method, gray, tuning, edge_path, gradient_path)
                prediction = edge_map >= ods_threshold
                localization_start = time.perf_counter()
                boundary = boundary_counts(prediction, annotations, tolerance)
                local = localization_multi_annotator(prediction, annotations)
                loc_seconds = time.perf_counter() - localization_start
                row["localization_seconds"] = f"{loc_seconds:.6f}"
                row["total_seconds"] = f"{float(row['edge_and_sweep_seconds']) + loc_seconds:.6f}"
                row["f_at_ods"] = f"{boundary['F']:.8f}"
                row["pratt_fom_mean_annotator"] = f"{local['PrattFOM']:.8f}"
                row["h95_median_annotator"] = f"{local['H95']:.8f}"
                row["ods_threshold"] = f"{ods_threshold:.8f}"
            ois = float(np.mean([float(row["ois_f"]) for row in method_rows]))
            summary_rows[method] = {
                "ods": aggregate[ods_index], "ois": ois, "threshold": ods_threshold,
                "pratt_mean": float(np.mean([float(row["pratt_fom_mean_annotator"]) for row in method_rows])),
                "h95_median": float(np.median([float(row["h95_median_annotator"]) for row in method_rows])),
                "external_f_mean": (
                    float(np.mean([float(row["external_fixed_setting_f"]) for row in method_rows
                                   if row["external_fixed_setting_f"] != "" ]))
                    if method != "QHED" else None
                ),
                "times": [float(row["total_seconds"]) for row in method_rows],
            }
        _write_csv(csv_path, rows_by_key, fields)
        summary = {
            "label": "PILOT" if args.pilot else "FULL RUN",
            "evaluation": "BSDS500 validation; center crop 256x256; QHED/Sobel/Canny; no interpolation of GT",
            "images": completed_ids,
            "tuning": tuning,
            "tuning_seconds_this_invocation": tuning_seconds,
            "tolerance_pixels": tolerance,
            "threshold_count": N_THRESHOLDS,
            "methods": {},
        }
        for method, values in summary_rows.items():
            times = np.asarray(values["times"], dtype=float)
            summary["methods"][method] = {
                "ODS": values["ods"], "OIS": values["ois"],
                "Pratt_FOM_mean": values["pratt_mean"],
                "H95_median_over_images": values["h95_median"],
                "F_at_ODS_threshold_mean": values["external_f_mean"],
                "mean_seconds_per_image": float(np.mean(times)),
                "p95_seconds_per_image": float(np.percentile(times, 95, method="linear")),
                "n": int(times.size),
            }
        result_path = OUT / ("pilot_metrics.json" if args.pilot else "metrics_summary.json")
        result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if not args.pilot:
            table_rows = []
            timing_rows = []
            for method in METHODS:
                values = summary_rows.get(method)
                if values is None:
                    table_rows.append([method, "NA", "NA", "NA", "NA"])
                    timing_rows.append([method, "NA", "NA", "0"])
                    continue
                external_f = ("-" if values["external_f_mean"] is None
                              else f"{values['external_f_mean']:.4f}")
                table_rows.append([method, f"{values['ods']:.4f}", f"{values['ois']:.4f}",
                                   f"{values['pratt_mean']:.4f}",
                                   f"{values['h95_median']:.3f}"])
                times = np.asarray(values["times"], dtype=float)
                timing_rows.append([method, f"{np.mean(times):.4f}",
                                    f"{np.percentile(times, 95, method='linear'):.4f}",
                                    str(times.size)])
            report = [
                "# T2: QHED on BSDS500 validation",
                "",
                "Full-run metrics use the first 30 validation images in sorted filename order, center-cropped to 256x256. Ground-truth boundary maps are center-cropped without interpolation.",
                f"Baseline tuning used {tuning['training_images_used']} BSDS500 training images only; see `baseline_tuning.json` for the grid and chosen parameters.",
                f"Matching tolerance: {tolerance:.6f} pixels (0.0075 x 256x256 diagonal); thresholds: {N_THRESHOLDS} evenly spaced values from 0.01 to 1.0.",
                "Precision matches predictions to any annotator; recall accumulates matches separately over all annotators. Pratt is averaged over annotators per image; H95 is median over annotators per image, then median over images. Empty conventions are documented in `tasks/ASSUMPTIONS.md`.",
                "",
                "| Method | ODS | OIS | Pratt FOM (mean) | H95 (median px) |",
                "|---|---:|---:|---:|---:|",
            ]
            report += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in table_rows]
            report += ["", "Externally tuned baseline F on validation (not ODS/OIS):"]
            for method in ("Sobel", "Canny"):
                value = summary_rows.get(method, {}).get("external_f_mean")
                report.append(f"- {method}: {value:.4f}" if value is not None else f"- {method}: unavailable")
            report += ["", "| Method | Mean seconds/image | P95 seconds/image | n |",
                       "|---|---:|---:|---:|"]
            report += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in timing_rows]
            report += ["", "See `per_image_metrics.csv` for image-level values and resumable sweep counts."]
            (OUT / "Table2.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Per-image CSV: {csv_path}")
        print(f"Summary JSON: {result_path}")
        if not args.pilot and len(completed_ids) == 30:
            _write_example_figure(selected[:4], tuning)


def _write_example_figure(image_paths, tuning) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(image_paths), 5, figsize=(15, 3 * len(image_paths)))
    if len(image_paths) == 1:
        axes = np.asarray([axes])
    for row, path in enumerate(image_paths):
        gray, annotations, _ = load_image_and_annotations(path)
        image_id = path.stem
        edge = np.load(OUT / "edge_maps" / f"{image_id}.npy")
        with np.load(OUT / "gradients" / f"{image_id}.npz") as grad:
            qhed = non_maximum_suppression(edge, grad["gx"], grad["gy"])
        sobel = _sobel_edges(gray, int(tuning["sobel"]["kernel_size"]))
        canny = _canny_edges(gray, int(tuning["canny"]["low"]), int(tuning["canny"]["high"]))
        gt = np.logical_or.reduce(annotations)
        axes[row, 0].imshow(gray, cmap="gray"); axes[row, 0].set_title(f"{image_id} image")
        axes[row, 1].imshow(qhed, cmap="magma"); axes[row, 1].set_title("QHED")
        axes[row, 2].imshow(sobel, cmap="magma"); axes[row, 2].set_title("Sobel")
        axes[row, 3].imshow(canny, cmap="gray"); axes[row, 3].set_title("Canny")
        axes[row, 4].imshow(gt, cmap="gray"); axes[row, 4].set_title("GT boundary union")
        for axis in axes[row]: axis.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "t2_examples.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
