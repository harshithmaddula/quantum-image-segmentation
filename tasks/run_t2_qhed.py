"""Resumable Step A: QHED edge maps for sorted BSDS500 validation images."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.qhed_yao import qhed_yao

BSDS = ROOT / "data" / "bsds500" / "BSDS500-master" / "BSDS500" / "data"
OUT = ROOT / "results" / "tasks" / "T2"
EDGE_DIR = OUT / "edge_maps"
GRAD_DIR = OUT / "gradients"
CSV_PATH = OUT / "per_image_qhed.csv"
THREADS = 6
SEED = 20261115
FIELDS = [
    "image_id", "filename", "width", "height", "crop_width", "crop_height",
    "status", "error_type", "error_message", "total_seconds",
    "state_preparation_seconds", "transpile_seconds", "simulation_seconds",
    "post_processing_seconds", "max_abs_difference_vs_classical",
    "stateprep_rounding_digits", "stateprep_max_abs_perturbation",
    "edge_npy", "gradient_npz",
]


def load_gray_center_crop(path: Path, side: int = 256) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as source:
        image = np.asarray(source.convert("L"), dtype=np.uint8)
    height, width = image.shape
    if height < side or width < side:
        raise ValueError(f"image too small for {side}x{side} center crop: {width}x{height}")
    top, left = (height - side) // 2, (width - side) // 2
    return image[top:top + side, left:left + side], (width, height)


def classical_forward_edge_map(image: np.ndarray) -> np.ndarray:
    """Normalized L2 magnitude of forward differences with zero bottom/right border."""
    values = np.asarray(image, dtype=np.float64)
    gx = np.zeros_like(values)
    gy = np.zeros_like(values)
    gx[:, :-1] = values[:, :-1] - values[:, 1:]
    gy[:-1, :] = values[:-1, :] - values[1:, :]
    magnitude = np.hypot(gx, gy)
    maximum = float(magnitude.max())
    return magnitude / maximum if maximum else np.zeros_like(magnitude)


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _ensure_provenance() -> None:
    path = OUT / "provenance.json"
    if path.exists():
        provenance = json.loads(path.read_text(encoding="utf-8"))
    else:
        provenance = {"task": "T2 QHED and BSDS500"}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    provenance.update({
        "git_commit": provenance.get("git_commit", commit),
        "python": provenance.get("python", __import__("platform").python_version()),
        "seed": SEED,
        "t2_step_a": {
            "split": "BSDS500 validation",
            "selection": "first 30 sorted .jpg filenames",
            "image_size": "center-crop 256x256, no interpolation",
            "position_qubits": 16,
            "auxiliary_qubits": 1,
            "thread_cap": THREADS,
            "outputs": "edge maps .npy, gradients .npz, resumable per_image_qhed.csv",
            "warmup": "one unreported QHED call on the first pending image before timing",
        },
    })
    _atomic_json(path, provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    image_dir = BSDS / "images" / "val"
    paths = sorted(image_dir.glob("*.jpg"), key=lambda path: path.name)[:args.limit]
    if len(paths) < args.limit:
        raise SystemExit(f"Found only {len(paths)} validation JPGs; requested {args.limit}")
    OUT.mkdir(parents=True, exist_ok=True)
    EDGE_DIR.mkdir(parents=True, exist_ok=True)
    GRAD_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_provenance()

    done = {}
    if CSV_PATH.exists():
        with CSV_PATH.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not set(reader.fieldnames or ()).issubset(FIELDS):
                raise ValueError(f"Unexpected Step A CSV columns in {CSV_PATH}")
            for row in reader:
                done[row["image_id"]] = {field: row.get(field, "") for field in FIELDS}
        temp_csv = CSV_PATH.with_suffix(".csv.tmp")
        with temp_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(done.values())
        os.replace(temp_csv, CSV_PATH)

    pending = []
    for path in paths:
        edge_path = EDGE_DIR / f"{path.stem}.npy"
        grad_path = GRAD_DIR / f"{path.stem}.npz"
        prior = done.get(path.stem)
        if not (prior and prior["status"] == "ok" and edge_path.exists() and grad_path.exists()):
            pending.append(path)
    if pending:
        warm_image, _ = load_gray_center_crop(pending[0])
        qhed_yao(warm_image, max_parallel_threads=THREADS)

    need_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with CSV_PATH.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if need_header:
            writer.writeheader()
            stream.flush()
        for path in paths:
            image_id = path.stem
            edge_path = EDGE_DIR / f"{image_id}.npy"
            grad_path = GRAD_DIR / f"{image_id}.npz"
            prior = done.get(image_id)
            if prior and prior["status"] == "ok" and edge_path.exists() and grad_path.exists():
                print(f"SKIP {image_id}: existing edge map")
                continue
            row = {field: "" for field in FIELDS}
            row.update({"image_id": image_id, "filename": path.name,
                        "status": "error", "edge_npy": str(edge_path.relative_to(ROOT)),
                        "gradient_npz": str(grad_path.relative_to(ROOT))})
            try:
                image, (width, height) = load_gray_center_crop(path)
                start = time.perf_counter()
                result = qhed_yao(image, max_parallel_threads=THREADS)
                total_seconds = time.perf_counter() - start
                edge_tmp = edge_path.with_suffix(".npy.tmp")
                with edge_tmp.open("wb") as handle:
                    np.save(handle, result.edge_map)
                os.replace(edge_tmp, edge_path)
                grad_tmp = grad_path.with_suffix(".npz.tmp")
                with grad_tmp.open("wb") as handle:
                    np.savez_compressed(handle, gx=result.gx, gy=result.gy)
                os.replace(grad_tmp, grad_path)
                reference = classical_forward_edge_map(image)
                row.update({
                    "width": width, "height": height,
                    "crop_width": 256, "crop_height": 256,
                    "status": "ok",
                    "total_seconds": f"{total_seconds:.6f}",
                    "state_preparation_seconds": f"{result.timings['state_preparation_s']:.6f}",
                    "transpile_seconds": f"{result.timings['transpile_s']:.6f}",
                    "simulation_seconds": f"{result.timings['simulation_s']:.6f}",
                    "post_processing_seconds": f"{result.timings['post_processing_s']:.6f}",
                    "max_abs_difference_vs_classical": f"{np.max(np.abs(result.edge_map-reference)):.12g}",
                    "stateprep_rounding_digits": "/".join(
                        str(value) if value is not None else "none"
                        for value in result.stateprep_rounding_digits
                    ),
                    "stateprep_max_abs_perturbation": f"{result.stateprep_max_abs_perturbation:.12g}",
                })
            except Exception as exc:
                row["error_type"] = type(exc).__name__
                row["error_message"] = str(exc).replace("\r", " ").replace("\n", " ")
            writer.writerow(row)
            stream.flush()
            done[image_id] = row
            print(f"{image_id}: {row['status']} {row['total_seconds']}s", flush=True)
    temp_csv = CSV_PATH.with_suffix(".csv.tmp")
    with temp_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(done.values())
    os.replace(temp_csv, CSV_PATH)
    print(f"Image IDs: {', '.join(path.stem for path in paths)}")
    print(f"CSV: {CSV_PATH}")
    print(f"Edge maps: {EDGE_DIR}")


if __name__ == "__main__":
    main()
