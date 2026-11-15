"""User-run, capped pilot for exact 256x256 BSDS500 validation QHED."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BSDS_DATA = ROOT / "data" / "bsds500" / "BSDS500-master" / "BSDS500" / "data"
OUTPUT_DIR = ROOT / "results" / "tasks" / "T2"
THREADS = 6
TIME_CAP_SECONDS = 30 * 60
MEMORY_CAP_BYTES = 10 * 1024**3
SEED = 20261115


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _process_tree_rss(root_process, psutil_module) -> int:
    try:
        children = root_process.children(recursive=True)
    except psutil_module.Error:
        children = []
    rss = 0
    for process in [root_process, *children]:
        try:
            rss += process.memory_info().rss
        except psutil_module.Error:
            continue
    return rss


def _terminate_tree(popen, root_process, psutil_module) -> None:
    try:
        children = root_process.children(recursive=True)
    except psutil_module.Error:
        children = []
    for process in children:
        try:
            process.terminate()
        except psutil_module.Error:
            pass
    try:
        root_process.terminate()
    except psutil_module.Error:
        pass
    _, alive = psutil_module.wait_procs(children + [root_process], timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil_module.Error:
            pass
    if popen.poll() is None:
        try:
            popen.wait(timeout=5)
        except subprocess.TimeoutExpired:
            popen.kill()
            popen.wait()


def _worker(image_path: Path, result_path: Path) -> int:
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)

    import numpy as np
    from PIL import Image
    from tasks.qhed_yao import qhed_yao

    try:
        with Image.open(image_path) as source:
            gray = np.asarray(source.convert("L"), dtype=np.uint8)
        height, width = gray.shape
        if height < 256 or width < 256:
            raise ValueError(f"image smaller than 256x256: {(height, width)}")
        top = (height - 256) // 2
        left = (width - 256) // 2
        crop = gray[top:top + 256, left:left + 256]
        start = time.perf_counter()
        result = qhed_yao(crop, max_parallel_threads=THREADS)
        elapsed = time.perf_counter() - start
        payload = {
            "status": "ok",
            "image": image_path.name,
            "original_shape": [int(height), int(width)],
            "crop_shape": [int(x) for x in crop.shape],
            "position_qubits": result.position_qubits,
            "total_qubits": result.total_qubits,
            "elapsed_seconds": elapsed,
            "edge_shape": [int(x) for x in result.edge_map.shape],
            "edge_min": float(np.min(result.edge_map)),
            "edge_max": float(np.max(result.edge_map)),
            "edge_range": float(np.ptp(result.edge_map)),
            "thread_cap": THREADS,
            "time_cap_seconds": TIME_CAP_SECONDS,
            "memory_cap_gib": MEMORY_CAP_BYTES / 1024**3,
        }
        _atomic_json(result_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        payload = {
            "status": "error",
            "image": image_path.name,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        _atomic_json(result_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return 1


def _write_provenance(output_dir: Path, image_path: Path) -> None:
    packages = {}
    for package in ("qiskit", "qiskit-aer", "numpy", "scipy", "opencv-contrib-python",
                    "pandas", "matplotlib", "psutil"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        cpu = platform.processor()
    except Exception:
        cpu = "unknown"
    _atomic_json(output_dir / "provenance.json", {
        "task": "T2 QHED 256x256 pilot",
        "git_commit": commit,
        "python": platform.python_version(),
        "cpu": cpu,
        "logical_processors": os.cpu_count(),
        "seed": SEED,
        "packages": packages,
        "settings": {
            "pilot_image": image_path.name,
            "split": "BSDS500 validation",
            "crop": "center 256x256, no interpolation",
            "threads": THREADS,
            "time_cap_seconds": TIME_CAP_SECONDS,
            "memory_cap_bytes": MEMORY_CAP_BYTES,
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()

    if args.worker:
        if args.image is None or args.result is None:
            parser.error("--worker requires --image and --result")
        raise SystemExit(_worker(args.image, args.result))

    import psutil

    image_dir = BSDS_DATA / "images" / "val"
    image_paths = sorted(image_dir.glob("*.jpg"), key=lambda path: path.name)
    if not image_paths:
        raise SystemExit(f"No validation JPG images found in {image_dir}")
    image_path = image_paths[0]
    annotation_path = BSDS_DATA / "groundTruth" / "val" / f"{image_path.stem}.mat"
    if not annotation_path.exists():
        raise SystemExit(f"Pilot image has no corresponding GT .mat: {annotation_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_provenance(OUTPUT_DIR, image_path)
    result_path = OUTPUT_DIR / "pilot.json"
    with tempfile.TemporaryDirectory(prefix="t2-qhed-pilot-") as temp_dir:
        log_path = Path(temp_dir) / "worker.log"
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": str(THREADS),
            "OPENBLAS_NUM_THREADS": str(THREADS),
            "MKL_NUM_THREADS": str(THREADS),
            "NUMEXPR_NUM_THREADS": str(THREADS),
        })
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--image", str(image_path), "--result", str(result_path)]
        start = time.monotonic()
        peak_rss = 0
        cap = ""
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
            root_process = psutil.Process(process.pid)
            while process.poll() is None:
                peak_rss = max(peak_rss, _process_tree_rss(root_process, psutil))
                elapsed = time.monotonic() - start
                if elapsed >= TIME_CAP_SECONDS:
                    cap = "time_cap"
                    break
                if peak_rss >= MEMORY_CAP_BYTES:
                    cap = "memory_cap"
                    break
                time.sleep(0.5)
            if cap:
                _terminate_tree(process, root_process, psutil)
            else:
                process.wait()
        if cap:
            payload = {
                "status": cap,
                "image": image_path.name,
                "elapsed_seconds": time.monotonic() - start,
                "peak_memory_gib": peak_rss / 1024**3,
                "note": "QHED pilot stopped at cap; edge-map shape/range not computed.",
            }
            _atomic_json(result_path, payload)
        elif not result_path.exists():
            worker_log = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            payload = {
                "status": "worker_error",
                "image": image_path.name,
                "worker_exit_code": process.returncode,
                "elapsed_seconds": time.monotonic() - start,
                "peak_memory_gib": peak_rss / 1024**3,
                "worker_output_tail": worker_log,
            }
            _atomic_json(result_path, payload)
        else:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["parent_elapsed_seconds"] = time.monotonic() - start
            payload["peak_memory_gib"] = peak_rss / 1024**3
            _atomic_json(result_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload["status"] not in {"ok", "time_cap", "memory_cap"}:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
