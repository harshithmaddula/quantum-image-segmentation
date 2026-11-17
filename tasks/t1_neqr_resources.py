"""Qiskit NEQR resource study for T1 (run from the repository root)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ("5.1.12.tiff", "4.2.03.tiff", "boat.512.tiff")
SIZES = (8, 16, 32)
QUBITS_BY_SIZE = {8: 14, 16: 16, 32: 18}
SEED_TRANSPILE = 17
OPTIMIZATION_LEVEL = 1
BASIS_GATES = ["cx", "u"]
MCX_SYNTHESIS = "noaux_v24"
TIME_CAP_SECONDS = 20 * 60
MEMORY_CAP_BYTES = 10 * 1024**3
THREADS = 4

T1_DIR = ROOT / "results" / "tasks" / "T1"
CSV_PATH = T1_DIR / "per_image.csv"
SUMMARY_PATH = T1_DIR / "SUMMARY.md"
PROVENANCE_PATH = T1_DIR / "provenance.json"

FIELDS = [
    "image", "size", "qubits", "status", "total_gates", "cx_count", "depth",
    "transpile_seconds", "aer_statevector_seconds", "peak_worker_memory_gb",
    "decoded_exact", "pennylane_state_max_abs_error", "old_pennylane_gates",
    "old_pennylane_depth", "error_type", "error_message",
]


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _write_event(path: Path, payload: dict) -> None:
    # The parent polls this file on Windows; in-place writes avoid replace-lock races.
    path.write_text(json.dumps(payload), encoding="utf-8")


def _crop_gray(image_path: Path, side: int):
    from PIL import Image
    import numpy as np

    with Image.open(image_path) as source:
        image = np.asarray(source.convert("L"), dtype=np.uint8)
    height, width = image.shape
    top = max(0, (height - side) // 2)
    left = max(0, (width - side) // 2)
    crop = image[top:top + side, left:left + side]
    if crop.shape != (side, side):
        padded = np.zeros((side, side), dtype=np.uint8)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded
    return crop


def build_qiskit_neqr(image, q: int = 8):
    """Construct the NEQR circuit using PennyLane's logical wire and bit order."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCXGate

    side = int(image.shape[0])
    n = side.bit_length() - 1
    if image.shape != (side, side) or (1 << n) != side:
        raise ValueError(f"NEQR requires a square power-of-two image, got {image.shape}")
    total_qubits = 2 * n + q
    circuit = QuantumCircuit(total_qubits)

    # Qiskit statevector bit strings are q_(N-1)...q_0; map logical wire 0
    # to the most-significant physical qubit to match PennyLane's wire order.
    position_qubits = [total_qubits - 1 - wire for wire in range(2 * n)]
    intensity_qubits = [total_qubits - 1 - (2 * n + bit) for bit in range(q)]
    for qubit in position_qubits:
        circuit.h(qubit)

    def bits_msb(value: int, count: int) -> list[int]:
        return [(value >> (count - 1 - index)) & 1 for index in range(count)]

    for y in range(side):
        for x in range(side):
            value = int(image[y, x])
            if value < 0 or value >= (1 << q):
                raise ValueError(f"pixel intensity {value} does not fit in {q} bits")
            if value == 0:
                continue
            control_values = bits_msb(y, n) + bits_msb(x, n)
            # Qiskit maps the first listed control to the low bit of ctrl_state.
            ctrl_state = sum(int(bit) << index for index, bit in enumerate(control_values))
            for color_index, bit in enumerate(bits_msb(value, q)):
                if bit:
                    circuit.append(
                        MCXGate(num_ctrl_qubits=2 * n, ctrl_state=ctrl_state),
                        position_qubits + [intensity_qubits[color_index]],
                    )
    return circuit


def decode_statevector(state, side: int, q: int = 8):
    import numpy as np

    n = side.bit_length() - 1
    total_bits = 2 * n + q
    decoded = np.zeros((side, side), dtype=np.uint8)
    for index in np.flatnonzero(np.abs(state) > 1e-8):
        bits = format(int(index), f"0{total_bits}b")
        y = int(bits[:n], 2)
        x = int(bits[n:2 * n], 2)
        value = int(bits[2 * n:], 2)
        decoded[y, x] = value
    return decoded


def _worker(args) -> int:
    import numpy as np
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit.transpiler.passes.synthesis import HLSConfig

    image_path = Path(args.image_path)
    size = int(args.size)
    event_path = Path(args.event_json)
    result_path = Path(args.result_json)
    _write_event(event_path, {"phase": "building_circuit", "time": time.time()})
    try:
        image = _crop_gray(image_path, size)
        circuit = build_qiskit_neqr(image)
        hls_config = HLSConfig(mcx=[(MCX_SYNTHESIS, {})])
        _write_event(event_path, {"phase": "transpiling", "time": time.time()})
        started = time.perf_counter()
        compiled = transpile(
            circuit,
            basis_gates=BASIS_GATES,
            optimization_level=OPTIMIZATION_LEVEL,
            seed_transpiler=SEED_TRANSPILE,
            hls_config=hls_config,
            num_processes=1,
        )
        transpile_seconds = time.perf_counter() - started
        if compiled.num_qubits != QUBITS_BY_SIZE[size]:
            raise AssertionError(
                f"transpilation changed qubits: expected {QUBITS_BY_SIZE[size]}, "
                f"got {compiled.num_qubits}"
            )

        ops = compiled.count_ops()
        gate_count = int(sum(int(count) for name, count in ops.items()
                             if name not in {"barrier", "save_statevector", "measure"}))
        cx_count = int(ops.get("cx", 0))
        depth = int(compiled.depth())

        _write_event(event_path, {
            "phase": "aer_statevector",
            "time": time.time(),
            "transpile_seconds": transpile_seconds,
        })
        state_circuit = compiled.copy()
        state_circuit.save_statevector(label="state")
        simulator = AerSimulator(
            method="statevector",
            max_parallel_threads=THREADS,
            max_parallel_experiments=1,
        )
        started = time.perf_counter()
        state = np.asarray(simulator.run(state_circuit, shots=1).result().data(0)["state"])
        aer_seconds = time.perf_counter() - started
        decoded = decode_statevector(state, size)
        payload = {
            "image": image_path.name,
            "size": size,
            "qubits": int(compiled.num_qubits),
            "status": "ok",
            "total_gates": gate_count,
            "cx_count": cx_count,
            "depth": depth,
            "transpile_seconds": transpile_seconds,
            "aer_statevector_seconds": aer_seconds,
            "decoded_exact": bool(np.array_equal(decoded, image)),
            "pennylane_state_max_abs_error": None,
            "error_type": "",
            "error_message": "",
        }
        reference_path = Path(args.reference_npy) if args.reference_npy else None
        if reference_path is not None and reference_path.exists():
            reference = np.load(reference_path)
            if reference.shape != state.shape:
                raise AssertionError(
                    f"PennyLane state shape {reference.shape} != Aer state shape {state.shape}"
                )
            payload["pennylane_state_max_abs_error"] = float(np.max(np.abs(reference - state)))
        _atomic_json(result_path, payload)
        _write_event(event_path, {"phase": "finished", "time": time.time()})
        return 0
    except Exception as exc:
        _atomic_json(result_path, {
            "image": image_path.name,
            "size": size,
            "qubits": QUBITS_BY_SIZE.get(size, ""),
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc).replace("\r", " ").replace("\n", " "),
        })
        _write_event(event_path, {"phase": "error", "time": time.time()})
        return 1


def _old_pennylane_counts() -> dict[tuple[str, int], tuple[int, int]]:
    import re

    report = ROOT / "results" / "metrics_report.md"
    if not report.exists():
        return {}
    pattern = re.compile(
        r"^\|\s*(5\.1\.12\.tiff|4\.2\.03\.tiff|boat\.512\.tiff)\s*"
        r"\|\s*(8x8|16x16|32x32)\s*\|\s*\d+\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|",
        re.MULTILINE,
    )
    counts = {}
    for image, size_text, gates, depth in pattern.findall(report.read_text(encoding="utf-8")):
        counts[(image, int(size_text.split("x")[0]))] = (int(gates), int(depth))
    return counts


def _make_pennylane_reference(image_name: str, reference_path: Path,
                               python_path: str | None) -> tuple[bool, str]:
    if not python_path or not Path(python_path).exists():
        return False, "PennyLane reference interpreter unavailable"
    code = (
        "import numpy as np, sys; "
        "from quantum_seg import data as D, neqr; "
        "img=D.misc_gray(sys.argv[1],size=8,mode='crop').astype(np.uint8); "
        "node,_,_=neqr.build_state_qnode(img,q=8); "
        "np.save(sys.argv[2],np.asarray(node()))"
    )
    try:
        proc = subprocess.run(
            [python_path, "-c", code, image_name, str(reference_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        output = (proc.stderr or proc.stdout).strip()[-1000:]
        return False, f"exit code {proc.returncode}; {output or 'no subprocess output'}"
    return True, ""


def _read_event(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _process_tree_rss(process, psutil_module) -> int:
    total = 0
    try:
        descendants = process.children(recursive=True)
    except psutil_module.Error:
        descendants = []
    for item in [process, *descendants]:
        try:
            total += item.memory_info().rss
        except psutil_module.Error:
            continue
    return total


def _terminate_process_tree(process, root_process, psutil_module) -> None:
    try:
        descendants = root_process.children(recursive=True)
    except psutil_module.Error:
        descendants = []
    for item in descendants:
        try:
            item.terminate()
        except psutil_module.Error:
            pass
    try:
        root_process.terminate()
    except psutil_module.Error:
        pass
    _, alive = psutil_module.wait_procs(descendants + [root_process], timeout=5)
    for item in alive:
        try:
            item.kill()
        except psutil_module.Error:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_case(image_path: Path, size: int, reference_path: Path | None,
              temp_dir: Path, python_exe: str, psutil_module) -> dict:
    result_path = temp_dir / "worker_result.json"
    event_path = temp_dir / "worker_event.json"
    log_path = temp_dir / "worker.log"
    for path in (result_path, event_path):
        path.unlink(missing_ok=True)
    command = [
        python_exe, str(Path(__file__).resolve()), "--worker",
        "--image-path", str(image_path), "--size", str(size),
        "--result-json", str(result_path), "--event-json", str(event_path),
        "--reference-npy", str(reference_path) if reference_path else "",
    ]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(THREADS),
        "OPENBLAS_NUM_THREADS": str(THREADS),
        "MKL_NUM_THREADS": str(THREADS),
        "NUMEXPR_NUM_THREADS": str(THREADS),
        "QISKIT_PARALLEL": "FALSE",
    })
    started = time.monotonic()
    peak_rss = 0
    cap_status = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        monitor = psutil_module.Process(process.pid)
        while process.poll() is None:
            try:
                rss = _process_tree_rss(monitor, psutil_module)
                peak_rss = max(peak_rss, rss)
                event = _read_event(event_path)
                if rss >= MEMORY_CAP_BYTES:
                    cap_status = "memory_cap"
                    _terminate_process_tree(process, monitor, psutil_module)
                    break
                if event.get("phase") == "transpiling":
                    elapsed = time.time() - float(event.get("time", time.time()))
                    if elapsed >= TIME_CAP_SECONDS:
                        cap_status = "time_cap"
                        _terminate_process_tree(process, monitor, psutil_module)
                        break
            except psutil_module.NoSuchProcess:
                break
            time.sleep(0.1)
        if cap_status:
            if process.poll() is None:
                _terminate_process_tree(process, monitor, psutil_module)
        else:
            process.wait()
    elapsed_total = time.monotonic() - started
    worker = {}
    if result_path.exists():
        try:
            worker = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            worker = {"error_type": type(exc).__name__, "error_message": str(exc)}
    row = {field: "" for field in FIELDS}
    row.update({"image": image_path.name, "size": size,
                "qubits": QUBITS_BY_SIZE[size],
                "peak_worker_memory_gb": f"{peak_rss / 1024**3:.6f}"})
    row.update({key: worker.get(key, "") for key in FIELDS if key in worker})
    if cap_status:
        event = _read_event(event_path)
        row["status"] = cap_status
        row["error_type"] = cap_status
        row["error_message"] = (
            f"Stopped in phase={event.get('phase', 'unknown')}; "
            f"worker elapsed={elapsed_total:.3f}s; peak RSS={peak_rss / 1024**3:.3f} GiB. "
            "No resource counts were estimated."
        )
    elif process.returncode != 0 and row["status"] != "error":
        row["status"] = "error"
        row["error_type"] = "WorkerExit"
        row["error_message"] = f"Worker exited with code {process.returncode}; see {log_path}"
    if row["status"] == "error":
        row["error_message"] = (row["error_message"] or log_path.read_text(encoding="utf-8")[-1000:])
    return row


def _read_csv() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_summary(rows: list[dict], old_counts: dict,
                   penny_reference_note: str) -> None:
    import numpy as np

    latest = {}
    for row in rows:
        latest[(row["image"], int(row["size"]))] = row
    rows = list(latest.values())
    successes = [row for row in rows if row["status"] == "ok"]
    capped = [row for row in rows if row["status"] in {"time_cap", "memory_cap"}]
    failed = [row for row in rows if row["status"] == "error"]
    lines = [
        "# T1: Qiskit NEQR resource table",
        "",
        "## Settings",
        "",
        "- NEQR: 8-bit grayscale; 2n position qubits + 8 intensity qubits; no ancillas.",
        "- Sizes processed size-major: 8x8, then 16x16, then 32x32; images: " + ", ".join(IMAGES) + ".",
        "- Image prep: `misc_gray(name, size, mode='crop').astype(uint8)`; centered crop, no resize interpolation.",
        "- Transpiler: Qiskit `transpile`, basis `{cx,u}`, optimization level 1, `seed_transpiler=17`, `num_processes=1`, no coupling map / all-to-all connectivity.",
        "- MCX synthesis: HLS `mcx=[('noaux_v24', {})]`, an ancilla-free decomposition; expected qubits remain 14/16/18.",
        f"- Resource caps: {TIME_CAP_SECONDS // 60} minutes and {MEMORY_CAP_BYTES / 1024**3:.0f} GiB peak worker RSS. Capped cells are uncomputed, not estimated.",
        f"- PennyLane comparison: {penny_reference_note}",
        "- No unreported warm-up was used for the transpile timings; each time is one transpile call.",
        "",
        "## Per-image results",
        "",
        "| Image | Size | Qubits | Qiskit gates | CX | Depth | Transpile (s) | Aer statevector (s) | Exact decode | PL max abs diff | Old PL gates/depth | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda r: (int(r["size"]), IMAGES.index(r["image"]))):
        old = old_counts.get((row["image"], int(row["size"])))
        old_text = f"{old[0]}/{old[1]}" if old else "not in existing report"
        gates = row["total_gates"] or "—"
        cx = row["cx_count"] or "—"
        depth = row["depth"] or "—"
        ttrans = row["transpile_seconds"] or "—"
        taer = row["aer_statevector_seconds"] or "—"
        exact = row["decoded_exact"] or "—"
        pl_error = row["pennylane_state_max_abs_error"] or "—"
        lines.append(
            f"| {row['image']} | {row['size']}x{row['size']} | {row['qubits']} | "
            f"{gates} | {cx} | {depth} | {ttrans} | {taer} | {exact} | {pl_error} | "
            f"{old_text} | {row['status']} |"
        )

    lines.extend(["", "## Timing summary", ""])
    for size in SIZES:
        sized = [row for row in successes if int(row["size"]) == size]
        if not sized:
            lines.append(f"- {size}x{size}: no successful transpiles (n=0).")
            continue
        for label, field in (("transpile", "transpile_seconds"),
                             ("Aer statevector verification", "aer_statevector_seconds")):
            values = np.asarray([float(row[field]) for row in sized], dtype=float)
            lines.append(
                f"- {size}x{size} {label}: mean {np.mean(values):.3f}s, "
                f"95th percentile {np.percentile(values, 95, method='linear'):.3f}s, n={len(values)}."
            )
    lines.extend(["", "## Caps and failures", ""])
    lines.append("- Capped: " + (
        ", ".join(f"{r['image']} {r['size']}x{r['size']} ({r['status']})" for r in capped)
        if capped else "none"
    ))
    lines.append("- Errors: " + (
        "; ".join(f"{r['image']} {r['size']}x{r['size']}: {r['error_type']}: {r['error_message']}"
                   for r in failed) if failed else "none"
    ))
    lines.extend([
        "",
        "## Comparison with previous PennyLane resource counts",
        "",
        "The existing report's PennyLane `qml.specs()` counts treat `MultiControlledX` as a native abstract operation. The Qiskit column counts the transpiled one- and two-qubit `u`/`cx` gates after explicit ancilla-free MCX synthesis, so a much larger count and depth are expected from decomposition; these are different counting levels, not a mismatch in pixel encoding. The Qiskit circuit retains the same total qubit count and its Aer statevector is checked by decoding intensities.",
        "",
        "## Limits",
        "",
        "No result is inferred for an uncompleted or capped transpile. See CSV status and error fields for each attempted item.",
        "",
    ])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def _make_provenance(old_counts: dict, reference_note: str) -> dict:
    import importlib.metadata
    import platform
    import psutil

    versions = {}
    for package in ("qiskit", "qiskit-aer", "numpy", "scipy", "opencv-contrib-python", "pandas", "matplotlib", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    return {
        "task": "T1 NEQR resources",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "python": platform.python_version(),
        "packages": versions,
        "cpu": platform.processor() or "unknown",
        "logical_processors": os.cpu_count(),
        "thread_cap": THREADS,
        "seed": SEED_TRANSPILE,
        "settings": {
            "images": list(IMAGES), "sizes": list(SIZES), "grayscale_bits": 8,
            "qubits_by_size": QUBITS_BY_SIZE, "ancillas": 0,
            "basis_gates": BASIS_GATES, "optimization_level": OPTIMIZATION_LEVEL,
            "seed_transpiler": SEED_TRANSPILE, "num_processes": 1,
            "mcx_hls": MCX_SYNTHESIS, "time_cap_seconds": TIME_CAP_SECONDS,
            "memory_cap_bytes": MEMORY_CAP_BYTES,
        },
        "pennylane_reference": reference_note,
        "old_pennylane_counts_loaded": len(old_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--image-path")
    parser.add_argument("--size", type=int, choices=SIZES)
    parser.add_argument("--result-json")
    parser.add_argument("--event-json")
    parser.add_argument("--reference-npy", default="")
    parser.add_argument("--pennylane-python", default=os.environ.get(
        "PENNYLANE_PYTHON",
        r"C:\Users\HARSHITH\AppData\Local\Programs\Python\Python314\python.exe",
    ))
    parser.add_argument("--retry-failed", action="store_true",
                        help="retry previously capped or failed image/size cases")
    parser.add_argument("--refresh-existing", action="store_true",
                        help="replace existing rows for sizes up to --max-size and rerun them")
    parser.add_argument("--max-size", type=int, choices=SIZES, default=32,
                        help="stop after this image size; default runs through 32x32")
    args = parser.parse_args()

    if args.worker:
        if not (args.image_path and args.size and args.result_json and args.event_json):
            parser.error("worker mode requires image path, size, result-json and event-json")
        return_code = _worker(args)
        raise SystemExit(return_code)

    import importlib.metadata
    import numpy as np
    import psutil

    T1_DIR.mkdir(parents=True, exist_ok=True)
    old_counts = _old_pennylane_counts()
    reference_path = Path(tempfile.gettempdir()) / "tasks_t1_neqr_5_1_12_8x8_pennylane.npy"
    reference_success = False
    reference_error = ""
    if reference_path.exists():
        try:
            reference = np.load(reference_path, mmap_mode="r")
            expected_amplitudes = 1 << (2 * 3 + 8)
            if reference.shape != (expected_amplitudes,) or not np.all(np.isfinite(reference)):
                reference_error = f"invalid reference state shape or values: {reference.shape}"
            else:
                reference_success = True
        except Exception as exc:
            reference_error = f"{type(exc).__name__}: {exc}"
    if not reference_success:
        reference_success, reference_error = _make_pennylane_reference(
            "5.1.12.tiff", reference_path, args.pennylane_python
        )
    reference_note = (
        f"generated with {args.pennylane_python}; comparison runs on 5.1.12.tiff 8x8"
        if reference_success else ("reference not generated: " + reference_error)
    )
    _atomic_json(PROVENANCE_PATH, _make_provenance(old_counts, reference_note))

    images_root = ROOT / "misc"
    image_paths = {name: images_root / name for name in IMAGES}
    missing = [name for name, path in image_paths.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Required misc images are missing: {missing}")

    rows_by_key = _load_rows_by_key(CSV_PATH)
    if args.refresh_existing:
        rows_by_key = {
            key: row for key, row in rows_by_key.items()
            if key[1] > args.max_size
        }
        tmp_csv = CSV_PATH.with_suffix(".csv.tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows_by_key.values())
        os.replace(tmp_csv, CSV_PATH)
    needs_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    T1_DIR.mkdir(parents=True, exist_ok=True)
    worker_tmp = Path(tempfile.mkdtemp(prefix="tasks-t1-worker-"))
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(THREADS), "OPENBLAS_NUM_THREADS": str(THREADS),
        "MKL_NUM_THREADS": str(THREADS), "NUMEXPR_NUM_THREADS": str(THREADS),
        "QISKIT_PARALLEL": "FALSE",
    })

    with CSV_PATH.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
            stream.flush()
        for size in (size for size in SIZES if size <= args.max_size):
            for image_name in IMAGES:
                key = (image_name, size)
                existing = rows_by_key.get(key)
                if existing and (existing["status"] == "ok" or not args.retry_failed):
                    print(f"SKIP {image_name} {size}x{size}: existing status={existing['status']}")
                    continue
                reference_for_case = reference_path if key == ("5.1.12.tiff", 8) and reference_success else None
                case_dir = worker_tmp / f"{Path(image_name).stem}_{size}"
                case_dir.mkdir(parents=True, exist_ok=True)
                row = _run_case(image_paths[image_name], size, reference_for_case,
                                case_dir, sys.executable, psutil)
                old = old_counts.get(key)
                row["old_pennylane_gates"] = old[0] if old else ""
                row["old_pennylane_depth"] = old[1] if old else ""
                writer.writerow(row)
                stream.flush()
                rows_by_key[key] = row
                print(
                    f"{image_name} {size}x{size}: {row['status']} "
                    f"transpile={row.get('transpile_seconds','')}s "
                    f"peak={row.get('peak_worker_memory_gb','')}GiB",
                    flush=True,
                )
                _write_summary(list(rows_by_key.values()), old_counts, reference_note)
                if size == 32 and row["status"] in {"time_cap", "memory_cap"}:
                    print("STOP: first 32x32 resource cap reached; later 32x32 cases were not attempted.",
                          flush=True)
                    break
            if size == 32 and any(
                int(row["size"]) == 32 and row["status"] in {"time_cap", "memory_cap"}
                for row in rows_by_key.values()
            ):
                break

    _write_summary(list(rows_by_key.values()), old_counts, reference_note)
    print(f"CSV: {CSV_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Provenance: {PROVENANCE_PATH}")
    print(f"Worker scratch: {worker_tmp}")


def _load_rows_by_key(csv_path: Path) -> dict[tuple[str, int], dict]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Existing CSV schema does not match expected T1 schema: {csv_path}")
        rows = list(reader)
    latest = {}
    for row in rows:
        latest[(row["image"], int(row["size"]))] = row
    if len(latest) != len(rows):
        tmp_csv = csv_path.with_suffix(".csv.tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(latest.values())
        os.replace(tmp_csv, csv_path)
    return latest


if __name__ == "__main__":
    main()
