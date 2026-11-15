"""Unrun T3 pilot: shot-based QAOA MaxCut on one 4x4 USC-SIPI image grid."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

from quantum_seg.data import load_gray, misc_paths, to_pow2
from quantum_seg.superpixel import sp_adjacency


K_DEFAULT = 16
K_MIN = 12
K_MAX = 20
SHOTS = 4096
RESTARTS = 5
MAXITER = 50
SEED = 17
PILOT_OUTPUT = ROOT / "results" / "tasks" / "T3" / "qaoa_maxcut_pilot.csv"
FULL_OUTPUT = ROOT / "results" / "tasks" / "T3" / "qaoa_maxcut_39.csv"
FIELDS = [
    "image", "image_size", "p", "K", "edge_count", "sigma", "shots",
    "restarts", "maxiter_per_restart", "gamma_parameters", "beta_parameters",
    "exact_maxcut", "exact_assignment", "otsu_threshold", "otsu_cut",
    "otsu_ratio_to_exact", "final_mode_bitstring", "final_mode_cut",
    "final_mode_ratio_to_exact", "final_expected_cut",
    "final_expected_ratio_to_exact", "final_shots", "final_seed",
    "optimizer_evaluations", "transpiled_depth", "transpile_seconds",
    "wall_clock_seconds", "status", "error_message",
]


def build_slic_graph(image: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, float]], float]:
    gray = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    target_region_size = max(4, int(round(np.sqrt(gray.size / K_DEFAULT))))
    region_sizes = [target_region_size]
    for offset in range(1, 9):
        region_sizes.extend((target_region_size + offset, target_region_size - offset))

    candidates = []
    seen = set()
    for region_size in region_sizes:
        if region_size < 4 or region_size in seen:
            continue
        seen.add(region_size)
        slic = cv2.ximgproc.createSuperpixelSLIC(
            bgr, algorithm=cv2.ximgproc.SLIC, region_size=region_size, ruler=10.0
        )
        slic.iterate(10)
        labels = slic.getLabels().astype(np.int32)
        n_nodes = int(slic.getNumberOfSuperpixels())
        if K_MIN <= n_nodes <= K_MAX:
            candidates.append((abs(n_nodes - K_DEFAULT), abs(region_size - target_region_size),
                               labels, n_nodes))
            if n_nodes == K_DEFAULT:
                break
    if not candidates:
        raise ValueError(f"SLIC did not produce K in [{K_MIN}, {K_MAX}]")
    _, _, labels, n_nodes = min(candidates, key=lambda item: (item[0], item[1]))

    flat_labels = labels.ravel()
    counts = np.bincount(flat_labels, minlength=n_nodes).astype(np.float64)
    means = np.bincount(flat_labels, weights=gray.ravel(), minlength=n_nodes) / counts / 255.0
    adjacent = sp_adjacency(labels, n_nodes)
    sigma = float(np.std(means))
    if sigma <= np.finfo(np.float64).eps:
        sigma = 1.0
    edges = [
        (left, right, float(1.0 - np.exp(-((means[left] - means[right]) ** 2)
                                           / (2.0 * sigma ** 2))))
        for left, right in adjacent
    ]
    return means, edges, sigma


def exact_maxcut(n_nodes: int, edges: list[tuple[int, int, float]]) -> tuple[np.ndarray, float]:
    shifts = np.arange(n_nodes, dtype=np.uint32)
    best_assignment = np.zeros(n_nodes, dtype=np.int8)
    best_value = -np.inf
    for start in range(0, 1 << n_nodes, 65536):
        indices = np.arange(start, min(start + 65536, 1 << n_nodes), dtype=np.uint32)
        assignments = ((indices[:, None] >> shifts) & 1).astype(np.uint8)
        values = np.zeros(indices.size, dtype=np.float64)
        for left, right, weight in edges:
            values += weight * (assignments[:, left] != assignments[:, right])
        local_index = int(np.argmax(values))
        if float(values[local_index]) > best_value:
            best_value = float(values[local_index])
            best_assignment = assignments[local_index].astype(np.int8)
    return best_assignment, best_value


def otsu_cut(image: np.ndarray, means: np.ndarray,
             edges: list[tuple[int, int, float]]) -> tuple[float, float]:
    threshold, _ = cv2.threshold(
        np.clip(np.rint(image), 0, 255).astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    labels = (means * 255.0 > threshold).astype(np.uint8)
    value = sum(weight for left, right, weight in edges if labels[left] != labels[right])
    return float(threshold), float(value)


def parameterized_circuit(n_nodes: int, p: int,
                          edges: list[tuple[int, int, float]]) -> QuantumCircuit:
    gammas = ParameterVector("gamma", p)
    betas = ParameterVector("beta", p)
    circuit = QuantumCircuit(n_nodes)
    circuit.h(range(n_nodes))
    for layer in range(p):
        for left, right, weight in edges:
            circuit.rzz(-gammas[layer] * weight, left, right)
        circuit.rx(2 * betas[layer], range(n_nodes))
    circuit.measure_all()
    return circuit


def cut_value(bitstring: str, edges: list[tuple[int, int, float]]) -> float:
    labels = [int(bit) for bit in bitstring.replace(" ", "")[::-1]]
    return float(sum(weight for left, right, weight in edges
                     if labels[left] != labels[right]))


def summarize_counts(counts: dict[str, int],
                     edges: list[tuple[int, int, float]]) -> tuple[str, float, float]:
    total = sum(counts.values())
    normalized_counts = {bits.replace(" ", ""): count for bits, count in counts.items()}
    max_count = max(normalized_counts.values())
    mode_bits = min(bits for bits, count in normalized_counts.items() if count == max_count)
    mode_cut = cut_value(mode_bits, edges)
    expected_cut = sum(count * cut_value(bits, edges)
                       for bits, count in normalized_counts.items()) / total
    return mode_bits, mode_cut, float(expected_cut)


def run_depth(n_nodes: int, p: int, edges: list[tuple[int, int, float]],
              exact_value: float, otsu_threshold: float, otsu_value: float,
              backend: AerSimulator, args: argparse.Namespace) -> dict:
    depth_started = time.perf_counter()
    circuit = parameterized_circuit(n_nodes, p, edges)
    compile_started = time.perf_counter()
    compiled = transpile(circuit, backend, optimization_level=1,
                         seed_transpiler=SEED + p)
    transpile_seconds = time.perf_counter() - compile_started
    gammas = list(circuit.parameters)
    gamma_params = sorted((item for item in gammas if item.name.startswith("gamma")),
                          key=lambda item: int(item.name.split("[")[1][:-1]))
    beta_params = sorted((item for item in gammas if item.name.startswith("beta")),
                         key=lambda item: int(item.name.split("[")[1][:-1]))
    best_fun = float("inf")
    best_params = None
    evaluations = 0
    rng = np.random.default_rng(SEED + p)
    final_seed = SEED + 900000 + p

    for restart in range(RESTARTS):
        initial = np.concatenate((rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)))
        def objective(parameters: np.ndarray) -> float:
            nonlocal evaluations
            evaluations += 1
            binding = dict(zip(gamma_params, parameters[:p]))
            binding.update(zip(beta_params, parameters[p:]))
            bound = compiled.assign_parameters(binding, inplace=False)
            job = backend.run(bound, shots=args.shots,
                              seed_simulator=SEED + p * 100000 + restart * 1000 + evaluations)
            return -summarize_counts(job.result().get_counts(), edges)[2]

        result = minimize(objective, initial, method="COBYLA",
                          options={"maxiter": args.maxiter, "rhobeg": 0.5})
        if float(result.fun) < best_fun:
            best_fun = float(result.fun)
            best_params = np.asarray(result.x, dtype=np.float64)

    final_binding = dict(zip(gamma_params, best_params[:p]))
    final_binding.update(zip(beta_params, best_params[p:]))
    final_circuit = compiled.assign_parameters(final_binding, inplace=False)
    final_job = backend.run(final_circuit, shots=args.shots, seed_simulator=final_seed)
    mode_bits, mode_cut, expected_cut = summarize_counts(
        final_job.result().get_counts(), edges
    )
    return {
        "p": p,
        "restarts": RESTARTS,
        "shots": args.shots,
        "maxiter_per_restart": args.maxiter,
        "gamma_parameters": ";".join(f"{value:.8g}" for value in best_params[:p]),
        "beta_parameters": ";".join(f"{value:.8g}" for value in best_params[p:]),
        "exact_maxcut": exact_value,
        "otsu_threshold": otsu_threshold,
        "otsu_cut": otsu_value,
        "otsu_ratio_to_exact": otsu_value / exact_value if exact_value else "",
        "final_mode_bitstring": mode_bits,
        "final_mode_cut": mode_cut,
        "final_mode_ratio_to_exact": mode_cut / exact_value if exact_value else "",
        "final_expected_cut": expected_cut,
        "final_expected_ratio_to_exact": expected_cut / exact_value if exact_value else "",
        "final_shots": args.shots,
        "final_seed": final_seed,
        "optimizer_evaluations": evaluations,
        "transpiled_depth": compiled.depth(),
        "transpile_seconds": transpile_seconds,
        "wall_clock_seconds": time.perf_counter() - depth_started,
    }


def write_rows(path: Path, rows_by_key: dict[tuple[str, int], dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS}
                         for row in rows_by_key.values())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="run one image instead of the sorted USC-SIPI set")
    parser.add_argument("--limit", type=int, default=39)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    args = parser.parse_args()
    if not 1 <= args.limit <= 39:
        raise SystemExit("--limit must be between 1 and 39")
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        image_paths = [image_path]
        default_output = PILOT_OUTPUT
    else:
        image_paths = [Path(path) for path in misc_paths("*.tiff")[:args.limit]]
        default_output = FULL_OUTPUT
    if len(image_paths) != (1 if args.image else args.limit):
        raise SystemExit(f"Found {len(image_paths)} images; expected {1 if args.image else args.limit}")
    for image_path in image_paths:
        if not image_path.is_file():
            raise SystemExit(f"USC-SIPI image not found: {image_path}")
    output_path = args.output or default_output

    rows_by_key = {}
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames == FIELDS:
                for row in reader:
                    key = (row["image"], int(row["p"]))
                    rows_by_key[key] = row
    backend = AerSimulator()
    for image_path in image_paths:
        try:
            image = to_pow2(load_gray(str(image_path)), 128, mode="crop")
            means, edges, sigma = build_slic_graph(image)
            n_nodes = int(means.size)
            exact_labels, exact_value = exact_maxcut(n_nodes, edges)
            otsu_threshold, otsu_value = otsu_cut(image, means, edges)
            for p in range(1, 5):
                key = (image_path.name, p)
                if key in rows_by_key and rows_by_key[key].get("status") == "ok":
                    print(f"SKIP {image_path.name} p={p}: completed row")
                    continue
                row = run_depth(n_nodes, p, edges, exact_value, otsu_threshold,
                                otsu_value, backend, args)
                row.update({
                    "image": image_path.name, "image_size": "128x128", "K": n_nodes,
                    "edge_count": len(edges), "sigma": f"{sigma:.12g}", "status": "ok",
                    "error_message": "",
                })
                rows_by_key[key] = row
                write_rows(output_path, rows_by_key)
                print(
                    f"{image_path.name} p={p}: mode={row['final_mode_cut']:.6f} "
                    f"({row['final_mode_ratio_to_exact']:.4f}), "
                    f"expected={row['final_expected_cut']:.6f} "
                    f"({row['final_expected_ratio_to_exact']:.4f}), "
                    f"exact={exact_value:.6f}, otsu={otsu_value:.6f}, "
                    f"depth={row['transpiled_depth']}, wall={row['wall_clock_seconds']:.2f}s",
                    flush=True,
                )
        except Exception as exc:
            for p in range(1, 5):
                key = (image_path.name, p)
                if key not in rows_by_key or rows_by_key[key].get("status") != "ok":
                    rows_by_key[key] = {
                        "image": image_path.name, "image_size": "128x128", "p": p,
                        "status": "error", "error_message": f"{type(exc).__name__}: {exc}",
                    }
            write_rows(output_path, rows_by_key)
            print(f"{image_path.name}: ERROR {type(exc).__name__}: {exc}", flush=True)
    print(f"CSV: {output_path}")


if __name__ == "__main__":
    main()