"""Measure finite-shot T1 NEQR distribution fidelity and decoded pixel quality."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.transpiler.passes.synthesis import HLSConfig

from t1_neqr_resources import (
    BASIS_GATES,
    IMAGES,
    MCX_SYNTHESIS,
    OPTIMIZATION_LEVEL,
    SEED_TRANSPILE,
    SIZES,
    _crop_gray,
    build_qiskit_neqr,
)


THREADS = 4
NOISY_IMAGE = "5.1.12.tiff"
NOISY_SIDE = 8
NOISY_BUDGET_SECONDS = 20 * 60
NOISE_P1 = 0.001
NOISE_P2 = 0.01
SHOT_LEVELS = (1000, 10000, 100000)
OUTPUT = ROOT / "results" / "tasks" / "T1" / "Table1b.md"
TIMING_FILE = ROOT / "results" / "tasks" / "T1" / "sampling_timing.md"
GATE_CSV = ROOT / "results" / "tasks" / "T1" / "per_image.csv"


def compile_measured(image: np.ndarray):
    circuit = build_qiskit_neqr(image)
    circuit.measure_all()
    return transpile(
        circuit,
        basis_gates=BASIS_GATES,
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
        hls_config=HLSConfig(mcx=[(MCX_SYNTHESIS, {})]),
        num_processes=1,
    )


def memory_to_counts(memory, image: np.ndarray) -> np.ndarray:
    side = image.shape[0]
    n = side.bit_length() - 1
    total_bits = 2 * n + 8
    counts = np.zeros((side * side, 256), dtype=np.int64)
    for raw_bits in memory:
        bits = format(int(raw_bits.replace(" ", ""), 2), f"0{total_bits}b")
        y = int(bits[:n], 2)
        x = int(bits[n:2 * n], 2)
        intensity = int(bits[2 * n:], 2)
        counts[y * side + x, intensity] += 1
    return counts


def qiskit_counts_to_matrix(raw_counts: dict[str, int], side: int) -> np.ndarray:
    n = side.bit_length() - 1
    total_bits = 2 * n + 8
    counts = np.zeros((side * side, 256), dtype=np.int64)
    for raw_bits, number in raw_counts.items():
        bits = format(int(raw_bits.replace(" ", ""), 2), f"0{total_bits}b")
        y = int(bits[:n], 2)
        x = int(bits[n:2 * n], 2)
        intensity = int(bits[2 * n:], 2)
        counts[y * side + x, intensity] += int(number)
    return counts


def direct_exact_distribution(image: np.ndarray, shots: int, seed: int) -> np.ndarray:
    flat = image.ravel().astype(np.uint8)
    positions = np.random.default_rng(seed).integers(0, flat.size, size=shots)
    counts = np.zeros((flat.size, 256), dtype=np.int64)
    np.add.at(counts, (positions, flat[positions]), 1)
    return counts


def metrics(image: np.ndarray, counts: np.ndarray, shots: int) -> dict[str, float]:
    flat = image.ravel().astype(np.uint8)
    position_counts = counts.sum(axis=1)
    predicted = np.argmax(counts, axis=1).astype(np.uint8)
    observed = position_counts > 0
    exact_pixels = observed & (predicted == flat)
    coverage = float(np.mean(observed))
    recovered = float(np.mean(exact_pixels))
    mse = float(np.mean((predicted.astype(np.float64) - flat.astype(np.float64)) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)
    correct_counts = counts[np.arange(flat.size), flat]
    bhattacharyya = np.sum(np.sqrt(correct_counts / (shots * flat.size)))
    fidelity = float(bhattacharyya ** 2)
    return {"fidelity": fidelity, "recovered": recovered, "coverage": coverage, "psnr": psnr}


def fmt(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.6f}"


def error_free_estimates() -> dict[tuple[str, int], str]:
    import csv

    estimates = {}
    with GATE_CSV.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "ok":
                continue
            gates = int(row["total_gates"])
            cx_count = int(row["cx_count"])
            log10_probability = ((gates - cx_count) * math.log10(1.0 - NOISE_P1)
                                 + cx_count * math.log10(1.0 - NOISE_P2))
            exponent = math.floor(log10_probability)
            mantissa = 10.0 ** (log10_probability - exponent)
            estimates[(row["image"], int(row["size"]))] = f"{mantissa:.2f}e{exponent}"
    return estimates


def run_noisy_budget(simulator, circuit, side: int, probe_seconds: float):
    counts = np.zeros((side * side, 256), dtype=np.int64)
    start = time.perf_counter()
    deadline = start + NOISY_BUDGET_SECONDS
    shots_total = 0
    seed = 20261118
    shot_rate = 8.0 / max(probe_seconds, 0.01)
    batch_size = max(20, min(4096, int(shot_rate * 10.0)))
    batch_index = 0
    while True:
        remaining = deadline - time.perf_counter()
        if shots_total >= 20 and remaining <= 5.0:
            break
        if shots_total > 0:
            safe_shots = max(1, int(shot_rate * max(1.0, remaining - 5.0)))
            batch_size = min(batch_size, safe_shots)
        batch_size = max(1, batch_size)
        batch_started = time.perf_counter()
        result = simulator.run(
            circuit,
            shots=batch_size,
            seed_simulator=seed + batch_index,
        ).result()
        batch_elapsed = time.perf_counter() - batch_started
        counts += qiskit_counts_to_matrix(result.get_counts(), side)
        shots_total += batch_size
        batch_index += 1
        if batch_elapsed > remaining:
            break
        shot_rate = batch_size / max(batch_elapsed, 0.01)
        batch_size = max(1, min(4096, int(shot_rate * 10.0)))
    return counts, shots_total, time.perf_counter() - start


def main() -> None:
    noiseless_rows = []
    rng_seeds = {}
    for image_index, name in enumerate(IMAGES):
        for side in SIZES:
            image = _crop_gray(ROOT / "misc" / name, side)
            seed = 20261117 + image_index * 100 + side
            rng_seeds[(name, side)] = seed
            if side < 32:
                compiled = compile_measured(image)
                simulator = AerSimulator(
                    method="statevector",
                    max_parallel_threads=THREADS,
                    max_parallel_experiments=1,
                )
                memory = simulator.run(
                    compiled, shots=max(SHOT_LEVELS), memory=True, seed_simulator=seed
                ).result().get_memory()
                for shots in SHOT_LEVELS:
                    case_counts = memory_to_counts(memory[:shots], image)
                    values = metrics(image, case_counts, shots)
                    noiseless_rows.append((name, side, shots, values,
                                           "Aer; nested prefix of one 100k-shot run"))
            else:
                positions = np.random.default_rng(seed).integers(
                    0, image.size, size=max(SHOT_LEVELS)
                )
                values_by_shots = []
                for shots in SHOT_LEVELS:
                    counts = np.zeros((image.size, 256), dtype=np.int64)
                    flat = image.ravel().astype(np.uint8)
                    prefix = positions[:shots]
                    np.add.at(counts, (prefix, flat[prefix]), 1)
                    values_by_shots.append((shots, metrics(image, counts, shots)))
                for shots, values in values_by_shots:
                    noiseless_rows.append((name, side, shots, values,
                                           "direct samples from verified ideal measurement distribution"))

    noise_image = _crop_gray(ROOT / "misc" / NOISY_IMAGE, NOISY_SIDE)
    noisy_compiled = compile_measured(noise_image)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(NOISE_P1, 1), ["u"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(NOISE_P2, 2), ["cx"])
    noisy_simulator = AerSimulator(
        method="statevector",
        noise_model=noise_model,
        max_parallel_threads=THREADS,
        max_parallel_experiments=1,
    )
    probe_started = time.perf_counter()
    noisy_simulator.run(
        noisy_compiled, shots=8, memory=True, seed_simulator=20261118
    ).result().get_memory()
    probe_seconds = time.perf_counter() - probe_started
    print(f"noisy_probe image={NOISY_IMAGE} size=8 shots=8 seconds={probe_seconds:.6f} "
          f"seconds_per_shot={probe_seconds / 8:.6f}", flush=True)

    noisy_counts, noisy_shots, noisy_seconds = run_noisy_budget(
        noisy_simulator, noisy_compiled, NOISY_SIDE, probe_seconds
    )
    noisy_values = metrics(noise_image, noisy_counts, noisy_shots)
    estimates = error_free_estimates()
    noisy_overrun = max(0.0, noisy_seconds - NOISY_BUDGET_SECONDS)

    lines = [
        "# Table 1b: Finite-Shot NEQR Sampling Quality",
        "",
        "Each pixel is decoded as the most frequent sampled intensity at its position; unsampled positions are set to zero and count as unrecovered. Coverage is the fraction of positions sampled at least once. Fidelity is classical Bhattacharyya fidelity `(sum_x sqrt(p_x q_x))^2` between the ideal and empirical computational-basis measurement distributions; it is not quantum-state fidelity. PSNR uses 8-bit intensities and the full image, including unsampled positions.",
        "For 8x8 and 16x16, each shot count is a nested prefix of one 100,000-shot Aer sample per image/size. For 32x32, samples are drawn directly from the exact ideal measurement distribution; Phase 1 verified exact NEQR decodes for all three 32x32 circuits. The exact-distribution sampling is not an Aer circuit run.",
        "",
        "| Image | Size | Shots | Classical fidelity | Exact pixels / all | Coverage | PSNR (dB) | Sampling method |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, side, shots, values, method in noiseless_rows:
        lines.append(
            f"| {name} | {side}x{side} | {shots:,} | {values['fidelity']:.6f} | "
            f"{values['recovered']:.6f} | {values['coverage']:.6f} | {fmt(values['psnr'])} | {method} |"
        )
        print(
            f"noiseless image={name} size={side} shots={shots} "
            f"fidelity={values['fidelity']:.6f} recovered={values['recovered']:.6f} "
            f"coverage={values['coverage']:.6f} psnr_db={fmt(values['psnr'])}",
            flush=True,
        )

    lines += [
        "",
        "## Noisy 8x8 Run",
        "",
        f"Circuit: `{NOISY_IMAGE}`, 8x8 (the canonical T1 timing image). The 8-shot single-run probe took {probe_seconds:.6f} s ({probe_seconds / 8:.6f} s/shot); the capped sampling run then used {noisy_shots:,} shots in {noisy_seconds:.2f} s, including a {noisy_overrun:.2f} s overrun while the final batch completed. The target budget was 1,200 s. Depolarizing error was applied after each transpiled `u` (p1=0.001) and `cx` (p2=0.01). The gate-count estimate of error-free probability is {estimates[(NOISY_IMAGE, NOISY_SIDE)]}.",
        "",
        "| Image | Size | Shots | Classical fidelity | Exact pixels / all | Coverage | PSNR (dB) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        f"| {NOISY_IMAGE} | 8x8 | {noisy_shots:,} | {noisy_values['fidelity']:.6f} | "
        f"{noisy_values['recovered']:.6f} | {noisy_values['coverage']:.6f} | "
        f"{fmt(noisy_values['psnr'])} | computed |",
        "",
        "Noisy 16x16 and 32x32 were not computed. Error-free probability estimates below use `(1-p1)^N_u (1-p2)^N_cx` from the saved transpiled u/CX counts; no other noisy metrics are estimated.",
        "",
        "| Image | Size | Noisy status | Estimated error-free probability |",
        "|---|---:|---|---:|",
    ]
    for name in IMAGES:
        for side in (16, 32):
            lines.append(
                f"| {name} | {side}x{side} | not computed | {estimates[(name, side)]} |"
            )
    lines += [
        "",
        f"Noiseless Aer sampling timings remain in [`sampling_timing.md`](sampling_timing.md). The single 8-shot noisy probe timing is {probe_seconds:.6f} s; the capped noisy run used {noisy_shots:,} shots.",
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"noisy image={NOISY_IMAGE} size=8 shots={noisy_shots} "
          f"fidelity={noisy_values['fidelity']:.6f} recovered={noisy_values['recovered']:.6f} "
          f"coverage={noisy_values['coverage']:.6f} psnr_db={fmt(noisy_values['psnr'])}",
          flush=True)
    print(f"Table1b={OUTPUT}")


if __name__ == "__main__":
    main()
