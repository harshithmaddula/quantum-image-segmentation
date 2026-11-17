"""Measure one noisy shot and three noiseless sampling sizes for T1's 8x8 circuit."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.transpiler.passes.synthesis import HLSConfig

from t1_neqr_resources import (
    BASIS_GATES,
    MCX_SYNTHESIS,
    OPTIMIZATION_LEVEL,
    SEED_TRANSPILE,
    _crop_gray,
    build_qiskit_neqr,
)


IMAGE = ROOT / "misc" / "5.1.12.tiff"
OUTPUT = ROOT / "results" / "tasks" / "T1" / "sampling_timing.md"
THREADS = 4


def timed_run(simulator: AerSimulator, circuit, shots: int, seed: int) -> float:
    started = time.perf_counter()
    simulator.run(circuit, shots=shots, seed_simulator=seed).result()
    return time.perf_counter() - started


def main() -> None:
    image = _crop_gray(IMAGE, 8)
    circuit = build_qiskit_neqr(image)
    circuit.measure_all()
    compiled = transpile(
        circuit,
        basis_gates=BASIS_GATES,
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
        hls_config=HLSConfig(mcx=[(MCX_SYNTHESIS, {})]),
        num_processes=1,
    )

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["u"])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])
    noisy_simulator = AerSimulator(
        method="statevector", noise_model=noise, max_parallel_threads=THREADS
    )
    noisy_seconds = timed_run(noisy_simulator, compiled, 1, 20261116)

    noiseless_simulator = AerSimulator(method="statevector", max_parallel_threads=THREADS)
    noiseless_times = {}
    for shots in (1000, 10000, 100000):
        noiseless_times[shots] = timed_run(noiseless_simulator, compiled, shots, 20261116 + shots)

    rows = [
        "# Table 1b: T1 Finite-Shot Sampling Time",
        "",
        "Circuit: `5.1.12.tiff`, centered 8x8 crop, 14-qubit NEQR circuit transpiled to `{cx,u}` with the T1 settings. CPU: Intel Core i5-12450H. Timings are simulator execution only and exclude circuit construction and transpilation; no warm-up was used.",
        "",
        "| Run | Noise | Shots | Simulation wall time (s) |",
        "|---|---|---:|---:|",
        f"| Noisy | depolarizing p1=0.001, p2=0.01 | 1 | {noisy_seconds:.6f} |",
    ]
    rows.extend(
        f"| Noiseless | none | {shots:,} | {seconds:.6f} |"
        for shots, seconds in noiseless_times.items()
    )
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"noisy_shots=1 p1=0.001 p2=0.01 seconds={noisy_seconds:.6f}")
    for shots, seconds in noiseless_times.items():
        print(f"noiseless_shots={shots} seconds={seconds:.6f}")
    print(f"transpiled_depth={compiled.depth()} output={OUTPUT}")


if __name__ == "__main__":
    main()