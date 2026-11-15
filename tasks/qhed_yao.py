"""Exact Qiskit QHED using an auxiliary-qubit Hadamard difference scheme.

The position register encodes the flattened image index in little-endian order.
For each scan, the auxiliary |1> branch after H-controlled modular decrement
contains the forward difference at each position, up to the factor 1/2 from the
two Hadamards. The final row/column is set to zero to discard cyclic wraparound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import time
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate, StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.transpiler.passes.synthesis import HLSConfig


@dataclass(frozen=True)
class QHEDEdgeResult:
    edge_map: np.ndarray
    magnitude: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    position_qubits: int
    total_qubits: int
    amplitude_norm: float
    timings: dict[str, float]


def _append_controlled_decrement(
    circuit: QuantumCircuit,
    control_qubit: int,
    target_qubits_lsb_first: list[int],
) -> None:
    """Apply x -> x-1 mod 2^n iff control is one, with no ancilla wires."""
    if not target_qubits_lsb_first:
        raise ValueError("a decrement register must contain at least one qubit")
    for bit_index in range(len(target_qubits_lsb_first) - 1, 0, -1):
        controls = [control_qubit, *target_qubits_lsb_first[:bit_index]]
        # Qiskit ctrl_state's first listed control is its least-significant bit:
        # control=1 and all lower position bits=0 is integer value 1.
        circuit.append(
            MCXGate(num_ctrl_qubits=len(controls), ctrl_state=1),
            controls + [target_qubits_lsb_first[bit_index]],
        )
    circuit.cx(control_qubit, target_qubits_lsb_first[0])


def _auxiliary_difference(
    signal: np.ndarray,
    shift_qubits: int,
    max_parallel_threads: int = 6,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Return signed cyclic forward differences via an exact statevector circuit."""
    from qiskit_aer import AerSimulator

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size != 1 << shift_qubits:
        raise ValueError("signal length must equal 2**shift_qubits")
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        return np.zeros_like(values), norm, {
            "state_preparation_s": 0.0, "transpile_s": 0.0,
            "simulation_s": 0.0, "post_processing_s": 0.0,
        }

    preparation_start = time.perf_counter()
    positions = list(range(shift_qubits))
    auxiliary = shift_qubits
    circuit = QuantumCircuit(shift_qubits + 1)
    circuit.append(StatePreparation(values / norm), positions)
    circuit.h(auxiliary)
    _append_controlled_decrement(circuit, auxiliary, positions)
    circuit.h(auxiliary)
    preparation_seconds = time.perf_counter() - preparation_start

    simulator = AerSimulator(
        method="statevector",
        max_parallel_threads=max(1, int(max_parallel_threads)),
        max_parallel_experiments=1,
    )
    transpile_start = time.perf_counter()
    compiled = transpile(
        circuit,
        backend=simulator,
        optimization_level=0,
        seed_transpiler=20261115,
        num_processes=1,
        hls_config=HLSConfig(mcx=[("noaux_v24", {})]),
    )
    transpile_seconds = time.perf_counter() - transpile_start
    compiled.save_statevector(label="state")
    simulation_start = time.perf_counter()
    state = np.asarray(simulator.run(compiled, shots=1).result().data(0)["state"])
    simulation_seconds = time.perf_counter() - simulation_start
    post_start = time.perf_counter()
    basis_indices = np.arange(values.size, dtype=np.int64)
    branch_one = state[basis_indices | (1 << auxiliary)]
    differences = np.real_if_close(branch_one * (2.0 * norm), tol=1000)
    post_seconds = time.perf_counter() - post_start
    return np.asarray(np.real(differences), dtype=np.float64), norm, {
        "state_preparation_s": preparation_seconds,
        "transpile_s": transpile_seconds,
        "simulation_s": simulation_seconds,
        "post_processing_s": post_seconds,
    }


def qhed_yao(image: np.ndarray, max_parallel_threads: int = 6) -> QHEDEdgeResult:
    """Compute QHED horizontal/vertical forward differences for power-of-two images.

    The image is amplitude-normalized separately for each scan and its norm is
    restored after reading the auxiliary |1> branch. Horizontal and vertical scans
    use the image and its transpose, respectively. Magnitude is L2-combined and
    normalized by its maximum for a soft edge map in [0,1].
    """
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise ValueError("image must be a non-empty 2D array")
    height, width = values.shape
    if height & (height - 1) or width & (width - 1):
        raise ValueError("QHED requires power-of-two image dimensions")

    flat_size = height * width
    position_qubits = flat_size.bit_length() - 1
    horizontal, _, horizontal_times = _auxiliary_difference(
        values.reshape(-1), position_qubits, max_parallel_threads
    )
    gx = horizontal.reshape(height, width)
    gx[:, -1] = 0.0

    vertical, _, vertical_times = _auxiliary_difference(
        values.T.reshape(-1), position_qubits, max_parallel_threads
    )
    gy = vertical.reshape(width, height).T
    gy[-1, :] = 0.0

    post_start = time.perf_counter()
    magnitude = np.hypot(gx, gy)
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    edge_map = magnitude / peak if peak > 0 else np.zeros_like(magnitude)
    timings = {
        key: horizontal_times[key] + vertical_times[key]
        for key in horizontal_times
    }
    timings["post_processing_s"] += time.perf_counter() - post_start
    return QHEDEdgeResult(
        edge_map=edge_map,
        magnitude=magnitude,
        gx=gx,
        gy=gy,
        position_qubits=position_qubits,
        total_qubits=position_qubits + 1,
        amplitude_norm=float(np.linalg.norm(values)),
        timings=timings,
    )
