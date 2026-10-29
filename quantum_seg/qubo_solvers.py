"""Exact and approximate solvers for the reduced superpixel cut objective."""

from __future__ import annotations

import numpy as np


EXACT_MAX_NODES = 20


class _Dinic:
    def __init__(self, n_nodes: int):
        self.graph: list[list[list[float | int]]] = [[] for _ in range(n_nodes)]

    def add_edge(self, source: int, target: int, capacity: float) -> None:
        forward: list[float | int] = [target, len(self.graph[target]), float(capacity)]
        reverse: list[float | int] = [source, len(self.graph[source]), 0.0]
        self.graph[source].append(forward)
        self.graph[target].append(reverse)

    def max_flow(self, source: int, sink: int) -> float:
        total = 0.0
        n_nodes = len(self.graph)
        while True:
            level = [-1] * n_nodes
            level[source] = 0
            queue = [source]
            for node in queue:
                for edge in self.graph[node]:
                    if edge[2] > 1e-12 and level[int(edge[0])] < 0:
                        level[int(edge[0])] = level[node] + 1
                        queue.append(int(edge[0]))
            if level[sink] < 0:
                return total

            cursor = [0] * n_nodes

            def send(node: int, amount: float) -> float:
                if node == sink:
                    return amount
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    target = int(edge[0])
                    if edge[2] > 1e-12 and level[target] == level[node] + 1:
                        pushed = send(target, min(amount, float(edge[2])))
                        if pushed > 1e-12:
                            edge[2] = float(edge[2]) - pushed
                            reverse = self.graph[target][int(edge[1])]
                            reverse[2] = float(reverse[2]) + pushed
                            return pushed
                    cursor[node] += 1
                return 0.0

            while True:
                pushed = send(source, float("inf"))
                if pushed <= 1e-12:
                    break
                total += pushed

    def source_reachable(self, source: int) -> list[bool]:
        seen = [False] * len(self.graph)
        seen[source] = True
        queue = [source]
        for node in queue:
            for edge in self.graph[node]:
                target = int(edge[0])
                if edge[2] > 1e-12 and not seen[target]:
                    seen[target] = True
                    queue.append(target)
        return seen


def cut_energy(
    assignment: np.ndarray,
    h_eff: np.ndarray,
    J_free: dict[tuple[int, int], float],
) -> float:
    """Evaluate E(z) = sum_i h_i*z_i + sum_(i,j) J_ij*|z_i-z_j|."""
    labels = np.asarray(assignment, dtype=np.int8).reshape(-1)
    unary = np.asarray(h_eff, dtype=np.float64).reshape(-1)
    if labels.size != unary.size:
        raise ValueError("assignment and h_eff must have the same number of entries")
    energy = float(np.dot(unary, labels))
    for (i, j), weight in J_free.items():
        energy += float(weight) * abs(int(labels[i]) - int(labels[j]))
    return energy


def _validate_problem(
    h_eff: np.ndarray,
    J_free: dict[tuple[int, int], float],
) -> np.ndarray:
    unary = np.asarray(h_eff, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(unary)):
        raise ValueError("h_eff must contain only finite values")
    n_nodes = unary.size
    for (i, j), weight in J_free.items():
        if i < 0 or j < 0 or i >= n_nodes or j >= n_nodes or i == j:
            raise ValueError(f"invalid pairwise edge {(i, j)} for {n_nodes} nodes")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(
                f"graph-cut requires finite nonnegative pairwise weights; {(i, j)}={weight}"
            )
    return unary


def solve_graph_cut(
    h_eff: np.ndarray,
    J_free: dict[tuple[int, int], float],
) -> tuple[np.ndarray, float]:
    """Minimize the reduced unary + Potts disagreement objective exactly by min-cut.

    Label 0 is the source side and label 1 is the sink side. Negative unary
    coefficients are shifted by a per-node constant, which does not change the
    minimizer. Pairwise capacities are checked at runtime for submodularity.
    """
    unary = _validate_problem(h_eff, J_free)
    n_nodes = unary.size
    if n_nodes == 0:
        return np.zeros(0, dtype=np.int32), 0.0

    source, sink = n_nodes, n_nodes + 1
    network = _Dinic(n_nodes + 2)
    for node, value in enumerate(unary):
        # The source/sink cut costs for labels 0/1 are shifted to nonnegative.
        cost_label_0 = max(0.0, -float(value))
        cost_label_1 = max(0.0, float(value))
        network.add_edge(source, node, cost_label_1)
        network.add_edge(node, sink, cost_label_0)

    for (i, j), weight in J_free.items():
        # Opposite directed arcs encode one undirected cut penalty.
        network.add_edge(i, j, float(weight))
        network.add_edge(j, i, float(weight))

    network.max_flow(source, sink)
    reachable = network.source_reachable(source)
    assignment = np.asarray([0 if reachable[i] else 1 for i in range(n_nodes)], dtype=np.int32)
    return assignment, cut_energy(assignment, unary, J_free)


def solve_exact(
    h_eff: np.ndarray,
    J_free: dict[tuple[int, int], float],
    max_nodes: int = EXACT_MAX_NODES,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, float]:
    """Brute-force the exact minimum for a small reduced QUBO/cut objective."""
    unary = _validate_problem(h_eff, J_free)
    n_nodes = unary.size
    if n_nodes > max_nodes:
        raise ValueError(f"exact enumeration limited to {max_nodes} nodes; got {n_nodes}")
    if n_nodes == 0:
        return np.zeros(0, dtype=np.int32), 0.0

    bit_shifts = np.arange(n_nodes, dtype=np.uint64)
    edge_indices = [(i, j, float(weight)) for (i, j), weight in J_free.items()]
    best_assignment = np.zeros(n_nodes, dtype=np.int32)
    best_energy = float("inf")
    total = 1 << n_nodes
    for start in range(0, total, chunk_size):
        indices = np.arange(start, min(start + chunk_size, total), dtype=np.uint64)
        labels = ((indices[:, None] >> bit_shifts) & 1).astype(np.int8)
        energies = labels @ unary
        for i, j, weight in edge_indices:
            energies += weight * np.abs(labels[:, i] - labels[:, j])
        local = int(np.argmin(energies))
        if float(energies[local]) < best_energy:
            best_energy = float(energies[local])
            best_assignment = labels[local].astype(np.int32)
    return best_assignment, best_energy
