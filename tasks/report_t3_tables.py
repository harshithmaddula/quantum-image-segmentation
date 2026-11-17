"""Build Tables 3 and 4 from the accepted T3 run without re-optimizing QAOA."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantum_seg.data import load_gray, to_pow2
from quantum_seg.superpixel import sp_adjacency
from run_t3_qaoa_pilot import exact_maxcut, parameterized_circuit, slic_partition


CSV_PATH = ROOT / "results" / "tasks" / "T3" / "qaoa_maxcut_39.csv"
OUT = ROOT / "results" / "tasks" / "T3"


def graph_from_image(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float]]]:
    image = to_pow2(load_gray(str(path)), 128, mode="crop")
    gray, labels, n_nodes = slic_partition(image)
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=n_nodes).astype(np.float64)
    means = np.bincount(flat, weights=gray.ravel(), minlength=n_nodes) / counts / 255.0
    sigma = float(np.std(means))
    if sigma <= np.finfo(np.float64).eps:
        sigma = 1.0
    edges = [
        (left, right, float(1.0 - np.exp(-((means[left] - means[right]) ** 2)
                                           / (2.0 * sigma ** 2))))
        for left, right in sp_adjacency(labels, n_nodes)
    ]
    return labels, means, edges


def objective_at_saved_parameters(n_nodes: int, p: int,
                                  edges: list[tuple[int, int, float]], row: dict) -> float:
    circuit = parameterized_circuit(n_nodes, p, edges)
    gamma = [float(value) for value in row["gamma_parameters"].split(";")]
    beta = [float(value) for value in row["beta_parameters"].split(";")]
    gamma_params = sorted((item for item in circuit.parameters if item.name.startswith("gamma")),
                          key=lambda item: int(item.name.split("[")[1][:-1]))
    beta_params = sorted((item for item in circuit.parameters if item.name.startswith("beta")),
                         key=lambda item: int(item.name.split("[")[1][:-1]))
    bound = circuit.assign_parameters(dict(zip(gamma_params, gamma)) |
                                      dict(zip(beta_params, beta)))
    state = Statevector.from_instruction(bound.remove_final_measurements(inplace=False))
    probabilities = state.probabilities()
    basis = np.arange(probabilities.size, dtype=np.uint32)
    bits = ((basis[:, None] >> np.arange(n_nodes, dtype=np.uint32)) & 1).astype(np.uint8)
    cuts = np.zeros(probabilities.size, dtype=np.float64)
    for left, right, weight in edges:
        cuts += weight * (bits[:, left] != bits[:, right])
    return float(np.dot(probabilities, cuts))


def pixel_metrics(candidate_nodes: np.ndarray, exact_nodes: np.ndarray,
                  labels: np.ndarray) -> dict[str, float]:
    reference = exact_nodes[labels].astype(bool)
    prediction = candidate_nodes[labels].astype(bool)
    if np.mean(prediction == reference) < np.mean(~prediction == reference):
        prediction = ~prediction
    true_positive = int(np.count_nonzero(prediction & reference))
    pred_count = int(prediction.sum())
    ref_count = int(reference.sum())
    union = int(np.count_nonzero(prediction | reference))
    return {
        "IoU": true_positive / union if union else 1.0,
        "Dice": 2.0 * true_positive / (pred_count + ref_count)
                if pred_count + ref_count else 1.0,
        "PixelAcc": float(np.mean(prediction == reference)),
    }


def mode_nodes(bitstring: str, n_nodes: int) -> np.ndarray:
    bits = bitstring.replace(" ", "")
    return np.asarray([int(bit) for bit in bits[::-1]], dtype=np.uint8)[:n_nodes]


def fmt_triplet(values: dict[str, float]) -> str:
    return "/".join(f"{values[key]:.3f}" for key in ("IoU", "Dice", "PixelAcc"))


def mean_std(rows: list[dict[str, float]]) -> dict[str, str]:
    result = {}
    for key in ("IoU", "Dice", "PixelAcc"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = f"{np.mean(values):.3f} +/- {np.std(values):.3f}"
    return result


def triplet_mean_std(rows: list[dict[str, float]]) -> str:
    stats = mean_std(rows)
    return "/".join(stats[key] for key in ("IoU", "Dice", "PixelAcc"))


def ratio_metrics(numerator: dict[str, float], denominator: dict[str, float]) -> dict[str, float]:
    return {key: numerator[key] / denominator[key] if denominator[key] else float("inf")
            for key in ("IoU", "Dice", "PixelAcc")}


def main() -> None:
    with CSV_PATH.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    images = sorted({row["image"] for row in rows})
    if len(images) != 39 or len(rows) != 156 or any(row["status"] != "ok" for row in rows):
        raise SystemExit("Expected 39 images x 4 successful depths in the accepted T3 CSV")

    by_image = defaultdict(dict)
    for row in rows:
        by_image[row["image"]][int(row["p"])] = row
    evaluated = {}
    for image_name in images:
        labels, means, edges = graph_from_image(ROOT / "misc" / image_name)
        n_nodes = means.size
        exact_nodes, exact_cut = exact_maxcut(n_nodes, edges)
        if not np.isclose(exact_cut, float(by_image[image_name][1]["exact_maxcut"]), rtol=1e-10):
            raise ValueError(f"Reconstructed exact cut differs for {image_name}")
        otsu_threshold = float(by_image[image_name][1]["otsu_threshold"])
        otsu_nodes = (means * 255.0 > otsu_threshold).astype(np.uint8)
        otsu = pixel_metrics(otsu_nodes, exact_nodes, labels)
        objectives, qaoa_metrics = {}, {}
        for p in range(1, 5):
            row = by_image[image_name][p]
            objectives[p] = objective_at_saved_parameters(n_nodes, p, edges, row)
            qaoa_metrics[p] = pixel_metrics(
                mode_nodes(row["final_mode_bitstring"], n_nodes), exact_nodes, labels
            )
        selected_p = max(range(1, 5), key=lambda p: (objectives[p], -p))
        selected = qaoa_metrics[selected_p]
        evaluated[image_name] = {
            "labels": labels, "edges": edges, "exact": exact_nodes,
            "exact_cut": exact_cut, "otsu": otsu, "qaoa": qaoa_metrics,
            "objectives": objectives, "selected_p": selected_p,
            "selected": selected,
        }

    report = [
        "# Table 3: T3 Segmentation Metrics",
        "",
        "Pixel masks are formed by expanding each superpixel label to 128x128. Binary label orientation is aligned against the exact MaxCut partition by choosing the orientation with higher pixel accuracy. Exact metrics are therefore 1.000 by construction. QAOA uses the saved independent final-sample mode bitstring; Otsu thresholds each superpixel mean with the image-level Otsu threshold.",
        "Depth selection uses the exact statevector expected MaxCut objective evaluated at each saved best parameter vector, not the final sample. The accepted CSV did not persist the optimizer's stochastic best objective, so this is a deterministic objective reconstruction; no COBYLA optimization was rerun. Means and standard deviations are across 39 images (population SD).",
        "",
        "Triplets are IoU / Dice / PixelAcc. QAOA/Otsu is the per-metric ratio.",
        "",
        "| Image | Selected p | Exact IoU/Dice/Acc | QAOA IoU/Dice/Acc | Otsu IoU/Dice/Acc | QAOA/Otsu IoU/Dice/Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    selected_qaoa, all_otsu, selected_ratios = [], [], []
    for image_name in images:
        item = evaluated[image_name]
        qaoa, otsu = item["selected"], item["otsu"]
        selected_qaoa.append(qaoa)
        all_otsu.append(otsu)
        selected_ratios.append(ratio_metrics(qaoa, otsu))
        report.append(
            f"| {image_name} | {item['selected_p']} | 1.000/1.000/1.000 | "
            f"{fmt_triplet(qaoa)} | {fmt_triplet(otsu)} | "
            f"{fmt_triplet(ratio_metrics(qaoa, otsu))} |"
        )
    report.append(
        f"| Mean +/- SD (selected) | - | 1.000/1.000/1.000 | "
        f"{triplet_mean_std(selected_qaoa)} | {triplet_mean_std(all_otsu)} | "
        f"{triplet_mean_std(selected_ratios)} |"
    )
    fixed_stats = {}
    for p in range(1, 5):
        qaoa_rows = [evaluated[name]["qaoa"][p] for name in images]
        ratios = [ratio_metrics(evaluated[name]["qaoa"][p], evaluated[name]["otsu"])
                  for name in images]
        fixed_stats[p] = qaoa_rows
        report.append(
            f"| Mean +/- SD (fixed p={p}) | {p} | 1.000/1.000/1.000 | "
            f"{triplet_mean_std(qaoa_rows)} | {triplet_mean_std(all_otsu)} | "
            f"{triplet_mean_std(ratios)} |"
        )
    (OUT / "Table3.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    representatives = (images[0], images[len(images) // 2], images[-1])
    table4 = [
        "# Table 4: T3 Depth and Runtime",
        "",
        "Representative images are the first, middle, and last filenames in sorted order. Runtime is the recorded per-depth wall time, including transpilation, five optimizer restarts, and the final 4,096-shot sample; it is not kernel-only simulator time. CPU: Intel Core i5-12450H.",
        "",
        "| Image | p | Parameters (gamma; beta) | Transpiled depth | Wall time (s) | IoU* | Dice* |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for image_name in representatives:
        for p in range(1, 5):
            row = by_image[image_name][p]
            metrics = evaluated[image_name]["qaoa"][p]
            parameters = f"{row['gamma_parameters']} ; {row['beta_parameters']}"
            table4.append(
                f"| {image_name} | {p} | `{parameters}` | {row['transpiled_depth']} | "
                f"{float(row['wall_clock_seconds']):.2f} | {metrics['IoU']:.3f} | "
                f"{metrics['Dice']:.3f} |"
            )
    table4 += [
        "",
        "| p | Mean wall time (s) | P95 wall time (s) | n |",
        "|---:|---:|---:|---:|",
    ]
    for p in range(1, 5):
        times = np.asarray([float(by_image[name][p]["wall_clock_seconds"]) for name in images])
        table4.append(f"| {p} | {np.mean(times):.2f} | {np.percentile(times, 95):.2f} | {times.size} |")
    (OUT / "Table4.md").write_text("\n".join(table4) + "\n", encoding="utf-8")

    summary = [
        "# T3 Full-Run Summary",
        "",
        "The accepted run contains 39 sorted USC-SIPI TIFF images and 156 successful image/depth rows. Each result uses K=16 SLIC superpixels, five COBYLA restarts per depth, 4,096 optimization shots per evaluation, and a separate 4,096-shot final sample.",
        "",
        "Random-assignment expected cut is one half of the total edge weight; its ratio is averaged per image before aggregation. Final expected-cut values and ratios use the independent final sample.",
        "",
        "| p | Mean final expected cut +/- SD | Mean expected/exact | Mean random/exact | Mean mode/exact | Mean wall (s) | P95 wall (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in range(1, 5):
        group = [by_image[name][p] for name in images]
        expected = np.asarray([float(row["final_expected_cut"]) for row in group])
        expected_ratio = np.asarray([float(row["final_expected_ratio_to_exact"]) for row in group])
        mode_ratio = np.asarray([float(row["final_mode_ratio_to_exact"]) for row in group])
        random_ratio = np.asarray([
            0.5 * sum(weight for _, _, weight in evaluated[name]["edges"])
            / evaluated[name]["exact_cut"] for name in images
        ])
        times = np.asarray([float(row["wall_clock_seconds"]) for row in group])
        summary.append(
            f"| {p} | {np.mean(expected):.4f} +/- {np.std(expected):.4f} | "
            f"{np.mean(expected_ratio):.4f} | {np.mean(random_ratio):.4f} | "
            f"{np.mean(mode_ratio):.4f} | {np.mean(times):.2f} | "
            f"{np.percentile(times, 95):.2f} |"
        )
    summary += [
        "",
        "At p=2, mean final expected/exact scoring (0.6779) is below p=1 (0.7552); this indicates the optimizer failed to match the p=1 result on this run, not that p=2 is intrinsically inferior.",
        "",
        "See `Table3.md` for selected-depth and fixed-depth pixel metrics and `Table4.md` for representative parameters, circuit depths, and runtimes.",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    for p in range(1, 5):
        group = [by_image[name][p] for name in images]
        ratio = np.mean([float(row["final_expected_ratio_to_exact"]) for row in group])
        print(f"p={p}: mean final expected/exact={ratio:.4f}, n={len(group)}")


if __name__ == "__main__":
    main()