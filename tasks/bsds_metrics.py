"""BSDS-style boundary metrics with tolerance and multiple human annotations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt


def _as_bool_map(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=bool)


def _f_measure(precision: float, recall: float, n_pred: int, n_gt: int) -> float:
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom else 0.0


def boundary_counts(
    prediction: np.ndarray,
    annotations: Sequence[np.ndarray],
    tolerance_px: float,
) -> dict:
    """Count matches; precision matches predictions to any annotator, recall sums annotators."""
    pred = _as_bool_map(prediction)
    gts = [_as_bool_map(gt) for gt in annotations]
    if not gts:
        raise ValueError("at least one ground-truth annotation is required")
    if any(gt.shape != pred.shape for gt in gts):
        raise ValueError("prediction and all annotations must have the same shape")
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be nonnegative")

    n_pred = int(pred.sum())
    n_gt_total = sum(int(gt.sum()) for gt in gts)
    if n_pred == 0 and n_gt_total == 0:
        return {"matched_pred": 0, "n_pred": 0, "matched_gt": 0,
                "n_gt": 0, "precision": 1.0, "recall": 1.0, "F": 1.0}
    if n_pred == 0 or n_gt_total == 0:
        return {"matched_pred": 0, "n_pred": n_pred,
                "matched_gt": 0, "n_gt": n_gt_total,
                "precision": 0.0, "recall": 0.0, "F": 0.0}

    union_gt = np.logical_or.reduce(gts)
    distance_to_any_gt = distance_transform_edt(~union_gt)
    matched_pred = int(np.count_nonzero(pred & (distance_to_any_gt <= tolerance_px)))

    distance_to_pred = distance_transform_edt(~pred)
    matched_gt = sum(
        int(np.count_nonzero(gt & (distance_to_pred <= tolerance_px))) for gt in gts
    )
    precision = matched_pred / n_pred
    recall = matched_gt / n_gt_total
    return {
        "matched_pred": matched_pred,
        "n_pred": n_pred,
        "matched_gt": matched_gt,
        "n_gt": n_gt_total,
        "precision": float(precision),
        "recall": float(recall),
        "F": float(_f_measure(precision, recall, n_pred, n_gt_total)),
    }


def pratt_fom(prediction: np.ndarray, ground_truth: np.ndarray, alpha: float = 1.0 / 9.0) -> float:
    """Pratt FOM for one annotator; empty-empty=1, exactly one empty=0."""
    pred = _as_bool_map(prediction)
    gt = _as_bool_map(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError("prediction and ground truth must have equal shape")
    n_pred, n_gt = int(pred.sum()), int(gt.sum())
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    distances = distance_transform_edt(~gt)[pred]
    return float(np.sum(1.0 / (1.0 + alpha * distances ** 2)) / max(n_pred, n_gt))


def h95(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Symmetric H95 for one annotator; empty-empty=0, exactly one empty=inf."""
    pred = _as_bool_map(prediction)
    gt = _as_bool_map(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError("prediction and ground truth must have equal shape")
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("inf")
    pred_to_gt = distance_transform_edt(~gt)[pred]
    gt_to_pred = distance_transform_edt(~pred)[gt]
    return float(max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95)))


def localization_multi_annotator(
    prediction: np.ndarray,
    annotations: Sequence[np.ndarray],
    alpha: float = 1.0 / 9.0,
) -> dict:
    """Per-image localization: mean Pratt and median H95 over annotators."""
    if not annotations:
        raise ValueError("at least one ground-truth annotation is required")
    return {
        "PrattFOM": float(np.mean([pratt_fom(prediction, gt, alpha) for gt in annotations])),
        "H95": float(np.median([h95(prediction, gt) for gt in annotations])),
    }


def non_maximum_suppression(magnitude: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Thin a soft edge map with four-bin gradient-direction NMS."""
    mag = np.asarray(magnitude, dtype=np.float64)
    dx = np.asarray(gx, dtype=np.float64)
    dy = np.asarray(gy, dtype=np.float64)
    if mag.shape != dx.shape or mag.shape != dy.shape:
        raise ValueError("magnitude and gradients must have identical shapes")
    angle = np.rad2deg(np.arctan2(dy, dx)) % 180.0
    padded = np.pad(mag, 1, mode="constant")
    out = np.zeros_like(mag)
    height, width = mag.shape
    for y in range(height):
        yy = y + 1
        for x in range(width):
            xx = x + 1
            direction = angle[y, x]
            if direction < 22.5 or direction >= 157.5:
                before, after = padded[yy, xx - 1], padded[yy, xx + 1]
            elif direction < 67.5:
                before, after = padded[yy - 1, xx + 1], padded[yy + 1, xx - 1]
            elif direction < 112.5:
                before, after = padded[yy - 1, xx], padded[yy + 1, xx]
            else:
                before, after = padded[yy - 1, xx - 1], padded[yy + 1, xx + 1]
            if mag[y, x] >= before and mag[y, x] >= after:
                out[y, x] = mag[y, x]
    return out


def threshold_values(edge_maps: Sequence[np.ndarray], count: int = 99) -> np.ndarray:
    """Shared thresholds on normalized edge maps: evenly spaced from 0.01 to 1."""
    if count < 1:
        raise ValueError("threshold count must be positive")
    return np.linspace(0.01, 1.0, count, dtype=np.float64)


def evaluate_dataset(
    edge_maps: Sequence[np.ndarray],
    annotation_sets: Sequence[Sequence[np.ndarray]],
    tolerance_px: float,
    thresholds: np.ndarray | None = None,
    n_thresholds: int = 99,
    alpha: float = 1.0 / 9.0,
) -> dict:
    """Compute dataset ODS/OIS and per-image localization at the ODS threshold."""
    if len(edge_maps) != len(annotation_sets) or not edge_maps:
        raise ValueError("edge_maps and annotation_sets must be non-empty and aligned")
    if thresholds is None:
        thresholds = threshold_values(edge_maps, n_thresholds)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    per_image_curves: list[list[dict]] = []
    aggregate_f = []
    for threshold in thresholds:
        total_matched_pred = total_pred = total_matched_gt = total_gt = 0
        image_scores = []
        for edge_map, annotations in zip(edge_maps, annotation_sets):
            pred = np.asarray(edge_map, dtype=np.float64) >= threshold
            counts = boundary_counts(pred, annotations, tolerance_px)
            total_matched_pred += counts["matched_pred"]
            total_pred += counts["n_pred"]
            total_matched_gt += counts["matched_gt"]
            total_gt += counts["n_gt"]
            image_scores.append(counts["F"])
        if total_pred == 0 and total_gt == 0:
            precision = recall = 1.0
        elif total_pred == 0 or total_gt == 0:
            precision = recall = 0.0
        else:
            precision = total_matched_pred / total_pred
            recall = total_matched_gt / total_gt
        aggregate_f.append(float(_f_measure(precision, recall, total_pred, total_gt)))
        per_image_curves.append(image_scores)

    ods_index = int(np.argmax(aggregate_f))
    best_per_image = np.max(np.asarray(per_image_curves), axis=0)
    per_image_at_ods = []
    for edge_map, annotations in zip(edge_maps, annotation_sets):
        pred = np.asarray(edge_map, dtype=np.float64) >= thresholds[ods_index]
        counts = boundary_counts(pred, annotations, tolerance_px)
        localization = localization_multi_annotator(pred, annotations, alpha)
        per_image_at_ods.append({**counts, **localization})
    return {
        "ODS": aggregate_f[ods_index],
        "ODS_threshold": float(thresholds[ods_index]),
        "OIS": float(np.mean(best_per_image)),
        "per_image_best_F": best_per_image.tolist(),
        "per_image_at_ODS": per_image_at_ods,
        "thresholds": thresholds.tolist(),
    }
