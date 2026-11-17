"""Fast deterministic unit tests for the T2 QHED circuit and BSDS metrics."""

import unittest

import numpy as np

from tasks.bsds_metrics import (
    boundary_counts,
    evaluate_dataset,
    h95,
    localization_multi_annotator,
    pratt_fom,
)
from tasks.qhed_yao import qhed_yao


class QHEDForwardDifferenceTests(unittest.TestCase):
    def test_auxiliary_branch_matches_forward_differences_8x8_and_16x16(self):
        rng = np.random.default_rng(20261115)
        for side in (8, 16):
            with self.subTest(side=side):
                image = rng.integers(0, 256, size=(side, side)).astype(np.float64)
                result = qhed_yao(image, max_parallel_threads=2)
                expected_x = np.zeros_like(image)
                expected_y = np.zeros_like(image)
                expected_x[:, :-1] = image[:, :-1] - image[:, 1:]
                expected_y[:-1, :] = image[:-1, :] - image[1:, :]
                self.assertLess(float(np.max(np.abs(result.gx - expected_x))), 1e-6)
                self.assertLess(float(np.max(np.abs(result.gy - expected_y))), 1e-6)
                self.assertEqual(result.position_qubits, 2 * (side.bit_length() - 1))
                self.assertEqual(result.total_qubits, result.position_qubits + 1)


class BsdMetricTests(unittest.TestCase):
    def test_perfect_and_offset_boundary_cases(self):
        reference = np.zeros((7, 7), dtype=bool)
        reference[3, 3] = True
        perfect = boundary_counts(reference, [reference], tolerance_px=0)
        self.assertEqual((perfect["precision"], perfect["recall"], perfect["F"]), (1.0, 1.0, 1.0))

        shifted = np.zeros_like(reference)
        shifted[3, 5] = True
        mismatch = boundary_counts(shifted, [reference], tolerance_px=1)
        self.assertEqual(mismatch["F"], 0.0)
        self.assertAlmostEqual(pratt_fom(shifted, reference), 9.0 / 13.0)
        self.assertAlmostEqual(h95(shifted, reference), 2.0)

    def test_multiple_annotator_precision_union_and_recall_sum(self):
        pred = np.zeros((7, 7), dtype=bool)
        pred[1, 1] = True
        first = pred.copy()
        second = np.zeros_like(pred)
        second[5, 5] = True
        score = boundary_counts(pred, [first, second], tolerance_px=0)
        self.assertEqual(score["precision"], 1.0)
        self.assertEqual(score["recall"], 0.5)
        self.assertAlmostEqual(score["F"], 2.0 / 3.0)
        localization = localization_multi_annotator(pred, [first, second])
        distant_pratt = 1.0 / (1.0 + (1.0 / 9.0) * (4.0 ** 2 + 4.0 ** 2))
        self.assertAlmostEqual(localization["PrattFOM"], (1.0 + distant_pratt) / 2.0)
        self.assertAlmostEqual(localization["H95"], np.sqrt(32.0) / 2.0)

    def test_empty_conventions(self):
        empty = np.zeros((5, 5), dtype=bool)
        point = empty.copy()
        point[2, 2] = True
        both_empty = boundary_counts(empty, [empty], tolerance_px=1)
        self.assertEqual(both_empty["F"], 1.0)
        self.assertEqual(pratt_fom(empty, empty), 1.0)
        self.assertEqual(h95(empty, empty), 0.0)

        one_empty = boundary_counts(point, [empty], tolerance_px=1)
        self.assertEqual(one_empty["F"], 0.0)
        self.assertEqual(pratt_fom(point, empty), 0.0)
        self.assertTrue(np.isinf(h95(point, empty)))

    def test_ods_ois_known_two_image_case(self):
        maps = [np.zeros((2, 2)), np.zeros((2, 2))]
        maps[0][0, 0] = 1.0
        maps[1][0, 0] = 1.0
        point = np.zeros((2, 2), dtype=bool)
        point[0, 0] = True
        annotations = [[point], [point]]
        result = evaluate_dataset(
            maps,
            annotations,
            tolerance_px=0,
            thresholds=np.array([0.5, 1.0]),
        )
        self.assertEqual(result["ODS"], 1.0)
        self.assertEqual(result["OIS"], 1.0)
        self.assertEqual(result["ODS_threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
