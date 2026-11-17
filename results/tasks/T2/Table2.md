# T2: QHED on BSDS500 validation

This run uses the first 30 validation images in sorted filename order, center-cropped to 256x256. The `FULL RUN` label means all 30 requested images, not the entire BSDS500 validation split. Ground-truth boundary maps are center-cropped without interpolation.
Baseline tuning used 200 BSDS500 training images only; see `baseline_tuning.json` for the grid and chosen parameters.
Matching tolerance: 2.715290 pixels (0.0075 x 256x256 diagonal); thresholds: 99 evenly spaced values from 0.01 to 1.0.
Precision matches predictions to any annotator; recall accumulates matches separately over all annotators. Pratt is averaged over annotators per image; H95 is median over annotators per image, then median over images. Empty conventions are documented in `tasks/ASSUMPTIONS.md`.

| Method | ODS | OIS | Pratt FOM (mean) | H95 (median px) |
|---|---:|---:|---:|---:|
| QHED | 0.4723 | 0.5357 | 0.2966 | 46.777 |
| Sobel | 0.5199 | 0.5724 | 0.3177 | 46.382 |
| Canny | 0.4484 | 0.4625 | 0.2391 | 49.799 |

Externally tuned baseline F on validation (not ODS/OIS):
- Sobel: 0.5377
- Canny: 0.4625

| Method | Mean seconds/image | P95 seconds/image | n |
|---|---:|---:|---:|
| QHED | 81.0465 | 89.5612 | 30 |
| Sobel | 1.0356 | 1.2976 | 30 |
| Canny | 1.1481 | 1.4089 | 30 |

Runtime scope: QHED uses Step A `per_image_qhed.csv` totals for both directional scans, including state preparation, transpilation, simulation, and postprocessing; image loading and cache writes are excluded. Sobel and Canny use Step B totals for edge-map generation, the 99-threshold boundary sweep, and ODS-threshold localization (Pratt/H95); baseline tuning and the unreported warm-up are excluded.

See `per_image_metrics.csv` for image-level values and resumable sweep counts.
