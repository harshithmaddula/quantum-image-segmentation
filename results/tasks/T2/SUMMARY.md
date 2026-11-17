# T2 Results Summary

## Scope

The evaluation covers the first 30 BSDS500 validation images in sorted filename order, center-cropped to 256x256. Ground-truth boundary maps are center-cropped without interpolation. Sobel and Canny parameters were tuned using 200 BSDS500 training images only.

Matching tolerance for boundary F is `0.0075 x sqrt(256^2 + 256^2) = 2.715290 pixels`.

## Results

| Method | ODS | OIS | Pratt FOM | H95 median (px) |
|---|---:|---:|---:|---:|
| QHED | 0.4723 | 0.5357 | 0.2966 | 46.777 |
| Sobel | 0.5199 | 0.5724 | 0.3177 | 46.382 |
| Canny | 0.4484 | 0.4625 | 0.2391 | 49.799 |

## H95 Convention

Each edge map is binarized at its method-specific ODS threshold selected by aggregate boundary F over the 30 validation images: QHED `0.22214286`, Sobel `0.23224490`, and Canny `0.01000000`. H95 is the maximum of the two directed 95th-percentile Euclidean distances, computed separately against each ground-truth annotator. QHED and Sobel maps receive non-maximum suppression before thresholding; Canny uses its native non-maximum suppression. No additional thinning is applied after thresholding. H95 is a localization distance and does not use the boundary-matching tolerance above.

Per-image H95 is the median across annotators; the reported H95 is the median of those per-image values. If a method has no predicted edges while an annotator has boundaries, H95 is infinity; if both are empty, H95 is zero. When one or more annotators are empty, the same per-annotator convention is applied before taking the median.

## QHED State Preparation

A focused retry diagnostic for image `106024` succeeded with 12-digit amplitude rounding on one scan. The persisted final Step A row records `none/none`, meaning the full-run edge map was produced without rounding fallback. The per-image CSV records the persisted run values.

## Timing

Table 2 reports QHED Step A totals and Sobel/Canny Step B totals. QHED includes both directional scans, state preparation, transpilation, simulation, and postprocessing; it excludes image loading and cache writes. Sobel and Canny include edge-map generation, the 99-threshold boundary sweep, and localization at the ODS threshold, including Pratt FOM and H95; they exclude training-set tuning and the unreported warm-up. P95 uses linear interpolation.