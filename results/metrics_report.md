# Quantum Image Segmentation -- Mandatory Metrics vs SOTA

## 1. NEQR encoding fidelity + quantum resources (spatial-preservation gate)

_Lossless representation: exact reconstruction => PSNR=inf, SSIM=1, MSE=0 (every boundary/texture pixel preserved). Resource columns substantiate the 'quantum' claim (per Ruan 2021)._

| image         | size  | qubits | gates | depth | PSNR | SSIM   | MSE    | EPI    | sim_time |
|---------------|-------|--------|-------|-------|------|--------|--------|--------|----------|
| 5.1.12.tiff   | 8x8   | 14     | 302   | 297   | inf  | 1.0000 | 0.0000 | 1.0000 | 0.72s    |
| 5.1.12.tiff   | 16x16 | 16     | 1246  | 1239  | inf  | 1.0000 | 0.0000 | 1.0000 | 8.18s    |
| 5.1.12.tiff   | 32x32 | 18     | 4991  | 4982  | inf  | 1.0000 | 0.0000 | 1.0000 | 200.44s  |
| 4.2.03.tiff   | 8x8   | 14     | 306   | 301   | inf  | 1.0000 | 0.0000 | 1.0000 | 0.61s    |
| 4.2.03.tiff   | 16x16 | 16     | 1106  | 1099  | inf  | 1.0000 | 0.0000 | 1.0000 | 12.22s   |
| 4.2.03.tiff   | 32x32 | 18     | 4339  | 4330  | inf  | 1.0000 | 0.0000 | 1.0000 | 126.88s  |
| boat.512.tiff | 8x8   | 14     | 337   | 332   | inf  | 1.0000 | 0.0000 | 1.0000 | 0.46s    |
| boat.512.tiff | 16x16 | 16     | 1286  | 1279  | inf  | 1.0000 | 0.0000 | 1.0000 | 5.39s    |
| boat.512.tiff | 32x32 | 18     | 4759  | 4750  | inf  | 1.0000 | 0.0000 | 1.0000 | 91.75s   |

## 2. QHED vs classical baselines on misc (no ground truth)

_misc has no masks -> tolerance-aware agreement with classical edges + edge density. Supervised SOTA numbers are in stage 3 (BSDS)._

| image         | size    | comparison        | QHED_edge_density | baseline_edge_density | tol1_agreement |
|---------------|---------|-------------------|-------------------|-----------------------|----------------|
| 5.1.12.tiff   | 128x128 | QHED-vs-sobel     | 0.1500            | 0.0559                | 0.9577         |
| 5.1.12.tiff   | 128x128 | QHED-vs-scharr    | 0.1500            | 0.0560                | 0.9590         |
| 5.1.12.tiff   | 128x128 | QHED-vs-laplacian | 0.1500            | 0.0084                | 0.9596         |
| 5.1.12.tiff   | 128x128 | QHED-vs-canny     | 0.1500            | 0.0975                | 0.4258         |
| boat.512.tiff | 128x128 | QHED-vs-sobel     | 0.1501            | 0.0113                | 0.9295         |
| boat.512.tiff | 128x128 | QHED-vs-scharr    | 0.1501            | 0.0114                | 0.9328         |
| boat.512.tiff | 128x128 | QHED-vs-laplacian | 0.1501            | 0.0041                | 0.8758         |
| boat.512.tiff | 128x128 | QHED-vs-canny     | 0.1501            | 0.1788                | 0.8204         |
| 4.2.03.tiff   | 128x128 | QHED-vs-sobel     | 0.1501            | 0.0287                | 0.8907         |
| 4.2.03.tiff   | 128x128 | QHED-vs-scharr    | 0.1501            | 0.0293                | 0.9032         |
| 4.2.03.tiff   | 128x128 | QHED-vs-laplacian | 0.1501            | 0.0123                | 0.9010         |
| 4.2.03.tiff   | 128x128 | QHED-vs-canny     | 0.1501            | 0.0842                | 0.5366         |
| house.tiff    | 128x128 | QHED-vs-sobel     | 0.1501            | 0.0784                | 0.9553         |
| house.tiff    | 128x128 | QHED-vs-scharr    | 0.1501            | 0.0763                | 0.9547         |
| house.tiff    | 128x128 | QHED-vs-laplacian | 0.1501            | 0.0078                | 0.9009         |
| house.tiff    | 128x128 | QHED-vs-canny     | 0.1501            | 0.1351                | 0.4845         |

## 3. BSDS500 supervised boundary SOTA (ODS / OIS / Boundary-F1 / Pratt / Hausdorff)

_Run with `--bsds` to download BSDS500 and compute supervised SOTA metrics._

## 4. Graph Cut vs QAOA vs GrabCut on Real Ground Truth

_Run with `--grabcut` to evaluate graph cut, QAOA and GrabCut on GrabCut-50 masks._

## 5. Selected Solver on Misc Images (proxy trimap, no human GT)

Evaluated on 39 misc images at 128×128, n_segments=12.

| method   | image           | IoU*   | Dice*  | PixelAcc* | n_free | latency |
|----------|-----------------|--------|--------|-----------|--------|---------|
| graphcut | 4.1.01.tiff     | 0.0755 | 0.1404 | 0.7618    | 12     | 8.7s    |
| GrabCut  | 4.1.01.tiff     | 0.0840 | 0.1549 | 0.6829    | -      | -       |
| graphcut | 4.1.02.tiff     | 0.0466 | 0.0891 | 0.3900    | 11     | 0.8s    |
| GrabCut  | 4.1.02.tiff     | 0.0891 | 0.1637 | 0.6341    | -      | -       |
| graphcut | 4.1.03.tiff     | 0.1631 | 0.2804 | 0.9647    | 7      | 0.7s    |
| GrabCut  | 4.1.03.tiff     | 0.2177 | 0.3575 | 0.9678    | -      | -       |
| graphcut | 4.1.04.tiff     | 0.0516 | 0.0982 | 0.4090    | 10     | 0.9s    |
| GrabCut  | 4.1.04.tiff     | 0.1113 | 0.2004 | 0.7110    | -      | -       |
| graphcut | 4.1.05.tiff     | 0.0608 | 0.1146 | 0.5870    | 9      | 0.7s    |
| GrabCut  | 4.1.05.tiff     | 0.1116 | 0.2009 | 0.7140    | -      | -       |
| graphcut | 4.1.06.tiff     | 0.0412 | 0.0791 | 0.1274    | 13     | 0.7s    |
| GrabCut  | 4.1.06.tiff     | 0.0992 | 0.1805 | 0.7008    | -      | -       |
| graphcut | 4.1.07.tiff     | 0.1904 | 0.3199 | 0.9410    | 6      | 0.7s    |
| GrabCut  | 4.1.07.tiff     | 0.3151 | 0.4792 | 0.9399    | -      | -       |
| graphcut | 4.1.08.tiff     | 0.0811 | 0.1500 | 0.8544    | 6      | 1.1s    |
| GrabCut  | 4.1.08.tiff     | 0.1177 | 0.2106 | 0.7967    | -      | -       |
| graphcut | 4.2.01.tiff     | 0.0443 | 0.0849 | 0.1800    | 6      | 0.8s    |
| GrabCut  | 4.2.01.tiff     | 0.1272 | 0.2256 | 0.7444    | -      | -       |
| graphcut | 4.2.03.tiff     | 0.0547 | 0.1036 | 0.6207    | 11     | 0.8s    |
| GrabCut  | 4.2.03.tiff     | 0.1744 | 0.2970 | 0.8636    | -      | -       |
| graphcut | 4.2.05.tiff     | 0.0371 | 0.0716 | 0.1483    | 12     | 0.8s    |
| GrabCut  | 4.2.05.tiff     | 0.1919 | 0.3219 | 0.8765    | -      | -       |
| graphcut | 4.2.06.tiff     | 0.0438 | 0.0839 | 0.1981    | 8      | 1.0s    |
| GrabCut  | 4.2.06.tiff     | 0.1591 | 0.2745 | 0.8251    | -      | -       |
| graphcut | 4.2.07.tiff     | 0.0625 | 0.1177 | 0.7432    | 10     | 0.7s    |
| GrabCut  | 4.2.07.tiff     | 0.0709 | 0.1324 | 0.5155    | -      | -       |
| graphcut | 5.1.09.tiff     | 0.0385 | 0.0742 | 0.0701    | 13     | 0.6s    |
| GrabCut  | 5.1.09.tiff     | 0.1371 | 0.2411 | 0.7933    | -      | -       |
| graphcut | 5.1.10.tiff     | 0.0280 | 0.0545 | 0.9190    | 15     | 0.8s    |
| GrabCut  | 5.1.10.tiff     | 0.1596 | 0.2753 | 0.8115    | -      | -       |
| graphcut | 5.1.11.tiff     | 0.0544 | 0.1032 | 0.3303    | 6      | 0.7s    |
| GrabCut  | 5.1.11.tiff     | 0.1205 | 0.2151 | 0.8125    | -      | -       |
| graphcut | 5.1.12.tiff     | 0.0481 | 0.0918 | 0.2943    | 10     | 0.7s    |
| GrabCut  | 5.1.12.tiff     | 0.1383 | 0.2430 | 0.8068    | -      | -       |
| graphcut | 5.1.13.tiff     | 0.0838 | 0.1547 | 0.9220    | 14     | 0.8s    |
| GrabCut  | 5.1.13.tiff     | 0.1477 | 0.2574 | 0.8151    | -      | -       |
| graphcut | 5.1.14.tiff     | 0.0748 | 0.1393 | 0.9423    | 11     | 0.6s    |
| GrabCut  | 5.1.14.tiff     | 0.1388 | 0.2437 | 0.7930    | -      | -       |
| graphcut | 5.2.08.tiff     | 0.0401 | 0.0771 | 0.0998    | 13     | 0.6s    |
| GrabCut  | 5.2.08.tiff     | 0.1798 | 0.3048 | 0.8484    | -      | -       |
| graphcut | 5.2.09.tiff     | 0.0379 | 0.0730 | 0.1231    | 14     | 0.7s    |
| GrabCut  | 5.2.09.tiff     | 0.2680 | 0.4227 | 0.9249    | -      | -       |
| graphcut | 5.2.10.tiff     | 0.0686 | 0.1284 | 0.8784    | 13     | 0.6s    |
| GrabCut  | 5.2.10.tiff     | 0.2413 | 0.3888 | 0.8976    | -      | -       |
| graphcut | 5.3.01.tiff     | 0.0420 | 0.0806 | 0.3987    | 12     | 0.7s    |
| GrabCut  | 5.3.01.tiff     | 0.0846 | 0.1560 | 0.5970    | -      | -       |
| graphcut | 5.3.02.tiff     | 0.0408 | 0.0784 | 0.9054    | 14     | 0.7s    |
| GrabCut  | 5.3.02.tiff     | 0.1313 | 0.2320 | 0.7655    | -      | -       |
| graphcut | 7.1.01.tiff     | 0.0387 | 0.0744 | 0.0577    | 12     | 0.8s    |
| GrabCut  | 7.1.01.tiff     | 0.2735 | 0.4295 | 0.9186    | -      | -       |
| graphcut | 7.1.02.tiff     | 0.0601 | 0.1133 | 0.9533    | 9      | 0.6s    |
| GrabCut  | 7.1.02.tiff     | 0.2806 | 0.4382 | 0.9634    | -      | -       |
| graphcut | 7.1.03.tiff     | 0.0407 | 0.0782 | 0.1554    | 10     | 0.6s    |
| GrabCut  | 7.1.03.tiff     | 0.1781 | 0.3024 | 0.8713    | -      | -       |
| graphcut | 7.1.04.tiff     | 0.0393 | 0.0757 | 0.9556    | 11     | 0.6s    |
| GrabCut  | 7.1.04.tiff     | 0.2275 | 0.3707 | 0.8943    | -      | -       |
| graphcut | 7.1.05.tiff     | 0.0375 | 0.0722 | 0.1235    | 12     | 0.7s    |
| GrabCut  | 7.1.05.tiff     | 0.1797 | 0.3046 | 0.8377    | -      | -       |
| graphcut | 7.1.06.tiff     | 0.0376 | 0.0725 | 0.0933    | 15     | 0.8s    |
| GrabCut  | 7.1.06.tiff     | 0.2253 | 0.3677 | 0.8974    | -      | -       |
| graphcut | 7.1.07.tiff     | 0.0390 | 0.0751 | 0.0768    | 12     | 0.6s    |
| GrabCut  | 7.1.07.tiff     | 0.3688 | 0.5388 | 0.9523    | -      | -       |
| graphcut | 7.1.08.tiff     | 0.0537 | 0.1019 | 0.7735    | 10     | 0.7s    |
| GrabCut  | 7.1.08.tiff     | 0.3069 | 0.4697 | 0.9546    | -      | -       |
| graphcut | 7.1.09.tiff     | 0.0799 | 0.1480 | 0.8924    | 9      | 0.6s    |
| GrabCut  | 7.1.09.tiff     | 0.3537 | 0.5226 | 0.9547    | -      | -       |
| graphcut | 7.1.10.tiff     | 0.0922 | 0.1689 | 0.9196    | 13     | 0.6s    |
| GrabCut  | 7.1.10.tiff     | 0.2399 | 0.3870 | 0.8923    | -      | -       |
| graphcut | 7.2.01.tiff     | 0.0473 | 0.0904 | 0.3760    | 10     | 0.7s    |
| GrabCut  | 7.2.01.tiff     | 0.1811 | 0.3067 | 0.8850    | -      | -       |
| graphcut | boat.512.tiff   | 0.0465 | 0.0888 | 0.4278    | 10     | 0.6s    |
| GrabCut  | boat.512.tiff   | 0.1485 | 0.2586 | 0.8296    | -      | -       |
| graphcut | gray21.512.tiff | 0.0406 | 0.0780 | 0.0854    | 11     | 0.7s    |
| GrabCut  | gray21.512.tiff | 0.1319 | 0.2331 | 0.8617    | -      | -       |
| graphcut | house.tiff      | 0.0437 | 0.0837 | 0.2134    | 11     | 0.7s    |
| GrabCut  | house.tiff      | 0.1047 | 0.1895 | 0.7014    | -      | -       |
| graphcut | ruler.512.tiff  | 0.0565 | 0.1070 | 0.5342    | 8      | 0.6s    |
| GrabCut  | ruler.512.tiff  | 0.0431 | 0.0826 | 0.9568    | -      | -       |
_\* metrics are vs the auto-generated center-ellipse trimap, NOT human GT. Use as a proxy-only comparison between the selected solver and GrabCut._

Figures saved to `results/figures/misc_fgbg_*.png`

## 6. Solver comparison: Graph cut vs QAOA (latency and exactness)

### Five-image profiling (QAOA, 128x128, n_segments=12)
Mean per-image runtime share; ranges show the minimum-to-maximum across the five images.

| Stage | Mean runtime | Per-image range |
|---|---:|---:|
| QAOA solve loop | 58.8% | 28.4-73.7% |
| QHED edge-map generation | 23.5% | 10.4-34.9% |
| GMM fitting | 12.0% | 0.4-57.0% |
| SLIC | 0.9% | 0.3-2.6% |
| QUBO construction | 0.5% | 0.3-1.1% |
| Hybrid feature aggregation | 0.1% | 0.0-0.2% |

### 39-image quality comparison (USC-SIPI misc; proxy target, not human GT)
Current graph-cut values and historical QAOA values from commit ce70682; GrabCut values are from the current report. All rows summarize 39 images at 128x128 with n_segments=12.

The 39-image QAOA-versus-graph-cut comparison is not solver-only: ce70682 used Lab-color-only superpixel features, while the current code adds local-contrast and hybrid QHED/Sobel edge features. The SLIC settings and auto-trimap are unchanged.

| Method | Samples | Mean IoU* | Mean Dice* |
|---|---:|---:|---:|
| Graph cut (current report) | 39 | 0.058026 | 0.108121 |
| QAOA (historical report, ce70682) | 39 | 0.050344 | 0.094854 |
| GrabCut (current report) | 39 | 0.175885 | 0.291813 |

### Five-image latency comparison (USC-SIPI misc; 128x128, n_segments=12)
Images: 4.1.01.tiff, 5.1.12.tiff, boat.512.tiff, house.tiff, ruler.512.tiff. Quality values use the same auto-ellipse proxy target; latency is median full-pipeline time.

| Method | Mean IoU* | Mean Dice* | Median latency |
|---|---:|---:|---:|
| Graph cut | 0.0541 | 0.1024 | 2.4602 s |
| QAOA | 0.0499 | 0.0950 | 6.9990 s |
| GrabCut | 0.1048 | 0.1876 | 0.3681 s |

### Exact-match validation

| Validation set | Solver | Exact matches | Result |
|---|---|---:|---|
| 39 USC-SIPI misc image-derived QUBOs (up to 20 free nodes) | Graph cut vs brute force | 39/39 | All matched within scale-aware floating-point tolerance |
| 10 synthetic QUBOs (3-8 free nodes) | Graph cut vs brute force | 10/10 | Exact |
| 10 synthetic QUBOs (3-8 free nodes) | QAOA vs brute force | 5/10 | Mean energy gap on misses: 4.1088 |

*Proxy IoU/Dice caveat: see Section 5; these are not human ground truth.*

Real ground-truth comparison on GrabCut-50 is pending because the dataset is not currently available locally and the configured download source is unavailable.

