# Quantum Image Segmentation -- Mandatory Metrics vs SOTA

## 1. NEQR encoding fidelity + quantum resources (spatial-preservation gate)

_Lossless representation: exact reconstruction => PSNR=inf, SSIM=1, MSE=0 (every boundary/texture pixel preserved). Resource columns substantiate the 'quantum' claim (per Ruan 2021)._

| image         | size  | qubits | gates | depth | PSNR | SSIM   | MSE    | EPI    | sim_time |
|---------------|-------|--------|-------|-------|------|--------|--------|--------|----------|
| 5.1.12.tiff   | 8x8   | 14     | 302   | 297   | inf  | 1.0000 | 0.0000 | 1.0000 | 0.33s    |
| 5.1.12.tiff   | 16x16 | 16     | 1246  | 1239  | inf  | 1.0000 | 0.0000 | 1.0000 | 7.12s    |
| 5.1.12.tiff   | 32x32 | 18     | 4991  | 4982  | inf  | 1.0000 | 0.0000 | 1.0000 | 140.06s  |
| 4.2.03.tiff   | 8x8   | 14     | 306   | 301   | inf  | 1.0000 | 0.0000 | 1.0000 | 0.38s    |
| 4.2.03.tiff   | 16x16 | 16     | 1106  | 1099  | inf  | 1.0000 | 0.0000 | 1.0000 | 7.30s    |
| 4.2.03.tiff   | 32x32 | 18     | 4339  | 4330  | inf  | 1.0000 | 0.0000 | 1.0000 | 134.84s  |
| boat.512.tiff | 8x8   | 14     | 337   | 332   | inf  | 1.0000 | 0.0000 | 1.0000 | 0.51s    |
| boat.512.tiff | 16x16 | 16     | 1286  | 1279  | inf  | 1.0000 | 0.0000 | 1.0000 | 7.77s    |
| boat.512.tiff | 32x32 | 18     | 4759  | 4750  | inf  | 1.0000 | 0.0000 | 1.0000 | 148.49s  |

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

## 4. QAOA Fg/Bg Segmentation on GrabCut-50

_Run with `--grabcut` to download GrabCut-50 and evaluate QAOA fg/bg segmentation._

## 5. QAOA Fg/Bg on Misc Images (auto-trimap, no human GT)

Evaluated on 39 misc images at 128×128, n_segments=12.

| method  | image           | IoU*   | Dice*  | PixelAcc* | n_qubits | sim_time |
|---------|-----------------|--------|--------|-----------|----------|----------|
| QAOA    | 4.1.01.tiff     | 0.0405 | 0.0779 | 0.3183    | 12       | 6.3s     |
| GrabCut | 4.1.01.tiff     | 0.0840 | 0.1549 | 0.6829    | -        | -        |
| QAOA    | 4.1.02.tiff     | 0.0463 | 0.0886 | 0.5861    | 11       | 2.2s     |
| GrabCut | 4.1.02.tiff     | 0.0891 | 0.1637 | 0.6341    | -        | -        |
| QAOA    | 4.1.03.tiff     | 0.0447 | 0.0855 | 0.6532    | 7        | 1.2s     |
| GrabCut | 4.1.03.tiff     | 0.2177 | 0.3575 | 0.9678    | -        | -        |
| QAOA    | 4.1.04.tiff     | 0.0504 | 0.0959 | 0.5792    | 10       | 2.5s     |
| GrabCut | 4.1.04.tiff     | 0.1113 | 0.2004 | 0.7110    | -        | -        |
| QAOA    | 4.1.05.tiff     | 0.0451 | 0.0864 | 0.4482    | 9        | 1.5s     |
| GrabCut | 4.1.05.tiff     | 0.1116 | 0.2009 | 0.7140    | -        | -        |
| QAOA    | 4.1.06.tiff     | 0.0270 | 0.0526 | 0.3926    | 13       | 5.6s     |
| GrabCut | 4.1.06.tiff     | 0.0992 | 0.1805 | 0.7008    | -        | -        |
| QAOA    | 4.1.07.tiff     | 0.1904 | 0.3199 | 0.9410    | 6        | 1.2s     |
| GrabCut | 4.1.07.tiff     | 0.3151 | 0.4792 | 0.9399    | -        | -        |
| QAOA    | 4.1.08.tiff     | 0.0822 | 0.1519 | 0.8348    | 6        | 1.2s     |
| GrabCut | 4.1.08.tiff     | 0.1177 | 0.2106 | 0.7967    | -        | -        |
| QAOA    | 4.2.01.tiff     | 0.0550 | 0.1043 | 0.3490    | 6        | 1.1s     |
| GrabCut | 4.2.01.tiff     | 0.1272 | 0.2256 | 0.7444    | -        | -        |
| QAOA    | 4.2.03.tiff     | 0.0479 | 0.0914 | 0.4941    | 11       | 2.0s     |
| GrabCut | 4.2.03.tiff     | 0.1744 | 0.2970 | 0.8636    | -        | -        |
| QAOA    | 4.2.05.tiff     | 0.0545 | 0.1034 | 0.6761    | 12       | 2.9s     |
| GrabCut | 4.2.05.tiff     | 0.1919 | 0.3219 | 0.8765    | -        | -        |
| QAOA    | 4.2.06.tiff     | 0.0347 | 0.0671 | 0.3502    | 8        | 1.4s     |
| GrabCut | 4.2.06.tiff     | 0.1591 | 0.2745 | 0.8251    | -        | -        |
| QAOA    | 4.2.07.tiff     | 0.0563 | 0.1066 | 0.4892    | 10       | 1.7s     |
| GrabCut | 4.2.07.tiff     | 0.0709 | 0.1324 | 0.5155    | -        | -        |
| QAOA    | 5.1.09.tiff     | 0.0458 | 0.0877 | 0.4870    | 13       | 2.4s     |
| GrabCut | 5.1.09.tiff     | 0.1371 | 0.2411 | 0.7933    | -        | -        |
| QAOA    | 5.1.10.tiff     | 0.0445 | 0.0852 | 0.4536    | 15       | 5.3s     |
| GrabCut | 5.1.10.tiff     | 0.1596 | 0.2753 | 0.8115    | -        | -        |
| QAOA    | 5.1.11.tiff     | 0.0543 | 0.1030 | 0.4238    | 6        | 1.7s     |
| GrabCut | 5.1.11.tiff     | 0.1205 | 0.2151 | 0.8125    | -        | -        |
| QAOA    | 5.1.12.tiff     | 0.0577 | 0.1091 | 0.5187    | 10       | 2.6s     |
| GrabCut | 5.1.12.tiff     | 0.1383 | 0.2430 | 0.8068    | -        | -        |
| QAOA    | 5.1.13.tiff     | 0.0429 | 0.0823 | 0.5399    | 14       | 5.9s     |
| GrabCut | 5.1.13.tiff     | 0.1477 | 0.2574 | 0.8151    | -        | -        |
| QAOA    | 5.1.14.tiff     | 0.0414 | 0.0795 | 0.6484    | 11       | 4.0s     |
| GrabCut | 5.1.14.tiff     | 0.1388 | 0.2437 | 0.7930    | -        | -        |
| QAOA    | 5.2.08.tiff     | 0.0390 | 0.0751 | 0.5716    | 13       | 6.1s     |
| GrabCut | 5.2.08.tiff     | 0.1798 | 0.3048 | 0.8484    | -        | -        |
| QAOA    | 5.2.09.tiff     | 0.0371 | 0.0716 | 0.3437    | 14       | 4.2s     |
| GrabCut | 5.2.09.tiff     | 0.2680 | 0.4227 | 0.9249    | -        | -        |
| QAOA    | 5.2.10.tiff     | 0.0497 | 0.0947 | 0.5111    | 13       | 4.5s     |
| GrabCut | 5.2.10.tiff     | 0.2413 | 0.3888 | 0.8976    | -        | -        |
| QAOA    | 5.3.01.tiff     | 0.0421 | 0.0807 | 0.3498    | 12       | 5.7s     |
| GrabCut | 5.3.01.tiff     | 0.0846 | 0.1560 | 0.5970    | -        | -        |
| QAOA    | 5.3.02.tiff     | 0.0375 | 0.0723 | 0.4209    | 14       | 7.2s     |
| GrabCut | 5.3.02.tiff     | 0.1313 | 0.2320 | 0.7655    | -        | -        |
| QAOA    | 7.1.01.tiff     | 0.0476 | 0.0909 | 0.3926    | 12       | 2.3s     |
| GrabCut | 7.1.01.tiff     | 0.2735 | 0.4295 | 0.9186    | -        | -        |
| QAOA    | 7.1.02.tiff     | 0.0488 | 0.0931 | 0.4290    | 9        | 2.0s     |
| GrabCut | 7.1.02.tiff     | 0.2806 | 0.4382 | 0.9634    | -        | -        |
| QAOA    | 7.1.03.tiff     | 0.0400 | 0.0768 | 0.2891    | 10       | 2.5s     |
| GrabCut | 7.1.03.tiff     | 0.1781 | 0.3024 | 0.8713    | -        | -        |
| QAOA    | 7.1.04.tiff     | 0.0371 | 0.0715 | 0.7710    | 11       | 2.1s     |
| GrabCut | 7.1.04.tiff     | 0.2275 | 0.3707 | 0.8943    | -        | -        |
| QAOA    | 7.1.05.tiff     | 0.0324 | 0.0628 | 0.2522    | 12       | 2.3s     |
| GrabCut | 7.1.05.tiff     | 0.1797 | 0.3046 | 0.8377    | -        | -        |
| QAOA    | 7.1.06.tiff     | 0.0341 | 0.0659 | 0.4627    | 15       | 6.9s     |
| GrabCut | 7.1.06.tiff     | 0.2253 | 0.3677 | 0.8974    | -        | -        |
| QAOA    | 7.1.07.tiff     | 0.0459 | 0.0878 | 0.5524    | 12       | 3.1s     |
| GrabCut | 7.1.07.tiff     | 0.3688 | 0.5388 | 0.9523    | -        | -        |
| QAOA    | 7.1.08.tiff     | 0.0547 | 0.1037 | 0.6165    | 10       | 2.6s     |
| GrabCut | 7.1.08.tiff     | 0.3069 | 0.4697 | 0.9546    | -        | -        |
| QAOA    | 7.1.09.tiff     | 0.0494 | 0.0942 | 0.6556    | 9        | 2.0s     |
| GrabCut | 7.1.09.tiff     | 0.3537 | 0.5226 | 0.9547    | -        | -        |
| QAOA    | 7.1.10.tiff     | 0.0432 | 0.0828 | 0.3329    | 13       | 3.8s     |
| GrabCut | 7.1.10.tiff     | 0.2399 | 0.3870 | 0.8923    | -        | -        |
| QAOA    | 7.2.01.tiff     | 0.0647 | 0.1215 | 0.6323    | 10       | 1.9s     |
| GrabCut | 7.2.01.tiff     | 0.1811 | 0.3067 | 0.8850    | -        | -        |
| QAOA    | boat.512.tiff   | 0.0554 | 0.1049 | 0.5396    | 10       | 2.5s     |
| GrabCut | boat.512.tiff   | 0.1485 | 0.2586 | 0.8296    | -        | -        |
| QAOA    | gray21.512.tiff | 0.0312 | 0.0604 | 0.3867    | 11       | 1.3s     |
| GrabCut | gray21.512.tiff | 0.1319 | 0.2331 | 0.8617    | -        | -        |
| QAOA    | house.tiff      | 0.0413 | 0.0793 | 0.3279    | 11       | 1.5s     |
| GrabCut | house.tiff      | 0.1047 | 0.1895 | 0.7014    | -        | -        |
| QAOA    | ruler.512.tiff  | 0.0546 | 0.1036 | 0.5329    | 8        | 1.7s     |
| GrabCut | ruler.512.tiff  | 0.0431 | 0.0826 | 0.9568    | -        | -        |
_\* metrics are vs the auto-generated center-ellipse trimap, NOT human GT. Use as a relative comparison between QAOA and GrabCut only._

Figures saved to `results/figures/misc_fgbg_*.png`

