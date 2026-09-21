# Quantum Image Segmentation -- Mandatory Metrics vs SOTA

## 1. NEQR encoding fidelity + quantum resources (spatial-preservation gate)

_Lossless representation: exact reconstruction => PSNR=inf, SSIM=1, MSE=0 (every boundary/texture pixel preserved). Resource columns substantiate the 'quantum' claim (per Ruan 2021)._

| image         | size  | qubits | gates | depth | PSNR | SSIM   | MSE    | EPI    | sim_time |
|---------------|-------|--------|-------|-------|------|--------|--------|--------|----------|
| 5.1.12.tiff   | 8x8   | 14     | 302   | 297   | inf  | 1.0000 | 0.0000 | 1.0000 | 0.18s    |
| 5.1.12.tiff   | 16x16 | 16     | 1246  | 1239  | inf  | 1.0000 | 0.0000 | 1.0000 | 4.54s    |
| 5.1.12.tiff   | 32x32 | 18     | 4991  | 4982  | inf  | 1.0000 | 0.0000 | 1.0000 | 112.39s  |
| 4.2.03.tiff   | 8x8   | 14     | 306   | 301   | inf  | 1.0000 | 0.0000 | 1.0000 | 0.34s    |
| 4.2.03.tiff   | 16x16 | 16     | 1106  | 1099  | inf  | 1.0000 | 0.0000 | 1.0000 | 5.05s    |
| 4.2.03.tiff   | 32x32 | 18     | 4339  | 4330  | inf  | 1.0000 | 0.0000 | 1.0000 | 91.77s   |
| boat.512.tiff | 8x8   | 14     | 337   | 332   | inf  | 1.0000 | 0.0000 | 1.0000 | 0.46s    |
| boat.512.tiff | 16x16 | 16     | 1286  | 1279  | inf  | 1.0000 | 0.0000 | 1.0000 | 6.11s    |
| boat.512.tiff | 32x32 | 18     | 4759  | 4750  | inf  | 1.0000 | 0.0000 | 1.0000 | 117.93s  |

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
| QAOA    | 4.1.01.tiff     | 0.0431 | 0.0826 | 0.5906    | 12       | 30.3s    |
| GrabCut | 4.1.01.tiff     | 0.0840 | 0.1549 | 0.6829    | -        | -        |
| QAOA    | 4.1.02.tiff     | 0.0470 | 0.0898 | 0.2633    | 11       | 16.0s    |
| GrabCut | 4.1.02.tiff     | 0.0891 | 0.1637 | 0.6341    | -        | -        |
| QAOA    | 4.1.03.tiff     | 0.0395 | 0.0760 | 0.0563    | 7        | 6.1s     |
| GrabCut | 4.1.03.tiff     | 0.2177 | 0.3575 | 0.9678    | -        | -        |
| QAOA    | 4.1.04.tiff     | 0.0414 | 0.0795 | 0.4501    | 10       | 10.3s    |
| GrabCut | 4.1.04.tiff     | 0.1113 | 0.2004 | 0.7110    | -        | -        |
| QAOA    | 4.1.05.tiff     | 0.0638 | 0.1199 | 0.4570    | 9        | 8.1s     |
| GrabCut | 4.1.05.tiff     | 0.1116 | 0.2009 | 0.7140    | -        | -        |
| QAOA    | 4.1.06.tiff     | 0.0440 | 0.0843 | 0.3605    | 13       | 16.4s    |
| GrabCut | 4.1.06.tiff     | 0.0992 | 0.1805 | 0.7008    | -        | -        |
| QAOA    | 4.1.07.tiff     | 0.1904 | 0.3199 | 0.9410    | 6        | 5.8s     |
| GrabCut | 4.1.07.tiff     | 0.3151 | 0.4792 | 0.9399    | -        | -        |
| QAOA    | 4.1.08.tiff     | 0.0650 | 0.1221 | 0.6008    | 6        | 5.4s     |
| GrabCut | 4.1.08.tiff     | 0.1177 | 0.2106 | 0.7967    | -        | -        |
| QAOA    | 4.2.01.tiff     | 0.0584 | 0.1104 | 0.5149    | 6        | 3.9s     |
| GrabCut | 4.2.01.tiff     | 0.1272 | 0.2256 | 0.7444    | -        | -        |
| QAOA    | 4.2.03.tiff     | 0.0793 | 0.1469 | 0.7965    | 11       | 14.1s    |
| GrabCut | 4.2.03.tiff     | 0.1744 | 0.2970 | 0.8636    | -        | -        |
| QAOA    | 4.2.05.tiff     | 0.0577 | 0.1090 | 0.5038    | 12       | 21.3s    |
| GrabCut | 4.2.05.tiff     | 0.1919 | 0.3219 | 0.8765    | -        | -        |
| QAOA    | 4.2.06.tiff     | 0.0347 | 0.0671 | 0.3502    | 8        | 4.1s     |
| GrabCut | 4.2.06.tiff     | 0.1591 | 0.2745 | 0.8251    | -        | -        |
| QAOA    | 4.2.07.tiff     | 0.0620 | 0.1168 | 0.5683    | 10       | 8.5s     |
| GrabCut | 4.2.07.tiff     | 0.0709 | 0.1324 | 0.5155    | -        | -        |
| QAOA    | 5.1.09.tiff     | 0.0393 | 0.0756 | 0.7841    | 13       | 16.7s    |
| GrabCut | 5.1.09.tiff     | 0.1371 | 0.2411 | 0.7933    | -        | -        |
| QAOA    | 5.1.10.tiff     | 0.0436 | 0.0835 | 0.3907    | 15       | 38.1s    |
| GrabCut | 5.1.10.tiff     | 0.1596 | 0.2753 | 0.8115    | -        | -        |
| QAOA    | 5.1.11.tiff     | 0.0661 | 0.1239 | 0.6808    | 6        | 3.0s     |
| GrabCut | 5.1.11.tiff     | 0.1205 | 0.2151 | 0.8125    | -        | -        |
| QAOA    | 5.1.12.tiff     | 0.0346 | 0.0669 | 0.4356    | 10       | 9.8s     |
| GrabCut | 5.1.12.tiff     | 0.1383 | 0.2430 | 0.8068    | -        | -        |
| QAOA    | 5.1.13.tiff     | 0.0383 | 0.0738 | 0.4270    | 14       | 48.8s    |
| GrabCut | 5.1.13.tiff     | 0.1477 | 0.2574 | 0.8151    | -        | -        |
| QAOA    | 5.1.14.tiff     | 0.0408 | 0.0785 | 0.3602    | 11       | 33.1s    |
| GrabCut | 5.1.14.tiff     | 0.1388 | 0.2437 | 0.7930    | -        | -        |
| QAOA    | 5.2.08.tiff     | 0.0288 | 0.0560 | 0.6236    | 13       | 34.3s    |
| GrabCut | 5.2.08.tiff     | 0.1798 | 0.3048 | 0.8484    | -        | -        |
| QAOA    | 5.2.09.tiff     | 0.0422 | 0.0811 | 0.4382    | 14       | 48.6s    |
| GrabCut | 5.2.09.tiff     | 0.2680 | 0.4227 | 0.9249    | -        | -        |
| QAOA    | 5.2.10.tiff     | 0.0415 | 0.0797 | 0.2869    | 13       | 36.1s    |
| GrabCut | 5.2.10.tiff     | 0.2413 | 0.3888 | 0.8976    | -        | -        |
| QAOA    | 5.3.01.tiff     | 0.0456 | 0.0872 | 0.4238    | 12       | 31.0s    |
| GrabCut | 5.3.01.tiff     | 0.0846 | 0.1560 | 0.5970    | -        | -        |
| QAOA    | 5.3.02.tiff     | 0.0402 | 0.0773 | 0.3027    | 14       | 60.7s    |
| GrabCut | 5.3.02.tiff     | 0.1313 | 0.2320 | 0.7655    | -        | -        |
| QAOA    | 7.1.01.tiff     | 0.0350 | 0.0676 | 0.1736    | 12       | 25.8s    |
| GrabCut | 7.1.01.tiff     | 0.2735 | 0.4295 | 0.9186    | -        | -        |
| QAOA    | 7.1.02.tiff     | 0.0443 | 0.0849 | 0.5522    | 9        | 14.8s    |
| GrabCut | 7.1.02.tiff     | 0.2806 | 0.4382 | 0.9634    | -        | -        |
| QAOA    | 7.1.03.tiff     | 0.0491 | 0.0936 | 0.6734    | 10       | 17.0s    |
| GrabCut | 7.1.03.tiff     | 0.1781 | 0.3024 | 0.8713    | -        | -        |
| QAOA    | 7.1.04.tiff     | 0.0393 | 0.0757 | 0.9556    | 11       | 28.9s    |
| GrabCut | 7.1.04.tiff     | 0.2275 | 0.3707 | 0.8943    | -        | -        |
| QAOA    | 7.1.05.tiff     | 0.0349 | 0.0674 | 0.2491    | 12       | 39.6s    |
| GrabCut | 7.1.05.tiff     | 0.1797 | 0.3046 | 0.8377    | -        | -        |
| QAOA    | 7.1.06.tiff     | 0.0348 | 0.0672 | 0.5967    | 15       | 108.4s   |
| GrabCut | 7.1.06.tiff     | 0.2253 | 0.3677 | 0.8974    | -        | -        |
| QAOA    | 7.1.07.tiff     | 0.0451 | 0.0863 | 0.6292    | 12       | 43.0s    |
| GrabCut | 7.1.07.tiff     | 0.3688 | 0.5388 | 0.9523    | -        | -        |
| QAOA    | 7.1.08.tiff     | 0.0490 | 0.0935 | 0.5067    | 10       | 16.4s    |
| GrabCut | 7.1.08.tiff     | 0.3069 | 0.4697 | 0.9546    | -        | -        |
| QAOA    | 7.1.09.tiff     | 0.0420 | 0.0806 | 0.5557    | 9        | 18.8s    |
| GrabCut | 7.1.09.tiff     | 0.3537 | 0.5226 | 0.9547    | -        | -        |
| QAOA    | 7.1.10.tiff     | 0.0355 | 0.0686 | 0.4190    | 13       | 34.8s    |
| GrabCut | 7.1.10.tiff     | 0.2399 | 0.3870 | 0.8923    | -        | -        |
| QAOA    | 7.2.01.tiff     | 0.0489 | 0.0932 | 0.4196    | 10       | 18.2s    |
| GrabCut | 7.2.01.tiff     | 0.1811 | 0.3067 | 0.8850    | -        | -        |
| QAOA    | boat.512.tiff   | 0.0469 | 0.0896 | 0.4174    | 10       | 21.4s    |
| GrabCut | boat.512.tiff   | 0.1485 | 0.2586 | 0.8296    | -        | -        |
| QAOA    | gray21.512.tiff | 0.0748 | 0.1392 | 0.7240    | 11       | 16.3s    |
| GrabCut | gray21.512.tiff | 0.1319 | 0.2331 | 0.8617    | -        | -        |
| QAOA    | house.tiff      | 0.0430 | 0.0825 | 0.4688    | 11       | 25.4s    |
| GrabCut | house.tiff      | 0.1047 | 0.1895 | 0.7014    | -        | -        |
| QAOA    | ruler.512.tiff  | 0.0535 | 0.1016 | 0.4540    | 8        | 3.4s     |
| GrabCut | ruler.512.tiff  | 0.0431 | 0.0826 | 0.9568    | -        | -        |
_\* metrics are vs the auto-generated center-ellipse trimap, NOT human GT. Use as a relative comparison between QAOA and GrabCut only._

Figures saved to `results/figures/misc_fgbg_*.png`

