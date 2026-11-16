# T3 Full-Run Summary

The accepted run contains 39 sorted USC-SIPI TIFF images and 156 successful image/depth rows. Each result uses K=16 SLIC superpixels, five COBYLA restarts per depth, 4,096 optimization shots per evaluation, and a separate 4,096-shot final sample.

Random-assignment expected cut is one half of the total edge weight; its ratio is averaged per image before aggregation. Final expected-cut values and ratios use the independent final sample.

| p | Mean final expected cut +/- SD | Mean expected/exact | Mean random/exact | Mean mode/exact | Mean wall (s) | P95 wall (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.8409 +/- 2.7691 | 0.7552 | 0.6123 | 0.9319 | 13.09 | 17.73 |
| 2 | 11.5218 +/- 2.5555 | 0.6779 | 0.6123 | 0.7415 | 18.63 | 21.47 |
| 3 | 12.6171 +/- 2.6347 | 0.7441 | 0.6123 | 0.9279 | 25.43 | 30.61 |
| 4 | 12.4087 +/- 2.7992 | 0.7311 | 0.6123 | 0.8579 | 27.54 | 33.57 |

At p=2, mean final expected/exact scoring (0.6779) is below p=1 (0.7552); this indicates the optimizer failed to match the p=1 result on this run, not that p=2 is intrinsically inferior.

See `Table3.md` for selected-depth and fixed-depth pixel metrics and `Table4.md` for representative parameters, circuit depths, and runtimes.
