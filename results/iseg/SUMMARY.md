# iSeg Phase 1: Graph-Cut Robustness and Speed

**Evaluation label:** no ground truth, ellipse trimap: robustness and speed only. No ground truth was available; no IoU or Dice is reported.

## Settings

- Input images: 151 local files from `data/iseg/images/`
- Input size: 128x128 using OpenCV resize, matching `segment_misc.py`
- Superpixels: `n_segments=12`; solver: `graphcut`; `use_qhed=True`; `p=1`
- Trimap: auto-generated center ellipse and border, values 255=foreground seed, 0=background seed, 128=unknown
- Warm-up: one unreported call on `106024`
- OpenCV threads: 1
- Native BLAS/OpenMP thread limit: 1 via threadpoolctl
- Threadpool: openblas / libscipy_openblas: 1 thread(s)
- Command: `python scripts/run_iseg.py`
- Resumability: completed image IDs in `per_image.csv` are skipped on subsequent runs.

## Run Summary

- Images found: 151
- Successful: 151
- Failed: 0
- Collapsed free regions: 47 / 151 successful images
- Median total latency: 1.0249 s
- 90th percentile total latency: 2.3302 s
- `max|h_eff| / max J` distribution: min/median/p90/max: 0 / 6.46521e+07 / 4.44199e+08 / 9.6179e+08

## Failures

- None

## Outputs and Limitations

- Per-image metrics and status: `results/iseg/per_image.csv`
- Predicted masks (not committed): `results/iseg/masks/`
- Sample overlays for the first 12 sorted images (not committed): `results/iseg/samples/`
- This run measures execution, timing, term scales, and mask collapse only. With no iSeg ground-truth masks or user strokes present, it cannot establish segmentation accuracy.
