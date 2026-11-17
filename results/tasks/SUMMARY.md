# T1-T3 Results Index

Commands below run from the repository root in PowerShell with `.venv-tasks`.

## T1: NEQR

- [Table 1](T1/Table1.md): Qiskit gate/depth resources. Reproduce with `python tasks/t1_neqr_resources.py`. Uses centered uint8 crops, `{cx,u}` transpilation, and ancilla-free `noaux_v24` MCX synthesis.
- [Table 1b](T1/Table1b.md): finite-shot classical fidelity, exact pixel recovery, coverage, and PSNR. Reproduce with `python tasks/t1_neqr_sampling_metrics.py`. For 8x8/16x16, 1k/10k/100k results are nested prefixes of a 100k-shot Aer run; 32x32 uses direct sampling from the verified ideal measurement distribution, not Aer. Fidelity is classical Bhattacharyya fidelity, not quantum-state fidelity. At high shots it follows approximately `1 - (N - 1)/(4 x shots)`.
- [Sampling timing](T1/sampling_timing.md): simulator timing. Reproduce with `python tasks/t1_neqr_sampling.py`; timing excludes circuit construction and transpilation.
- Not computed: noisy 8x8 for `4.2.03.tiff` and `boat.512.tiff`; only the canonical `5.1.12.tiff` noisy case was run. Noisy 16x16 and 32x32 for all three images were not run because their gate-count error-free probabilities are as low as `1.12e-14034` (the largest is `1.83e-1949`); no other noisy metrics are estimated.

## T2: QHED / BSDS500

- [Table 2](T2/Table2.md): QHED, Sobel, and Canny boundary metrics and timing. Reproduce with `python tasks/run_t2_qhed.py --limit 30`, then `python tasks/run_t2_metrics.py --limit 30`. Uses the first 30 sorted validation images, 256x256 center crops, 200 training images for baseline tuning, 99 thresholds, and the documented multi-annotator/empty-set conventions in [ASSUMPTIONS.md](../../tasks/ASSUMPTIONS.md).
- Not computed: validation images outside the first-30 evaluation scope; all requested rows within the 30-image scope completed.

## T3: SLIC-QAOA MaxCut

- [Table 3](T3/Table3.md): per-image selected-depth and fixed-depth pixel metrics. [Table 4](T3/Table4.md): representative image/depth parameters, transpiled depth, and runtime. Reproduce the report from the committed full-run CSV with `python tasks/report_t3_tables.py`. Reproduce the raw run with `python tasks/run_t3_qaoa_pilot.py --limit 39 --output results/tasks/T3/qaoa_maxcut_39.csv`.
- Uses 128x128 grayscale images, SLIC K=16, shared-boundary adjacency, the specified contrast weights, five COBYLA restarts, and a separate 4,096-shot final sample. Depth selection reconstructs ideal statevector expected cut from saved best parameters because the accepted CSV did not retain the optimizer's stochastic objective; optimization is not rerun. Pixel labels are aligned against exact MaxCut by choosing the orientation with higher accuracy.
- Not computed: none within the 39-image, four-depth run; all 156 rows succeeded.
