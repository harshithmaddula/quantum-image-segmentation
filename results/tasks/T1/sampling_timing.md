# T1 Sampling Timing

Circuit: `5.1.12.tiff`, centered 8x8 crop, 14-qubit NEQR circuit transpiled to `{cx,u}` with the T1 settings. CPU: Intel Core i5-12450H. Timings are simulator execution only and exclude circuit construction and transpilation; no warm-up was used.

| Run | Noise | Shots | Simulation wall time (s) |
|---|---|---:|---:|
| Noisy | depolarizing p1=0.001, p2=0.01 | 1 | 5.856037 |
| Noiseless | none | 1,000 | 6.869829 |
| Noiseless | none | 10,000 | 6.804134 |
| Noiseless | none | 100,000 | 7.408407 |