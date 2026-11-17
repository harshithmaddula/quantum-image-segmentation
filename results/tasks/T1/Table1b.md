# Table 1b: Finite-Shot NEQR Sampling Quality

Each pixel is decoded as the most frequent sampled intensity at its position; unsampled positions are set to zero and count as unrecovered. Coverage is the fraction of positions sampled at least once. Fidelity is classical Bhattacharyya fidelity `(sum_x sqrt(p_x q_x))^2` between the ideal and empirical computational-basis measurement distributions; it is not quantum-state fidelity. PSNR uses 8-bit intensities and the full image, including unsampled positions.
For 8x8 and 16x16, each shot count is a nested prefix of one 100,000-shot Aer sample per image/size. For 32x32, samples are drawn directly from the exact ideal measurement distribution; Phase 1 verified exact NEQR decodes for all three 32x32 circuits. The exact-distribution sampling is not an Aer circuit run.
For a noiseless uniform position register with `N` positions, the high-shot classical fidelity follows the approximation `1 - (N - 1) / (4 x shots)`; at low shot counts, finite occupancy and unsampled positions cause larger deviations.

| Image | Size | Shots | Classical fidelity | Exact pixels / all | Coverage | PSNR (dB) | Sampling method |
|---|---:|---:|---:|---:|---:|---:|---|
| 5.1.12.tiff | 8x8 | 1,000 | 0.984267 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| 5.1.12.tiff | 8x8 | 10,000 | 0.998023 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| 5.1.12.tiff | 8x8 | 100,000 | 0.999841 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| 5.1.12.tiff | 16x16 | 1,000 | 0.925647 | 0.988281 | 0.988281 | 20.156938 | Aer; nested prefix of one 100k-shot run |
| 5.1.12.tiff | 16x16 | 10,000 | 0.994146 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| 5.1.12.tiff | 16x16 | 100,000 | 0.999421 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| 5.1.12.tiff | 32x32 | 1,000 | 0.590743 | 0.624023 | 0.624023 | 5.559445 | direct samples from verified ideal measurement distribution |
| 5.1.12.tiff | 32x32 | 10,000 | 0.974003 | 1.000000 | 1.000000 | inf | direct samples from verified ideal measurement distribution |
| 5.1.12.tiff | 32x32 | 100,000 | 0.997430 | 1.000000 | 1.000000 | inf | direct samples from verified ideal measurement distribution |
| 4.2.03.tiff | 8x8 | 1,000 | 0.987317 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| 4.2.03.tiff | 8x8 | 10,000 | 0.998493 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| 4.2.03.tiff | 8x8 | 100,000 | 0.999828 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| 4.2.03.tiff | 16x16 | 1,000 | 0.925444 | 0.984375 | 0.984375 | 22.481509 | Aer; nested prefix of one 100k-shot run |
| 4.2.03.tiff | 16x16 | 10,000 | 0.993212 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| 4.2.03.tiff | 16x16 | 100,000 | 0.999416 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| 4.2.03.tiff | 32x32 | 1,000 | 0.604375 | 0.637695 | 0.637695 | 8.991531 | direct samples from verified ideal measurement distribution |
| 4.2.03.tiff | 32x32 | 10,000 | 0.974763 | 1.000000 | 1.000000 | inf | direct samples from verified ideal measurement distribution |
| 4.2.03.tiff | 32x32 | 100,000 | 0.997405 | 1.000000 | 1.000000 | inf | direct samples from verified ideal measurement distribution |
| boat.512.tiff | 8x8 | 1,000 | 0.991217 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| boat.512.tiff | 8x8 | 10,000 | 0.998078 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| boat.512.tiff | 8x8 | 100,000 | 0.999881 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| boat.512.tiff | 16x16 | 1,000 | 0.918461 | 0.980469 | 0.980469 | 19.548816 | Aer; nested prefix of one 100k-shot run |
| boat.512.tiff | 16x16 | 10,000 | 0.993951 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| boat.512.tiff | 16x16 | 100,000 | 0.999312 | 1.000000 | 1.000000 | inf | Aer; nested prefix of one 100k-shot run |
| boat.512.tiff | 32x32 | 1,000 | 0.597682 | 0.630859 | 0.630859 | 6.796550 | direct samples from verified ideal measurement distribution |
| boat.512.tiff | 32x32 | 10,000 | 0.975866 | 1.000000 | 1.000000 | inf | direct samples from verified ideal measurement distribution |
| boat.512.tiff | 32x32 | 100,000 | 0.997305 | 1.000000 | 1.000000 | inf | direct samples from verified ideal measurement distribution |

## Noisy 8x8 Run

Circuit: `5.1.12.tiff`, 8x8 (the canonical T1 timing image). The 8-shot single-run probe took 11.306396 s (1.413299 s/shot); the capped sampling run used 191 shots in 1201.02 s, including a 1.02 s overrun while the final batch completed, or about 6.29 s/shot. The target budget was 1,200 s. Depolarizing error was applied after each transpiled `u` (p1=0.001) and `cx` (p2=0.01). The gate-count estimate of error-free probability is 2.75e-201.

| Image | Size | Shots | Classical fidelity | Exact pixels / all | Coverage | PSNR (dB) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| 5.1.12.tiff | 8x8 | 191 | 0.000082 | 0.000000 | 0.859375 | 3.506075 | computed |

The observed noisy coverage (0.859375) is below the 0.950606 expected from 191 independent uniform position draws. The cause was not diagnosed; depolarizing circuit noise may bias position probabilities, and finite sampling variation is also possible. No uniformity test was performed. The 8-shot probe's higher per-shot time is consistent with fixed job/startup overhead being amortized over longer runs and Aer processing larger shot batches more efficiently, but these runs do not isolate the cause or establish a parallelization speedup.

Noisy 16x16 and 32x32 were not computed. Error-free probability estimates below use `(1-p1)^N_u (1-p2)^N_cx` from the saved transpiled u/CX counts; no other noisy metrics are estimated.

| Image | Size | Noisy status | Estimated error-free probability |
|---|---:|---|---:|
| 5.1.12.tiff | 16x16 | not computed | 1.83e-1949 |
| 5.1.12.tiff | 32x32 | not computed | 1.12e-14034 |
| 4.2.03.tiff | 16x16 | not computed | 4.33e-1729 |
| 4.2.03.tiff | 32x32 | not computed | 1.13e-12197 |
| boat.512.tiff | 16x16 | not computed | 1.99e-2012 |
| boat.512.tiff | 32x32 | not computed | 5.09e-13381 |

Noiseless Aer sampling timings remain in [`sampling_timing.md`](sampling_timing.md). The single 8-shot noisy probe timing is 11.306396 s; the capped noisy run used 191 shots.
