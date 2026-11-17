# T1: Qiskit NEQR resource table

## Settings

- NEQR: 8-bit grayscale; 2n position qubits + 8 intensity qubits; no ancillas.
- Sizes processed size-major: 8x8, then 16x16, then 32x32; images: 5.1.12.tiff, 4.2.03.tiff, boat.512.tiff.
- Image prep: `misc_gray(name, size, mode='crop').astype(uint8)`; centered crop, no resize interpolation.
- Transpiler: Qiskit `transpile`, basis `{cx,u}`, optimization level 1, `seed_transpiler=17`, `num_processes=1`, no coupling map / all-to-all connectivity.
- MCX synthesis: HLS `mcx=[('noaux_v24', {})]`, an ancilla-free decomposition; expected qubits remain 14/16/18.
- `total_gates` counts transpiled `u` plus `cx` operations; measurement, barrier, and state-save instructions are excluded. `CX` is the `cx` subset.
- Resource caps: 20 minutes and 10 GiB peak worker RSS. Capped cells are uncomputed, not estimated.
- PennyLane comparison: generated with C:\Users\HARSHITH\AppData\Local\Programs\Python\Python314\python.exe; comparison runs on 5.1.12.tiff 8x8
- No unreported warm-up was used for the transpile timings; each time is one transpile call.

## Per-image results

| Image | Size | Qubits | Qiskit gates (u+cx) | CX | Depth | Transpile (s) | Aer statevector (s) | Exact decode | PL max abs diff | Old PL gates/depth | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 5.1.12.tiff | 8x8 | 14 | 86739 | 41440 | 56833 | 1.0440764999948442 | 8.104995000059716 | True | 1.3009230494358742e-13 | 302/297 | ok |
| 4.2.03.tiff | 8x8 | 14 | 87913 | 42000 | 57601 | 0.9375293999910355 | 8.281503700069152 | True | — | 306/301 | ok |
| boat.512.tiff | 8x8 | 14 | 96996 | 46340 | 63553 | 1.2741694999858737 | 9.0722624999471 | True | — | 337/332 | ok |
| 5.1.12.tiff | 16x16 | 16 | 856707 | 401112 | 601683 | 7.2240352999651805 | 41.781287399935536 | True | — | 1246/1239 | ok |
| 4.2.03.tiff | 16x16 | 16 | 759827 | 355752 | 533643 | 6.627041200059466 | 36.70072439999785 | True | — | 1106/1099 | ok |
| boat.512.tiff | 16x16 | 16 | 884387 | 414072 | 621123 | 7.521484400029294 | 41.78624679998029 | True | — | 1286/1279 | ok |
| 5.1.12.tiff | 32x32 | 18 | 6166491 | 2888980 | 4567597 | 47.755981800029986 | 949.7592634999892 | True | — | 4991/4982 | ok |
| 4.2.03.tiff | 32x32 | 18 | 5359315 | 2510820 | 3969713 | 41.089019499951974 | 724.0045520000858 | True | — | 4339/4330 | ok |
| boat.512.tiff | 32x32 | 18 | 5879275 | 2754420 | 4354853 | 31.331081000040285 | 757.578506299993 | True | — | 4759/4750 | ok |

## Timing summary

- 8x8 transpile: mean 1.085s, 95th percentile 1.251s, n=3.
- 8x8 Aer statevector verification: mean 8.486s, 95th percentile 8.993s, n=3.
- 16x16 transpile: mean 7.124s, 95th percentile 7.492s, n=3.
- 16x16 Aer statevector verification: mean 40.089s, 95th percentile 41.786s, n=3.
- 32x32 transpile: mean 40.059s, 95th percentile 47.089s, n=3.
- 32x32 Aer statevector verification: mean 810.447s, 95th percentile 930.541s, n=3.

## Caps and failures

- Capped: none
- Errors: none
- Retry history: `5.1.12.tiff` 8x8 and `boat.512.tiff` 8x8 each failed once with a Windows `PermissionError` while updating the worker event file. After the event-file update was changed to an in-place write, both were rerun successfully. The CSV keeps only the latest row per image/size.

## Comparison with previous PennyLane resource counts

The existing report's PennyLane `qml.specs()` counts treat `MultiControlledX` as a native abstract operation. The Qiskit column counts the transpiled one- and two-qubit `u`/`cx` gates after explicit ancilla-free MCX synthesis, so a much larger count and depth are expected from decomposition; these are different counting levels, not a mismatch in pixel encoding. The Qiskit circuit retains the same total qubit count and its Aer statevector is checked by decoding intensities.

## Limits

No result is inferred for an uncompleted or capped transpile. See CSV status and error fields for each attempted item.
