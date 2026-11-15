# Table 1: NEQR circuit resources

| Image | Size | Qubits | Gates (u+cx) | Depth |
|---|---:|---:|---:|---:|
| 5.1.12.tiff | 8x8 | 14 | 86739 | 56833 |
| 4.2.03.tiff | 8x8 | 14 | 87913 | 57601 |
| boat.512.tiff | 8x8 | 14 | 96996 | 63553 |
| 5.1.12.tiff | 16x16 | 16 | 856707 | 601683 |
| 4.2.03.tiff | 16x16 | 16 | 759827 | 533643 |
| boat.512.tiff | 16x16 | 16 | 884387 | 621123 |
| 5.1.12.tiff | 32x32 | 18 | 6166491 | 4567597 |
| 4.2.03.tiff | 32x32 | 18 | 5359315 | 3969713 |
| boat.512.tiff | 32x32 | 18 | 5879275 | 4354853 |

Gate counts are from Qiskit circuits transpiled to `{cx,u}`; `Gates` is `u+cx` and excludes measurements, barriers, and state-save instructions. Depth is the transpiled circuit depth. Multi-controlled X gates use the ancilla-free `noaux_v24` synthesis method.
