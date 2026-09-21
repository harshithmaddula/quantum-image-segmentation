"""Quantum image segmentation — edge detection + foreground/background separation.

Modules
-------
neqr       : NEQR representation (lossless, position-exact) encode/decode in PennyLane.
qsobel     : Quantum edge detection -- QHED (scalable) + faithful QSobel (small patches).
superpixel : SLIC superpixels + QUBO construction for fg/bg graph cut.
qaoa_seg   : QAOA-based fg/bg segmentation (Stage 4 pipeline entry point).
baselines  : Classical edge + GrabCut baselines (OpenCV).
metrics    : Boundary + fidelity + segmentation metric battery.
data       : misc loader + BSDS500 + GrabCut-50 fetch/load.

Run everything with the working interpreter: ``python3`` (NumPy on ``python`` 3.14 segfaults).
"""

__all__ = ["neqr", "qsobel", "superpixel", "qaoa_seg", "baselines", "metrics", "data"]
