"""NEQR -- Novel Enhanced Quantum Representation of digital images.

NEQR (Zhang et al., 2013) stores a 2^n x 2^n image with q-bit gray levels as

    |I> = (1 / 2^n) * sum_{y,x} |f(y,x)> (X) |y> |x>

where |y>|x| is a 2n-qubit *position* register (basis-encoded coordinates) and
|f(y,x)> is a q-qubit *colour* register holding the pixel's intensity as a basis
state.  Because both position and intensity are basis-encoded (not amplitude- or
angle-encoded), NEQR is **lossless**: encode -> measure -> decode returns the image
exactly.  That exactness is precisely why it is the right representation for
*spatial preservation* -- no boundary, texture, or fine detail is blurred by the
encoding, and it is the representation QSobel is built on.

Circuit
-------
1. Hadamard on every position qubit  -> uniform superposition over all 2^(2n) pixels.
2. For each pixel (y, x) and each colour bit i that is 1, a multi-controlled X
   (controls = the 2n position qubits fixed to the pattern of (y, x)) flips colour
   qubit i.  ``control_values`` encodes the coordinate pattern, so no X-sandwiching
   is needed.

Gate count is O(2^(2n) * q) multi-controlled X gates, so faithful full-image NEQR is
only practical on small tiles/thumbnails on a statevector simulator.  The losslessness
result is independent of size, so the fidelity gate is demonstrated on small crops and
the conclusion (PSNR -> infinity, SSIM = 1) generalises.

Wire layout (total 2n + q):
    position wires : 0 .. 2n-1   (first n = row y MSB-first, next n = col x MSB-first)
    colour wires   : 2n .. 2n+q-1  (MSB-first; wire 2n is the most-significant colour bit)
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


# --------------------------------------------------------------------------- #
# bit helpers
# --------------------------------------------------------------------------- #
def _bits(value: int, nbits: int) -> list[int]:
    """Integer -> list of bits, MSB first."""
    return [(value >> (nbits - 1 - i)) & 1 for i in range(nbits)]


def _from_bits(bits) -> int:
    """List of bits (MSB first) -> integer."""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def image_dims(side: int) -> int:
    """Return n such that side == 2^n, else raise."""
    n = int(round(np.log2(side)))
    if 2 ** n != side:
        raise ValueError(f"image side {side} is not a power of two")
    return n


# --------------------------------------------------------------------------- #
# encoding ops (call inside a QNode)
# --------------------------------------------------------------------------- #
def neqr_ops(img: np.ndarray, q: int = 8) -> tuple[int, int]:
    """Apply the NEQR preparation gates for ``img`` on the default wire layout.

    Parameters
    ----------
    img : (2^n, 2^n) uint array of q-bit intensities.
    q   : number of colour (intensity) bits.

    Returns
    -------
    (n, q) so callers know the register sizes.
    """
    h, w = img.shape
    if h != w:
        raise ValueError("NEQR demo expects a square image")
    n = image_dims(h)
    pos_wires = list(range(2 * n))
    col_wires = list(range(2 * n, 2 * n + q))

    # 1) uniform superposition over all pixel coordinates
    for wexpr in pos_wires:
        qml.Hadamard(wires=wexpr)

    # 2) basis-encode each pixel's intensity, controlled on its coordinate
    maxv = (1 << q) - 1
    for y in range(h):
        ybits = _bits(y, n)
        for x in range(w):
            v = int(img[y, x])
            if v < 0 or v > maxv:
                raise ValueError(f"pixel {v} out of range for q={q}")
            if v == 0:
                continue  # all colour bits already |0>
            ctrl_vals = ybits + _bits(x, n)
            cbits = _bits(v, q)
            for i, b in enumerate(cbits):
                if b:
                    qml.MultiControlledX(
                        wires=pos_wires + [col_wires[i]],
                        control_values=ctrl_vals,
                    )
    return n, q


def build_state_qnode(img: np.ndarray, q: int = 8):
    """Return (qnode, n, q); the qnode prepares NEQR and returns the statevector."""
    n = image_dims(img.shape[0])
    try:
        dev = qml.device("lightning.qubit", wires=2 * n + q)
    except ImportError:
        # Fallback for Python 3.14+ where lightning.qubit binaries may not be available
        dev = qml.device("default.qubit", wires=2 * n + q)

    @qml.qnode(dev)
    def circuit():
        neqr_ops(img, q=q)
        return qml.state()

    return circuit, n, q


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def decode_state(state: np.ndarray, n: int, q: int, tol: float = 1e-9) -> np.ndarray:
    """Reconstruct the 2^n x 2^n image from a NEQR statevector.

    PennyLane orders the computational-basis index MSB-first over the wire list,
    i.e. wire 0 is the most-significant bit of the basis index.  With our layout the
    index decomposes as  [ y (n) | x (n) | colour (q) ].
    """
    side = 2 ** n
    img = np.zeros((side, side), dtype=np.uint16)
    probs = np.abs(state) ** 2
    nz = np.nonzero(probs > tol)[0]
    total_bits = 2 * n + q
    for k in nz:
        bits = _bits(int(k), total_bits)
        y = _from_bits(bits[0:n])
        x = _from_bits(bits[n:2 * n])
        c = _from_bits(bits[2 * n:2 * n + q])
        img[y, x] = c
    return img


def roundtrip(img: np.ndarray, q: int = 8) -> np.ndarray:
    """NEQR encode -> simulate (exact statevector) -> decode. Should equal ``img``."""
    qnode, n, q = build_state_qnode(img, q=q)
    state = qnode()
    return decode_state(np.asarray(state), n, q).astype(img.dtype)


# --------------------------------------------------------------------------- #
# resource accounting (a "quantum" claim needs this -- per Ruan 2021)
# --------------------------------------------------------------------------- #
def resource_report(img: np.ndarray, q: int = 8) -> dict:
    """Qubit count, gate count, and circuit depth for the NEQR preparation."""
    n = image_dims(img.shape[0])

    def tape():
        neqr_ops(img, q=q)

    try:
        dev_specs = qml.device("lightning.qubit", wires=2 * n + q)
    except ImportError:
        dev_specs = qml.device("default.qubit", wires=2 * n + q)
    specs = qml.specs(qml.QNode(lambda: (neqr_ops(img, q=q), qml.state())[1],
                                dev_specs))()
    # qml.specs() returns a dict in older PennyLane and CircuitSpecs in 0.36+
    try:
        res = specs["resources"]
    except (TypeError, KeyError):
        res = getattr(specs, "resources", None)
    out = {
        "image_side": img.shape[0],
        "n": n,
        "q": q,
        "num_qubits": 2 * n + q,
    }
    if res is not None:
        out["gate_count"] = int(res.num_gates)
        out["depth"] = int(res.depth)
        out["gate_types"] = dict(res.gate_types)
    return out
