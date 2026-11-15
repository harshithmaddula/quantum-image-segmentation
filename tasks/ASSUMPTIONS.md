# T1-T3 Assumptions

This file records implementation choices where the brief leaves details open. Any reported task results should cite the applicable choices.

## T1 NEQR resources

- **Encoding:** 8-bit grayscale NEQR with `2n` position qubits plus 8 intensity qubits and no ancillas; the tested sizes are 8x8, 16x16, and 32x32 crops from `5.1.12.tiff`, `4.2.03.tiff`, and `boat.512.tiff`.
- **Image preparation:** centered crops with no resize interpolation, converted to `uint8`.
- **Qiskit transpilation:** basis `{cx,u}`, optimization level 1, transpiler seed 17, one process, no coupling map, and ancilla-free `noaux_v24` MCX synthesis. Reported gate count is transpiled `u+cx`; measurements, barriers, and state-save instructions are excluded. Depth is the transpiled circuit depth.
- **Validation and timing:** Aer statevector decoding is checked for exact pixel recovery. The PennyLane comparison is only for `5.1.12.tiff` at 8x8. No unreported warm-up is used; each transpile timing is one call. Worker caps are 20 minutes and 10 GiB RSS; capped runs are not estimated or reported as completed.

## T2 QHED / BSDS500

- **Paper-directed by the brief:** amplitude-encode normalized image data, append an auxiliary qubit, use Hadamard/controlled position shift/Hadamard, read the auxiliary `|1>` branch, scan horizontal and vertical directions, and combine to an edge magnitude.
- **Implementation choice:** use the complete flattened 256x256 image in a 16-qubit little-endian position register, plus one auxiliary qubit. Horizontal forward differences use a controlled modular decrement by 1; vertical differences use the transposed image and the same decrement circuit. The cyclic last-column/last-row differences are dropped and set to zero.
- **Numerical fallback:** if Qiskit HLS fails specifically on `state_preparation`, retry by rounding normalized amplitudes to 15, 12, 10, then 8 decimal places and renormalizing. Record the selected precision and maximum amplitude perturbation per image/scan. This is a fallback for synthesis robustness, not a change to the default image data or algorithm; validate against the classical edge map using the configured error check.
- **Implementation choice:** restore the image L2 norm after reading the auxiliary branch, combine signed gradients with L2 magnitude, and max-normalize the soft edge map to [0,1].
- **Transpilation reuse:** the state-preparation amplitudes change with each image, so transpilation is image-dependent and is not reused across images. Step A performs one unreported warm-up on the first pending image, then times each image including both scan transpilations.
- **Baselines grid:** Sobel kernel sizes {3, 5, 7}; Canny low thresholds {25, 50, 75} and high thresholds {100, 150, 200, 225}, retaining pairs where low < high. Tune only on the BSDS500 train split. This grid is a choice, not a claim from the QHED paper.
- **Pilot/full tuning samples:** the one-image Step B timing pilot tunes on the first 10 sorted training images; the full Step B command tunes on the entire training split. Step B performs one unreported warm-up per method on its first selected validation image before timed evaluation.
- **Threshold sweep:** 99 thresholds linearly spaced from 0.01 through 1.0 on normalized edge maps.
- **Multiple annotations:** precision matches each predicted pixel to the union of any annotator; recall counts matched boundary pixels separately for every annotator and divides by the total annotation-pixel count. Pratt FOM is computed against each annotator and averaged within image. H95 is computed per annotator, then the median annotator H95 is used per image; report the median of these image values across the test set.
- **Empty-set conventions:** boundary F is 1 for empty prediction and empty annotations, otherwise 0 if either side is empty. Pratt FOM is 1 for empty-empty and 0 if exactly one side is empty. H95 is 0 for empty-empty and infinity if exactly one side is empty.
- **Void pixels:** the available BSDS boundary maps are interpreted as binary boundaries; pixels not marked as boundary are treated as non-boundary. No void mask is inferred unless the loaded annotation data exposes one.
- **Pilot image:** first filename in sorted BSDS500 validation JPG paths; center crop to 256x256 with no interpolation for the pilot image.

## T3 QAOA MaxCut pilot

- **Image and graph:** use a centered 128x128 grayscale crop from each USC-SIPI TIFF. Apply OpenCV SLIC to the grayscale image (replicated to three channels), targeting K=16; adjust region size only as needed to keep the actual superpixel count in [12, 20]. Each superpixel is one node, with its normalized mean grayscale intensity `mu_i`.
- **Adjacency and weights:** connect superpixels sharing a 4-connected pixel boundary. Set `sigma` to the standard deviation of all node means, with `sigma=1` for the constant-image zero-variance case. Use `w_ij = 1 - exp(-(mu_i - mu_j)^2 / (2 sigma^2))` exactly, without an additive weight floor.
- **Optimization and references:** use standard MaxCut QAOA with the X mixer; test p=1..4, five COBYLA restarts per depth, at most 50 optimizer iterations per restart, and 4,096 shots per objective evaluation. Compute the exact optimum by brute force over all `2^K` assignments. Otsu labels each superpixel by thresholding its mean against the image-level Otsu threshold.
- **Final statistics and timing:** after selecting the best parameters, draw a separate 4,096-shot sample with a fresh simulator seed. Report the cut of the most frequent sampled bitstring and the mean cut over that sample, each divided by the exact optimum. Per-depth wall time includes transpilation, all restarts, and this final sample. The CSV is atomically updated after each image/depth row and successful rows are skipped on resume.
