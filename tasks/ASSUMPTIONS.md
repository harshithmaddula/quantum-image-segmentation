# T1-T3 Assumptions

This file records implementation choices where the brief leaves details open. Any reported task results should cite the applicable choices.

## T2 QHED / BSDS500

- **Paper-directed by the brief:** amplitude-encode normalized image data, append an auxiliary qubit, use Hadamard/controlled position shift/Hadamard, read the auxiliary `|1>` branch, scan horizontal and vertical directions, and combine to an edge magnitude.
- **Implementation choice:** use the complete flattened 256x256 image in a 16-qubit little-endian position register, plus one auxiliary qubit. Horizontal forward differences use a controlled modular decrement by 1; vertical differences use the transposed image and the same decrement circuit. The cyclic last-column/last-row differences are dropped and set to zero.
- **Implementation choice:** restore the image L2 norm after reading the auxiliary branch, combine signed gradients with L2 magnitude, and max-normalize the soft edge map to [0,1].
- **Transpilation reuse:** the state-preparation amplitudes change with each image, so transpilation is image-dependent and is not reused across images. Step A performs one unreported warm-up on the first pending image, then times each image including both scan transpilations.
- **Baselines grid:** Sobel kernel sizes {3, 5, 7}; Canny low thresholds {25, 50, 75} and high thresholds {100, 150, 200, 225}, retaining pairs where low < high. Tune only on the BSDS500 train split. This grid is a choice, not a claim from the QHED paper.
- **Pilot/full tuning samples:** the one-image Step B timing pilot tunes on the first 10 sorted training images; the full Step B command tunes on the entire training split. Step B performs one unreported warm-up per method on its first selected validation image before timed evaluation.
- **Threshold sweep:** 99 thresholds linearly spaced from 0.01 through 1.0 on normalized edge maps.
- **Multiple annotations:** precision matches each predicted pixel to the union of any annotator; recall counts matched boundary pixels separately for every annotator and divides by the total annotation-pixel count. Pratt FOM is computed against each annotator and averaged within image. H95 is computed per annotator, then the median annotator H95 is used per image; report the median of these image values across the test set.
- **Empty-set conventions:** boundary F is 1 for empty prediction and empty annotations, otherwise 0 if either side is empty. Pratt FOM is 1 for empty-empty and 0 if exactly one side is empty. H95 is 0 for empty-empty and infinity if exactly one side is empty.
- **Void pixels:** the available BSDS boundary maps are interpreted as binary boundaries; pixels not marked as boundary are treated as non-boundary. No void mask is inferred unless the loaded annotation data exposes one.
- **Pilot image:** first filename in sorted BSDS500 validation JPG paths; center crop to 256x256 with no interpolation for the pilot image.
