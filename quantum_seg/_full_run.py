"""One-shot full experiment: stages 1-3 -> results/metrics_report.md (avoids CLI arg parsing)."""
import os
from quantum_seg import run_experiments as R

os.makedirs(R.RESULTS, exist_ok=True)
fid_imgs = ["5.1.12.tiff", "4.2.03.tiff", "boat.512.tiff"]
edge_imgs = ["5.1.12.tiff", "boat.512.tiff", "4.2.03.tiff", "house.tiff"]

parts = [
    "# Quantum Image Segmentation -- Mandatory Metrics vs SOTA\n",
    "## 1. NEQR encoding fidelity + quantum resources (spatial-preservation gate)\n",
    "_Lossless: exact reconstruction => PSNR=inf, SSIM=1, MSE=0 (every boundary/texture pixel "
    "preserved). Resource columns substantiate the 'quantum' claim (Ruan 2021)._\n",
    R.stage_fidelity(fid_imgs),
    "\n## 2. QHED vs classical baselines on misc (no ground truth)\n",
    "_misc has no masks -> tolerance agreement with classical edges + edge density._\n",
    R.stage_misc_edges(edge_imgs, size=128),
    "\n## 3. BSDS500 supervised boundary SOTA (ODS / OIS / Boundary-F1 / Pratt / Hausdorff)\n",
    R.stage_bsds(limit=30, size=256, split="val", tol=2),
]
report = "\n".join(parts) + "\n"
open(os.path.join(R.RESULTS, "metrics_report.md"), "w", encoding="utf-8").write(report)
print("FULL_RUN_DONE")
