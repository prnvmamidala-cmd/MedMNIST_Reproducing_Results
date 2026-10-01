# extension_resolution_gradcam.py
# ============================================================
# EXTENSION SKELETON — RESOLUTION PARETO + GRAD-CAM STABILITY
# ============================================================
#
# Targets author limitations (Discussion / conclusion):
# 1) "further investigation is needed to determine the existence
#    and dataset specificity of this optimal image resolution"
# 2) "we do not analyze the interpretability and explainability
#    of the assessed architectures" / "future work should further
#    investigate these aspects"
#
# This file is a DRY-RUN / scaffolding entrypoint.
# It does NOT invent metrics. Full training across 28/64/128
# is left for a GPU pass; see docs/EXPERIMENTAL_PLAN.md.
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs" / "extension"


# ============================================================
# 2. CONFIG — planned ablations (not executed for metrics here)
# ============================================================

DATASETS   = ["pneumoniamnist", "bloodmnist", "pathmnist"]
RESOLUTIONS = [28, 64, 128]          # 224 deferred (paper: diminishing returns)
MODEL      = "resnet18_imagenet"
K_CAM_LAYERS = ["layer3", "layer4"]  # Grad-CAM target candidates
METRICS_PLANNED = [
    "test_acc", "test_auc", "train_sec_per_epoch", "infer_ms_per_image",
    "cam_spearman_vs_64", "cam_iou_topk10_vs_64", "cam_entropy",
]


# ============================================================
# 3. ABLATION SPECS (controls)
# ============================================================

def ablation_specs() -> list[dict]:
    # A0: paper-like single-res baseline (our reduced 28 run)
    # A1: same model/hparams across {28,64,128} — resolution sweep
    # A2: Grad-CAM on best val checkpoint per resolution
    # A3: CAM stability = correlate 28/128 maps upsampled to 64 grid
    specs = []
    for ds in DATASETS:
        for r in RESOLUTIONS:
            specs.append({
                "ablation": "A1_resolution_sweep",
                "dataset": ds,
                "img_size": r,
                "model": MODEL,
                "train": True,
                "gradcam": False,
            })
            specs.append({
                "ablation": "A2_gradcam",
                "dataset": ds,
                "img_size": r,
                "model": MODEL,
                "train": False,
                "gradcam": True,
                "cam_layers": K_CAM_LAYERS,
            })
    specs.append({
        "ablation": "A3_cam_stability",
        "dataset": "ALL",
        "reference_size": 64,
        "compare_sizes": [28, 128],
        "metrics": ["spearman", "topk10_iou"],
    })
    return specs


# ============================================================
# 4. MAIN — dry-run only
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Write ablation specs JSON; do not train or score")
    parser.add_argument("--execute", action="store_true",
                        help="Reserved for GPU pass (not implemented here)")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    specs = ablation_specs()
    out_path = OUT / "ablation_specs.json"
    payload = {
        "status": "DRY_RUN_ONLY",
        "note": "No extension metrics invented. Run --execute on GPU after baselines.",
        "n_jobs_planned": len(specs),
        "metrics_planned": METRICS_PLANNED,
        "specs": specs,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] wrote {out_path} ({len(specs)} planned jobs)")
    if args.execute:
        print("[ERROR] --execute not implemented in this CPU package; see EXPERIMENTAL_PLAN.md")
        raise SystemExit(2)
    print("[INFO] dry-run complete — no metrics fabricated")


if __name__ == "__main__":
    main()
