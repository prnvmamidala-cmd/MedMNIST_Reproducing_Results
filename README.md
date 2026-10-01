# MedMNIST+ baseline reproduction + extension plan

Independent reproduction package for **Doerrich et al., Scientific Reports 2025**:
*Rethinking model prototyping through the MedMNIST+ dataset collection*
(doi:[10.1038/s41598-025-92156-9](https://doi.org/10.1038/s41598-025-92156-9)).

## What this package does

1. Downloads **MedMNIST+** via the official `medmnist` pip API (`download=True`) — no PhysioNet wall, no full GitHub clone.
2. Re-runs a **meaningful ResNet-18 end-to-end baseline** on **PneumoniaMNIST, BloodMNIST, PathMNIST** at **28×28**.
3. Writes **measured** test ACC / AUC to `outputs/` and compares against the authors’ published ResNet-18 E2E @ 28 numbers (Supp. S1 / S9 / S10).
4. Ships an experimental plan + dry-run skeleton for a **resolution–accuracy–latency + Grad-CAM stability** extension targeting the paper’s stated limitations — **without fabricating extension metrics**.

## What is measured vs not

| Measured (this laptop/CPU package) | Not run |
|---|---|
| ResNet-18 ImageNet-init, end-to-end, AdamW 1e-4, cosine, batch 64 | Full 100-epoch paper protocol |
| Native **28×28** (no zero-pad to 224) | Pad-to-224 paper preprocess |
| Reduced epoch budget (default 10) + early stop on **val AUC** | 3-seed mean±sd (we use paper seed #1 = 9930641) |
| Pneumonia / Blood / Path at 28 | All 12 datasets × 4 resolutions × CNN/ViT × E2E/LP/kNN |

Discrepancies vs published numbers are **expected** under the reduced/native-28 protocol; they are labeled in the report, not hidden.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# CPU torch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python code/reproduce_medmnist_baselines.py --smoke
python code/reproduce_medmnist_baselines.py
python code/extension_resolution_gradcam.py --dry-run
```

## Provenance

| Artifact | Source |
|---|---|
| Paper (OA) | https://www.nature.com/articles/s41598-025-92156-9 |
| Preprint | https://arxiv.org/abs/2404.15786 |
| Author code | https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus (MIT-style research code; we cite API, do not vendor the tree) |
| Data | MedMNIST+ via `pip install medmnist` — mostly **CC BY 4.0**; DermaMNIST is **CC BY-NC 4.0** (not used here) |
| Zenodo | https://zenodo.org/records/10519652 |

## Key results (measured 2026-09-22 PT)

ResNet-18 E2E, **REDUCED** protocol (native 28, 10 epochs, seed 9930641) vs Doerrich et al. published mean @ 28:

| Dataset | Ours ACC | Pub ACC | Ours AUC | Pub AUC |
|---|---:|---:|---:|---:|
| PneumoniaMNIST | 0.8574 | 0.8312 | 0.9618 | 0.9501 |
| BloodMNIST | 0.9430 | 0.9193 | 0.9962 | 0.9924 |
| PathMNIST | 0.8546 | 0.8554 | 0.9809 | 0.9824 |

Details: `docs/REPRODUCTION_REPORT.md` · plan: `docs/EXPERIMENTAL_PLAN.md`
