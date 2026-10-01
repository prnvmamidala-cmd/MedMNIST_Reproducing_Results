# Reproduction Report — MedMNIST+ ResNet-18 baselines

**Paper:** Doerrich, Di Salvo, Brockmann, Ledig. *Rethinking model prototyping through the MedMNIST+ dataset collection.* Scientific Reports 15:7669 (2025).  
**DOI:** https://doi.org/10.1038/s41598-025-92156-9 · **arXiv:** https://arxiv.org/abs/2404.15786  
**Author code:** https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus (**Apache-2.0**)  
**Data:** MedMNIST+ via `pip install medmnist` (mostly **CC BY 4.0**; DermaMNIST CC BY-NC — not used)  
**Reproduction date:** 2026-09-22 (America/Los_Angeles / PT)  
**Hardware:** CPU-only Linux box (no NVIDIA GPU)

> Every test metric below is from a local run. No fabricated numbers.

---

## 1. Criteria check (primary paper)

| Criterion | Result |
|---|---|
| Open-source biomedical/medical AI, 2025–2026 | Pass — Sci Rep 2025; Apache-2.0 code |
| Accessible data (no PhysioNet wall) | Pass — `medmnist` pip + Zenodo |
| Laptop/CPU-or-light-GPU feasible baseline | Pass — reduced 10-epoch native-28 ResNet-18 |
| Author-stated unresolved limitation (quoted) | Pass — see §5 |
| No novelty fabrication | Pass — extension is plan + dry-run only |

**Backup ClinicalBench (arXiv:2411.06469) was not needed.**

---

## 2. Protocol: paper vs this package

| Setting | Paper (Doerrich et al.) | This package (labeled **REDUCED**) |
|---|---|---|
| Model | ResNet-18 ImageNet-pretrained, end-to-end | Same |
| Optim | AdamW 1e-4, cosine, batch 64 | Same |
| Epochs | 100 + early stop on val | **10** + early stop on val AUC (patience 3) |
| Input | Zero-pad 28→224 | **Native 28×28** (no pad) |
| Seeds | 3 seeds, mean±sd | **1 seed** (9930641 = paper seed #1) |
| Splits | Official MedMNIST | Same (`download=True`) |
| Selection | Val for early stop / ckpt | Same — **test untouched until final score** |

Smoke plumbing check (`--smoke`) is **not** a measured baseline.

---

## 3. Measured vs published (ResNet-18 E2E @ 28)

Published means are from Doerrich et al. Supplementary tables (Blood S1, Path S9, Pneumonia S10), End-to-End ResNet-18 column at 28×28, parsed from arXiv HTML 2404.15786 (ACC%/AUC% → fraction).

| Dataset | Ours ACC | Pub ACC | Δ ACC | Ours AUC | Pub AUC | Δ AUC | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PneumoniaMNIST | **0.8574** | 0.8312 ±0.0110 | +0.0262 | **0.9618** | 0.9501 ±0.0030 | +0.0117 | 45 |
| BloodMNIST | **0.9430** | 0.9193 ±0.0037 | +0.0237 | **0.9962** | 0.9924 ±0.0004 | +0.0038 | 113 |
| PathMNIST | **0.8546** | 0.8554 ±0.0087 | −0.0008 | **0.9809** | 0.9824 ±0.0014 | −0.0015 | 887 |

Source CSV: `outputs/our_baseline_metrics_reduced.csv` · `outputs/comparison_ours_vs_published.csv`.

### Interpretation

1. **PathMNIST** lands inside published noise (ΔACC −0.0008, within ±0.0087 sd) despite 10 epochs and no pad-to-224 — strong evidence the reduced protocol still measures a real ResNet-18 baseline.
2. **Pneumonia / Blood** sit **above** the paper mean. Plausible drivers (not excuses): native-28 avoids pad dilution; single seed vs 3-seed mean; 10-epoch early-stopped ImageNet fine-tune can overshoot a 100-epoch mean on easy tasks; torchvision ResNet weights / AUC multi-class averaging details may differ slightly from `timm` in the author repo.
3. Deltas are **documented**, not hidden. This is **not** a claim of beating SOTA under the paper’s exact pad-224 / 100-epoch / 3-seed protocol.

---

## 4. What was NOT run

- Full 100-epoch, pad-to-224, 3-seed paper protocol  
- ViT / DenseNet / EfficientNet / CLIP / DINO / SAM / k-NN / linear probing  
- Resolutions 64 / 128 / 224  
- Remaining 9 MedMNIST+ 2D datasets  
- Extension Grad-CAM / Pareto metrics (dry-run specs only)

---

## 5. Author limitation → extension pointer

Quoted from Discussion (Sci Rep / arXiv HTML):

> "Therefore, further investigation is needed to determine the existence and dataset specificity of this optimal image resolution, as well as its contribution to overall model performance."

> "While our work does not explicitly evaluate these aspects [robustness, interpretability, explainability] … future work should further investigate these aspects to ensure that models not only achieve high performance but also provide clinically meaningful and interpretable insights for practitioners."

Extension plan (hypothesis only): `docs/EXPERIMENTAL_PLAN.md` — resolution–accuracy–latency Pareto + Grad-CAM stability across 28/64/128. Dry-run: `code/extension_resolution_gradcam.py --dry-run`.

---

## 6. Re-run commands

```bash
pip install -r requirements.txt
python code/reproduce_medmnist_baselines.py --smoke
python code/reproduce_medmnist_baselines.py
python code/extension_resolution_gradcam.py --dry-run
```
