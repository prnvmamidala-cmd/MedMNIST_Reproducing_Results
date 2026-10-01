# Experimental Plan — Original Extension

**Paper:** Doerrich, Di Salvo, Brockmann, Ledig. *Rethinking model prototyping through the MedMNIST+ dataset collection.* Sci Rep 15:7669 (2025). doi:10.1038/s41598-025-92156-9 · arXiv:2404.15786.

**Author limitation quotes (Discussion):**

> "Therefore, further investigation is needed to determine the existence and dataset specificity of this optimal image resolution, as well as its contribution to overall model performance."

> "While our work does not explicitly evaluate these aspects [robustness, interpretability, explainability] … future work should further investigate these aspects to ensure that models not only achieve high performance but also provide clinically meaningful and interpretable insights for practitioners."

**Novelty stance:** Treat the design below as a **hypothesis**. Literature check (2026-09-22): Doerrich et al. leave both optimal-resolution dataset-specificity and XAI unmeasured; Grad-CAM reliability papers exist for medical imaging broadly, but no follow-up was found that jointly reports a **resolution–accuracy–latency Pareto** and **Grad-CAM spatial stability across 28/64/128** on the MedMNIST+ splits/models from this paper. Do **not** claim priority without a refreshed citation sweep at write-up.

---

## 1. Research question

For ResNet-18 end-to-end on PneumoniaMNIST / BloodMNIST / PathMNIST:

1. Is there a **dataset-specific** resolution on the accuracy–latency Pareto between 28, 64, and 128 (224 deferred given the paper’s diminishing-returns finding)?
2. Do Grad-CAM maps remain **spatially stable** when the same architecture is trained at those resolutions (Spearman / top-k IoU after aligning to a 64×64 grid)?

---

## 2. Controls (fixed)

| Control | Setting |
|---|---|
| Splits | Official MedMNIST train/val/test (no reshuffle) |
| Model | ResNet-18 ImageNet init; replace FC |
| Optim | AdamW 1e-4, cosine, batch 64 |
| Selection | Early stop / best ckpt on **val AUC only** |
| Test | Scored once per (dataset, resolution) |
| Seed | 9930641 (paper seed #1) for primary; ±2 seeds optional |

---

## 3. Ablations

| ID | What changes | What it tests |
|---|---|---|
| A0 | Native 28, reduced epochs (this package’s measured baseline) | Reproduction floor |
| A1 | Same hparams at {28, 64, 128} | Resolution–ACC/AUC curve |
| A2 | Wall-clock train sec/epoch + infer ms/image at each res | Latency axis of Pareto |
| A3 | Grad-CAM on `layer3`/`layer4` for correct-class maps | Explainability layer |
| A4 | Upsample CAMs to 64; Spearman + top-10% IoU vs 64-ref | CAM stability across res |

Optional A5: zero-pad-to-224 (paper preprocess) at fixed native content size — isolates pad vs resize.

---

## 4. Metrics (planned — **not invented here**)

- Test ACC (op=0.5), macro/ binary AUC
- Train sec/epoch, infer ms/image (CPU and GPU if available)
- CAM Spearman (map vs 64-ref), top-10% IoU, CAM entropy

Success (exploratory): for ≥2/3 datasets, a resolution **<224** lies on the Pareto front (no other res with both higher AUC and lower latency), **and** CAM Spearman(28,64) or (128,64) stays ≥0.5 on a majority of correctly classified test images.

---

## 5. Literature check (brief)

- Doerrich et al. 2025: document diminishing returns 128→224; call for dataset-specific optimal resolution; explicitly skip XAI.
- MedMNIST+ construction notes (Yang et al. / Zenodo): same splits across sizes — enables fair multi-res comparison.
- Grad-CAM reliability varies by architecture/modality; validate with faithfulness metrics, not visuals alone (general XAI lit.).
- No located 2025–2026 paper that closes **both** gaps on this exact MedMNIST+ ResNet-18 protocol.

---

## 6. Compute budget

| Stage | Hardware | Est. |
|---|---|---|
| A0 reduced 28×3 datasets | CPU | ~30–60 min (this package) |
| A1–A2 full 3 res × 3 ds | GPU | hours |
| A3–A4 CAM pass | GPU/CPU | <1 h after checkpoints |

Dry-run skeleton: `code/extension_resolution_gradcam.py --dry-run` (writes planned job specs only; **no fabricated metrics**).
