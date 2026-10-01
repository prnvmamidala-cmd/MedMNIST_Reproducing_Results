# reproduce_medmnist_baselines.py
# ============================================================
# MEDMNIST+ RESNET-18 END-TO-END BASELINE REPRODUCTION
# ============================================================
#
# Paper: Doerrich, Di Salvo, Brockmann, Ledig.
#        "Rethinking model prototyping through the MedMNIST+
#         dataset collection" Scientific Reports 2025.
#        https://doi.org/10.1038/s41598-025-92156-9
# Code:  https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus
# Data:  pip install medmnist  (MedMNIST+; download=True)
#
# Goal:
# - Train / evaluate ResNet-18 end-to-end on 3 MedMNIST+ subsets
#   (PneumoniaMNIST, BloodMNIST, PathMNIST) at 28x28.
# - Write measured ACC / AUC to outputs/ (real numbers only).
# - Compare against authors' published ResNet-18 E2E @ 28x28
#   (Supplementary Tables S1 / S9 / S10; ACC%/AUC% as reported).
#
# Protocol vs paper (IMPORTANT — anti-leakage + honesty):
# - Paper: 100 epochs, AdamW 1e-4, cosine anneal, batch 64,
#   early stop on VAL, ImageNet-pretrained, ZERO-PAD to 224,
#   3 seeds, report mean+/-sd.
# - THIS SCRIPT (REDUCED, labeled): native 28x28 (no pad-to-224),
#   fewer epochs (see EPOCHS / EARLY_STOP_PATIENCE), ONE seed.
#   Val used ONLY for early-stopping / best-checkpoint selection.
#   Test scored ONCE with the frozen best checkpoint.
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import medmnist
    from medmnist import INFO
except Exception as e:  # pragma: no cover
    print(f"[ERROR] medmnist import failed: {e}")
    sys.exit(1)


# ============================================================
# 2. CONFIG
# ============================================================

SEED              = 9930641          # paper seed #1
IMG_SIZE          = 28               # MedMNIST+ 28x28 track
BATCH_SIZE        = 64               # paper
BATCH_SIZE_EVAL   = 256              # paper
LR                = 1e-4             # paper AdamW
EPOCHS_FULL       = 100              # paper (not default here)
EPOCHS_REDUCED    = 10               # our measured reduced protocol
EARLY_STOP_PATIENCE = 3
NUM_WORKERS       = 0
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# Subsets emphasized in the brief / paper taxonomy
DATASETS = ["pneumoniamnist", "bloodmnist", "pathmnist"]

# Published ResNet-18 End-to-End @ 28x28 from Doerrich et al. 2025
# Supplementary tables (ACC% / AUC%, mean over 3 seeds). Parsed from
# arXiv HTML 2404.15786 appendix (Blood S1, Path S9, Pneumonia S10).
PUBLISHED_28 = {
    "bloodmnist":     {"acc": 0.9193, "auc": 0.9924, "acc_sd": 0.0037, "auc_sd": 0.0004},
    "pathmnist":      {"acc": 0.8554, "auc": 0.9824, "acc_sd": 0.0087, "auc_sd": 0.0014},
    "pneumoniamnist": {"acc": 0.8312, "auc": 0.9501, "acc_sd": 0.0110, "auc_sd": 0.0030},
}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"


# ============================================================
# 3. HELPERS — seed, model, metrics
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_resnet18(n_classes: int, task: str) -> nn.Module:
    # ImageNet-pretrained ResNet-18; replace final FC  # +bias in Linear
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    # multi-label chest uses BCE; our three tasks are BC / MC -> CE over logits
    model.fc = nn.Linear(in_features, n_classes)
    return model


def get_transform():
    # Keep native 28x28; ImageNet norm (paper pads to 224 — we document the delta)
    # Force RGB so grayscale flags (e.g. PneumoniaMNIST) match ResNet-18 stem
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_medmnist(flag: str, split: str, size: int = IMG_SIZE):
    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])
    # download=True — official MedMNIST API; no third-party clone
    ds = DataClass(split=split, download=True, transform=get_transform(), size=size,
                   root=str(DATA_DIR))
    return ds, info


@torch.no_grad()
def evaluate(model, loader, task: str, n_classes: int, device: str):
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        x = x.to(device)
        # MedMNIST labels shape (N,1) or (N,C)
        y_np = y.numpy()
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np.reshape(-1)
        logits = model(x)
        if task == "multi-label, binary-class":
            p = torch.sigmoid(logits).cpu().numpy()
        else:
            p = torch.softmax(logits, dim=1).cpu().numpy()
        ys.append(y_np)
        probs.append(p)
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(probs, axis=0)

    # ACC at fixed operating point 0.5 (paper / MedMNIST convention)
    if task == "multi-label, binary-class":
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))
        try:
            auc = roc_auc_score(y_true, y_prob, average="macro")
        except ValueError:
            auc = float("nan")
    else:
        y_pred = y_prob.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
    return float(acc), float(auc)


def train_one_dataset(flag: str, epochs: int, device: str, smoke: bool = False) -> dict:
    print(f"\n====== {flag.upper()} ======")
    set_seed(SEED)

    train_ds, info = load_medmnist(flag, "train")
    val_ds, _      = load_medmnist(flag, "val")
    test_ds, _     = load_medmnist(flag, "test")
    task = info["task"]
    n_classes = len(info["label"])
    print(f"[INFO] task={task} n_classes={n_classes} "
          f"n_train={len(train_ds)} n_val={len(val_ds)} n_test={len(test_ds)}")

    if smoke:
        # Tiny subsets for plumbing check — NOT a measured baseline
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(128, len(train_ds)))))
        val_ds   = Subset(val_ds,   list(range(min(64,  len(val_ds)))))
        test_ds  = Subset(test_ds,  list(range(min(64,  len(test_ds)))))
        epochs   = min(epochs, 1)
        print(f"[INFO] SMOKE mode: subsetted splits, epochs={epochs}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE_EVAL, shuffle=False,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE_EVAL, shuffle=False,
                              num_workers=NUM_WORKERS)

    model = build_resnet18(n_classes, task).to(device)
    # CE for BC / MC / OR (paper); our three flags are BC or MC
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val_auc = -1.0
    best_state = None
    patience_left = EARLY_STOP_PATIENCE
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.squeeze().long().to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n_seen += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        # Val ONLY for early stop / checkpoint — never touch test here  # anti-leakage
        val_acc, val_auc = evaluate(model, val_loader, task, n_classes, device)
        print(f"[INFO] epoch={epoch} train_loss={running/max(n_seen,1):.4f} "
              f"val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[INFO] early stop at epoch={epoch} (best_val_auc={best_val_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # TEST once with frozen best checkpoint
    test_acc, test_auc = evaluate(model, test_loader, task, n_classes, device)
    elapsed = time.time() - t0
    print(f"[INFO] TEST acc={test_acc:.4f} auc={test_auc:.4f} elapsed_s={elapsed:.1f}")

    pub = PUBLISHED_28.get(flag, {})
    row = {
        "dataset": flag,
        "protocol": "REDUCED_native28_noPad224",
        "model": "resnet18_imagenet",
        "img_size": IMG_SIZE,
        "epochs_ran_max": epochs,
        "seed": SEED,
        "device": device,
        "n_train": len(train_ds) if not smoke else f"smoke:{len(train_ds)}",
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "our_test_acc": round(test_acc, 6),
        "our_test_auc": round(test_auc, 6),
        "published_acc_28_e2e": pub.get("acc"),
        "published_auc_28_e2e": pub.get("auc"),
        "published_acc_sd": pub.get("acc_sd"),
        "published_auc_sd": pub.get("auc_sd"),
        "delta_acc": round(test_acc - pub["acc"], 6) if pub else None,
        "delta_auc": round(test_auc - pub["auc"], 6) if pub else None,
        "elapsed_sec": round(elapsed, 1),
        "smoke": smoke,
    }
    return row


# ============================================================
# 4. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny subset + 1 epoch plumbing check (NOT measured baseline)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch budget (default: reduced=10, or 1 if --smoke)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--paper_full", action="store_true",
                        help="Attempt paper-like 100 epochs (heavy; still no pad-to-224)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.paper_full:
        epochs = EPOCHS_FULL
        protocol_tag = "NEAR_PAPER_100ep_native28"
    elif args.epochs is not None:
        epochs = args.epochs
        protocol_tag = f"CUSTOM_{epochs}ep_native28"
    else:
        epochs = EPOCHS_REDUCED
        protocol_tag = "REDUCED_10ep_native28"

    print("====== MEDMNIST+ RESNET-18 REPRODUCTION ======")
    print(f"[INFO] device={DEVICE} protocol={protocol_tag} seed={SEED}")
    print(f"[INFO] datasets={args.datasets}")
    print(f"[INFO] NOTE: paper zero-pads 28->224; we train native 28 for CPU budget")

    rows = []
    for flag in args.datasets:
        if flag not in INFO:
            print(f"[ERROR] unknown dataset flag: {flag}")
            continue
        try:
            row = train_one_dataset(flag, epochs=epochs, device=DEVICE, smoke=args.smoke)
            row["protocol"] = protocol_tag + ("_SMOKE" if args.smoke else "")
            rows.append(row)
        except Exception as e:
            print(f"[ERROR] {flag} failed: {e}")
            rows.append({"dataset": flag, "error": str(e), "smoke": args.smoke})

    df = pd.DataFrame(rows)
    stamp = "smoke" if args.smoke else "reduced"
    out_csv = OUT_DIR / f"our_baseline_metrics_{stamp}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] wrote {out_csv}")

    # Comparison table (non-smoke only makes sense vs published)
    if not args.smoke and len(df) and "our_test_acc" in df.columns:
        cmp_path = OUT_DIR / "comparison_ours_vs_published.csv"
        df.to_csv(cmp_path, index=False)
        print(f"[INFO] wrote {cmp_path}")
        print("\n====== OURS vs PUBLISHED (ResNet-18 E2E @ 28) ======")
        for _, r in df.iterrows():
            if "our_test_acc" not in r or pd.isna(r.get("our_test_acc")):
                continue
            print(f"  {r['dataset']}: ours ACC={r['our_test_acc']:.4f} AUC={r['our_test_auc']:.4f} | "
                  f"pub ACC={r['published_acc_28_e2e']:.4f} AUC={r['published_auc_28_e2e']:.4f} | "
                  f"dACC={r['delta_acc']:+.4f} dAUC={r['delta_auc']:+.4f}")

    summary = {
        "paper": "Doerrich et al. Sci Rep 2025; doi:10.1038/s41598-025-92156-9",
        "protocol_tag": protocol_tag,
        "smoke": args.smoke,
        "device": DEVICE,
        "seed": SEED,
        "n_rows": len(rows),
        "outputs": str(out_csv),
    }
    with open(OUT_DIR / f"run_summary_{stamp}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] done")


if __name__ == "__main__":
    main()
