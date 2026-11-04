#!/usr/bin/env python3
"""
evaluate.py
===========
Rigorous cross-domain evaluation engine for quantifying the Generalization Gap
between laboratory data (PlantVillage) and real-world field conditions (PlantDoc).

Metrics:
- In-Domain Test: Top-1 Accuracy, Macro F1, Weighted F1, Precision, Recall
- Out-of-Domain Zero-Shot: Top-1 Accuracy, Macro F1, Weighted F1, Precision, Recall
- Domain Shift Statistics: Delta Accuracy, Delta Macro F1, Retention Rate (%)
- Per-class performance breakdown and confusion matrix visualizations.

Author: KU Leuven / VSC Deep Learning Project
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless HPC nodes
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src and root are in sys.path
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.dataset import (
    MappedCropDataset,
    get_transforms,
    get_or_create_dataset_splits,
    create_stratified_split,
)
from src.models import build_model
from data.dataset_mapping import (
    get_canonical_class_list,
    get_canonical_name_to_id,
    get_id_to_canonical_name,
)


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs forward inference and collects all predictions, ground-truths, and softmax probabilities.
    
    Device Transfer Policy:
    - images: transferred to device with non_blocking=True for model forward pass.
    - targets: kept on CPU to eliminate redundant GPU allocation and transfer overhead,
      since metrics (accuracy, macro F1, confusion matrix) are computed on host via scikit-learn.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_preds.extend(preds.detach().cpu().numpy())
        if isinstance(targets, torch.Tensor):
            all_targets.extend(targets.detach().cpu().numpy())
        else:
            all_targets.extend(np.array(targets))
        all_probs.extend(probs.detach().cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def calculate_domain_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str]
) -> Dict[str, Any]:
    """Computes aggregate and per-class classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    p_cls, r_cls, f1_cls, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names))), zero_division=0
    )
    
    per_class_data = {}
    for idx, cname in enumerate(class_names):
        per_class_data[cname] = {
            "class_id": idx,
            "precision": float(p_cls[idx]),
            "recall": float(r_cls[idx]),
            "f1": float(f1_cls[idx]),
            "support": int(support[idx]),
        }

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "weighted_f1": float(f1_weighted),
        "total_samples": int(len(y_true)),
        "per_class": per_class_data,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
):
    """Generates and saves a normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Normalize by row (true class count)
    with np.errstate(all='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Accuracy'}
    )
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {output_path}")


def plot_per_class_gap(
    pv_per_class: Dict[str, Any],
    pd_per_class: Dict[str, Any],
    class_names: List[str],
    output_path: Path,
):
    """Side-by-side bar chart showing F1 degradation for each crop disease under domain shift."""
    active_classes = [c for c in class_names if pd_per_class[c]["support"] > 0]
    
    pv_f1s = [pv_per_class[c]["f1"] for c in active_classes]
    pd_f1s = [pd_per_class[c]["f1"] for c in active_classes]

    x = np.arange(len(active_classes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(16, 7))
    rects1 = ax.bar(x - width/2, pv_f1s, width, label='In-Domain (PlantVillage Lab)', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, pd_f1s, width, label='Out-of-Domain (PlantDoc Field)', color='#d95f02')

    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class Generalization Gap: Lab vs. Field Performance', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(active_classes, rotation=60, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved per-class domain gap plot to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Domain Shift (PlantVillage -> PlantDoc).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--model_name", type=str, default="", help="Architecture ('resnet50' or 'swin_t'). If empty, reads from checkpoint.")
    parser.add_argument("--pv_dir", type=str, default="data/PlantVillage", help="PlantVillage source dataset root directory")
    parser.add_argument("--pd_dir", type=str, default="data/PlantDoc", help="PlantDoc target field dataset root directory")
    parser.add_argument("--split_manifest", type=str, default="data/dataset_splits.json", help="Path to persistent dataset split manifest JSON")
    parser.add_argument("--output_dir", type=str, default="experiments/evaluation_results", help="Directory to save report and figures")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Seed matching training split")
    return parser.parse_args()


def main():
    args = parse_args()
    args.pv_dir = os.path.expandvars(os.path.expanduser(args.pv_dir))
    args.pd_dir = os.path.expandvars(os.path.expanduser(args.pd_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" DOMAIN SHIFT BENCHMARK EVALUATION ENGINE")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Source:     {args.pv_dir} (PlantVillage)")
    print(f" Target:     {args.pd_dir} (PlantDoc)")
    print(f" Manifest:   {args.split_manifest}")
    print(f" Device:     {device}")
    print("=" * 80)

    # 1. Load Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = args.model_name or checkpoint.get("model_name") or checkpoint.get("config", {}).get("model_name", "resnet50")
    
    class_names = get_canonical_class_list()
    num_classes = len(class_names)

    # 2. Reconstruct Model
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    eval_transform = get_transforms(img_size=224, is_training=False)

    # 3. Prepare In-Domain Source Test Split (PlantVillage) via persistent manifest
    splits = get_or_create_dataset_splits(
        pv_dir=args.pv_dir,
        manifest_path=args.split_manifest,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )
    test_pv_dataset = MappedCropDataset(
        args.pv_dir, 
        domain="pv", 
        transform=eval_transform, 
        samples_file_list=splits["test_pv"]
    )
    
    pv_test_loader = torch.utils.data.DataLoader(
        test_pv_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. Prepare Out-of-Domain Target Dataset (PlantDoc)
    pd_dataset = MappedCropDataset(args.pd_dir, domain="pd", transform=eval_transform)
    pd_test_loader = torch.utils.data.DataLoader(
        pd_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 5. Run Inferences
    print("\n[1/2] Running inference on In-Domain PlantVillage test set...")
    pv_preds, pv_targets, pv_probs = run_inference(model, pv_test_loader, device)
    pv_metrics = calculate_domain_metrics(pv_targets, pv_preds, class_names)

    print("[2/2] Running inference on Out-of-Domain PlantDoc field set...")
    pd_preds, pd_targets, pd_probs = run_inference(model, pd_test_loader, device)
    pd_metrics = calculate_domain_metrics(pd_targets, pd_preds, class_names)

    # 6. Domain Shift Statistics
    delta_acc = pv_metrics["accuracy"] - pd_metrics["accuracy"]
    delta_f1 = pv_metrics["macro_f1"] - pd_metrics["macro_f1"]
    retention_f1 = (pd_metrics["macro_f1"] / max(pv_metrics["macro_f1"], 1e-6)) * 100.0

    # 7. Print Comprehensive Executive Summary Table
    print("\n" + "=" * 80)
    print(" DOMAIN GENERALIZATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} {'In-Domain (PlantVillage)':<25} {'Out-of-Domain (PlantDoc)':<25} {'Domain Gap (Drop)':<15}")
    print("-" * 80)
    print(f"{'Top-1 Accuracy':<25} {pv_metrics['accuracy']*100:6.2f}%{'':<18} {pd_metrics['accuracy']*100:6.2f}%{'':<18} {-delta_acc*100:6.2f}%")
    print(f"{'Macro F1 Score':<25} {pv_metrics['macro_f1']*100:6.2f}%{'':<18} {pd_metrics['macro_f1']*100:6.2f}%{'':<18} {-delta_f1*100:6.2f}%")
    print(f"{'Weighted F1 Score':<25} {pv_metrics['weighted_f1']*100:6.2f}%{'':<18} {pd_metrics['weighted_f1']*100:6.2f}%{'':<18} {-(pv_metrics['weighted_f1']-pd_metrics['weighted_f1'])*100:6.2f}%")
    print(f"{'Macro Precision':<25} {pv_metrics['macro_precision']*100:6.2f}%{'':<18} {pd_metrics['macro_precision']*100:6.2f}%{'':<18} {-(pv_metrics['macro_precision']-pd_metrics['macro_precision'])*100:6.2f}%")
    print(f"{'Macro Recall':<25} {pv_metrics['macro_recall']*100:6.2f}%{'':<18} {pd_metrics['macro_recall']*100:6.2f}%{'':<18} {-(pv_metrics['macro_recall']-pd_metrics['macro_recall'])*100:6.2f}%")
    print("-" * 80)
    print(f"F1 Performance Retention Rate: {retention_f1:.2f}%")
    print(f"Tested Samples: In-Domain = {pv_metrics['total_samples']}, Out-of-Domain = {pd_metrics['total_samples']}")
    print("=" * 80 + "\n")

    # 8. Visualizations
    plot_confusion_matrix(
        pv_targets, pv_preds, class_names, 
        out_dir / "confusion_matrix_source_lab.png", 
        title="PlantVillage In-Domain Confusion Matrix (Lab)"
    )
    plot_confusion_matrix(
        pd_targets, pd_preds, class_names, 
        out_dir / "confusion_matrix_target_field.png", 
        title="PlantDoc Out-of-Domain Confusion Matrix (Field)"
    )
    plot_per_class_gap(
        pv_metrics["per_class"], pd_metrics["per_class"], class_names,
        out_dir / "per_class_domain_gap.png"
    )

    # 9. Export JSON Report
    report = {
        "model_architecture": model_name,
        "checkpoint_path": str(args.checkpoint),
        "source_domain": {
            "dataset": "PlantVillage",
            "accuracy": pv_metrics["accuracy"],
            "macro_f1": pv_metrics["macro_f1"],
            "weighted_f1": pv_metrics["weighted_f1"],
            "macro_precision": pv_metrics["macro_precision"],
            "macro_recall": pv_metrics["macro_recall"],
            "total_samples": pv_metrics["total_samples"],
            "per_class": pv_metrics["per_class"],
        },
        "target_domain": {
            "dataset": "PlantDoc",
            "accuracy": pd_metrics["accuracy"],
            "macro_f1": pd_metrics["macro_f1"],
            "weighted_f1": pd_metrics["weighted_f1"],
            "macro_precision": pd_metrics["macro_precision"],
            "macro_recall": pd_metrics["macro_recall"],
            "total_samples": pd_metrics["total_samples"],
            "per_class": pd_metrics["per_class"],
        },
        "domain_shift_analysis": {
            "delta_accuracy": delta_acc,
            "delta_macro_f1": delta_f1,
            "f1_retention_percentage": retention_f1,
        }
    }

    report_path = out_dir / "domain_shift_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"[INFO] Detailed JSON domain gap report saved to {report_path}")


if __name__ == "__main__":
    main()
