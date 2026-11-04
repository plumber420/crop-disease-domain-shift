#!/usr/bin/env python3
"""
train.py
========
Production-grade training pipeline for deep learning-based crop disease classification.
Supports ResNet-50 and Swin Transformer, Automatic Mixed Precision (AMP),
CutMix/MixUp regularization, and metric tracking (Loss, Accuracy, Macro F1).

Designed for execution on KU Leuven VSC (wICE / Tier-2) HPC GPU clusters.

Author: KU Leuven / VSC Deep Learning Project
"""

import os
import sys
import time
import math
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score

# Ensure src and root are in sys.path
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.dataset import (
    build_dataloaders,
    SoftTargetCrossEntropy,
)
from src.models import (
    build_model,
    get_parameter_groups,
)
from data.dataset_mapping import get_canonical_class_list


def set_seed(seed: int = 42):
    """Ensures complete determinism across PyTorch, NumPy, and Python builtins."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    use_cutmix_mixup: bool = False,
) -> Dict[str, float]:
    """Runs a single training epoch with AMP support."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    start_time = time.time()
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        # Gradient clipping to prevent exploding gradients (vital for Vision Transformers)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # Track accuracy for non-soft or soft argmax
        if use_cutmix_mixup:
            target_indices = targets.argmax(dim=-1)
        else:
            target_indices = targets
            
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(target_indices.detach().cpu().numpy())

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    elapsed = time.time() - start_time

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "f1_macro": epoch_f1,
        "time": elapsed,
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluates the model on validation or test dataloader."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            if targets.ndim > 1 and targets.dtype == torch.float32:
                loss = criterion(logits, targets)
                target_indices = targets.argmax(dim=-1)
            else:
                loss = nn.functional.cross_entropy(logits, targets)
                target_indices = targets

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(target_indices.detach().cpu().numpy())

    eval_loss = running_loss / max(total_samples, 1)
    eval_acc = accuracy_score(all_targets, all_preds)
    eval_f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return {
        "loss": eval_loss,
        "acc": eval_acc,
        "f1_macro": eval_f1_macro,
        "total_samples": total_samples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Crop Disease Classifier on VSC Cluster.")
    
    # Dataset and paths
    parser.add_argument("--pv_dir", type=str, default="data/PlantVillage", help="Path to PlantVillage root directory")
    parser.add_argument("--pd_dir", type=str, default="data/PlantDoc", help="Path to PlantDoc root directory (optional validation tracking)")
    parser.add_argument("--split_manifest", type=str, default="data/dataset_splits.json", help="Path to persistent dataset split manifest JSON")
    parser.add_argument("--output_dir", type=str, default="experiments/outputs", help="Directory to save logs and checkpoints")
    parser.add_argument("--experiment_name", type=str, default="resnet50_baseline", help="Unique identifier for experiment")
    
    # Model architecture
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet50", "swin_t"], help="Model backbone")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use ImageNet-1K pretrained weights (use --no-pretrained to train from scratch)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30, help="Total number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for backbone")
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="Learning rate multiplier for new classification head")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Linear warmup epochs")
    
    # Data Augmentation & Regularization
    parser.add_argument("--use_cutmix_mixup", action="store_true", help="Enable CutMix & MixUp regularization")
    parser.add_argument("--mixup_alpha", type=float, default=0.8, help="MixUp Beta alpha")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0, help="CutMix Beta alpha")
    parser.add_argument("--heavy_aug", action="store_true", help="Enable field-simulating distortions")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    
    # Hardware & Environment
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader subprocesses")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training")

    return parser.parse_args()


def main():
    args = parse_args()
    args.pv_dir = os.path.expandvars(os.path.expanduser(args.pv_dir))
    args.pd_dir = os.path.expandvars(os.path.expanduser(args.pd_dir)) if args.pd_dir else ""
    set_seed(args.seed)

    use_amp = not args.no_amp and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup output directories
    exp_dir = Path(args.output_dir) / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    print("=" * 80)
    print(f" EXPERIMENT: {args.experiment_name}")
    print(f" Model:      {args.model_name.upper()} | Pretrained: {args.pretrained}")
    print(f" Regularizer CutMix/MixUp: {args.use_cutmix_mixup} (MixUp={args.mixup_alpha}, CutMix={args.cutmix_alpha})")
    print(f" Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Output dir: {exp_dir}")
    print("=" * 80)

    # 1. Build DataLoaders
    loaders = build_dataloaders(
        pv_dir=args.pv_dir,
        pd_dir=args.pd_dir if args.pd_dir else None,
        split_manifest_path=args.split_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cutmix_mixup=args.use_cutmix_mixup,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        heavy_aug=args.heavy_aug,
        seed=args.seed,
    )
    
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_pd_loader = loaders["test_pd_loader"]
    num_classes = loaders["num_classes"]

    # 2. Build Model
    model = build_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    model.to(device)

    # 3. Criterion
    if args.use_cutmix_mixup:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 4. Optimizer with differential LR
    param_groups = get_parameter_groups(
        model=model,
        base_lr=args.lr,
        head_lr_multiplier=args.head_lr_mult,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # 5. Cosine Annealing Learning Rate Scheduler with Warmup
    total_epochs = args.epochs
    warmup_epochs = args.warmup_epochs
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = GradScaler(enabled=use_amp)

    # State tracking
    start_epoch = 1
    best_val_f1 = 0.0

    # Resume from checkpoint if provided
    if args.resume and Path(args.resume).exists():
        print(f"[INFO] Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_f1 = checkpoint.get("best_val_f1", 0.0)

    # Initialize CSV metrics log
    csv_log_path = exp_dir / "metrics_history.csv"
    if not csv_log_path.exists() or args.resume == "":
        with open(csv_log_path, "w", encoding="utf-8") as f:
            f.write("epoch,lr,train_loss,train_acc,train_f1,val_loss,val_acc,val_f1,pd_loss,pd_acc,pd_f1\n")

    print(f"\nStarting training from epoch {start_epoch} to {total_epochs}...")
    for epoch in range(start_epoch, total_epochs + 1):
        # Current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Train 1 epoch
        train_res = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            use_cutmix_mixup=args.use_cutmix_mixup,
        )

        # Validate on in-domain PlantVillage validation split
        val_res = evaluate_loader(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        # Optionally evaluate on field Target domain (PlantDoc) to observe domain transfer curve
        pd_loss, pd_acc, pd_f1 = -1.0, -1.0, -1.0
        if test_pd_loader is not None and (epoch % 5 == 0 or epoch == total_epochs):
            pd_res = evaluate_loader(
                model=model,
                data_loader=test_pd_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )
            pd_loss = pd_res["loss"]
            pd_acc = pd_res["acc"]
            pd_f1 = pd_res["f1_macro"]

        scheduler.step()

        # Terminal feedback
        pd_status_str = f" | PD Field F1: {pd_f1:.4f}" if pd_f1 >= 0 else ""
        print(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] "
            f"Train Loss: {train_res['loss']:.4f} Acc: {train_res['acc']:.4f} F1: {train_res['f1_macro']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f} Acc: {val_res['acc']:.4f} F1: {val_res['f1_macro']:.4f}"
            f"{pd_status_str} | LR: {current_lr:.6f}"
        )

        # Append to CSV
        with open(csv_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{current_lr:.8f},{train_res['loss']:.5f},{train_res['acc']:.5f},{train_res['f1_macro']:.5f},"
                f"{val_res['loss']:.5f},{val_res['acc']:.5f},{val_res['f1_macro']:.5f},"
                f"{pd_loss:.5f},{pd_acc:.5f},{pd_f1:.5f}\n"
            )

        # Save checkpoint dictionary
        ckpt_state = {
            "epoch": epoch,
            "model_name": args.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_f1": val_res["f1_macro"],
            "best_val_f1": best_val_f1,
            "config": vars(args),
        }

        # Always save last checkpoint for SLURM preemption safety
        torch.save(ckpt_state, checkpoints_dir / "last_checkpoint.pt")

        # Save best model if validation Macro F1 improved
        if val_res["f1_macro"] > best_val_f1:
            best_val_f1 = val_res["f1_macro"]
            ckpt_state["best_val_f1"] = best_val_f1
            torch.save(ckpt_state, checkpoints_dir / "best_checkpoint.pt")
            print(f"  --> Saved new best model checkpoint (Val Macro F1: {best_val_f1:.4f})")

    print("\n" + "=" * 80)
    print(f" Training Complete for {args.experiment_name}!")
    print(f" Best Validation Macro F1: {best_val_f1:.4f}")
    print(f" Checkpoints saved to:     {checkpoints_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
