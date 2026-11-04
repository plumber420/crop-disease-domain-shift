#!/usr/bin/env python3
"""
models.py
=========
Model definitions and initialization for crop disease domain shift benchmarking:
1. CNN Baseline: ResNet-50 (Residual Network with localized inductive biases)
2. Vision Transformer: Swin Transformer Tiny (Swin-T, shifted window self-attention)

Supports:
- ImageNet-1K pre-trained weight initialization.
- Head replacement for canonical crop-disease classes.
- Parameter grouping with differential learning rates (backbone vs new head).
- Target layer extraction for Grad-CAM explainability.

Author: KU Leuven / VSC Deep Learning Project
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from torchvision import models


def create_resnet50(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Initializes ResNet-50 with ImageNet weights and customized classification head.
    
    Args:
        num_classes: Number of target crop disease classes.
        pretrained: If True, load ImageNet-1K pretrained weights (V2).
        dropout: Dropout probability before the final linear projection.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # In ResNet-50, fc is the classification head (in_features = 2048)
    in_features = model.fc.in_features
    if dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    # Attach model metadata for explainability & logging
    model.model_name = "resnet50"
    model.num_classes = num_classes
    # Target layer for Grad-CAM is the final bottleneck block of stage 4
    model.gradcam_target_layer = model.layer4[-1]

    return model


def create_swin_t(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Initializes Swin Transformer Tiny (Swin-T) with ImageNet weights.
    
    Architecture:
    - Hierarchical feature maps with shifted window self-attention.
    - Captures multi-scale contextual features with linear computational complexity.
    """
    weights = models.Swin_T_Weights.DEFAULT if pretrained else None
    model = models.swin_t(weights=weights)

    # In torchvision Swin-T, head is the final projection layer (in_features = 768)
    in_features = model.head.in_features
    if dropout > 0.0:
        model.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.head = nn.Linear(in_features, num_classes)

    model.model_name = "swin_t"
    model.num_classes = num_classes
    # Target layer for Swin feature attribution is the final normalization/stage
    model.gradcam_target_layer = model.features[-1]

    return model


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory function to build and configure the requested vision architecture.
    """
    name_clean = model_name.lower().replace("-", "_")
    if name_clean in ("resnet50", "resnet_50"):
        model = create_resnet50(num_classes, pretrained=pretrained, dropout=dropout)
    elif name_clean in ("swin_t", "swin_tiny", "swin"):
        model = create_swin_t(num_classes, pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(
            f"Unsupported model architecture: '{model_name}'. "
            f"Choose from ['resnet50', 'swin_t']."
        )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Architecture: {model.model_name.upper()}")
    print(f"[INFO] Total parameters:     {total_params / 1e6:.2f}M")
    print(f"[INFO] Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"[INFO] Target classes:       {num_classes}")

    return model


def get_parameter_groups(
    model: nn.Module,
    base_lr: float,
    head_lr_multiplier: float = 5.0,
    weight_decay: float = 1e-2,
) -> List[Dict[str, Any]]:
    """
    Constructs parameter groups with differential learning rates:
    - Pretrained backbone parameters receive base_lr (fine-tuning).
    - Newly initialized classification head parameters receive (base_lr * head_lr_multiplier).
    - Weight decay is omitted from 1D parameters (biases, LayerNorm, BatchNorm).
    """
    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    is_head_param = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Determine if parameter belongs to the classification head
        if "fc" in name or "head" in name:
            if param.ndim <= 1 or name.endswith(".bias"):
                head_no_decay.append(param)
            else:
                head_decay.append(param)
        else:
            if param.ndim <= 1 or name.endswith(".bias"):
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

    groups = [
        {"params": backbone_decay, "lr": base_lr, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": base_lr, "weight_decay": 0.0},
        {"params": head_decay, "lr": base_lr * head_lr_multiplier, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": base_lr * head_lr_multiplier, "weight_decay": 0.0},
    ]

    return groups


if __name__ == "__main__":
    print("Testing model initialization...")
    num_classes = 29
    
    # 1. ResNet-50
    resnet = build_model("resnet50", num_classes=num_classes, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out_resnet = resnet(x)
    assert out_resnet.shape == (2, num_classes), f"ResNet output shape mismatch: {out_resnet.shape}"
    print("[SUCCESS] ResNet-50 forward pass verified.")

    # 2. Swin-T
    swin = build_model("swin_t", num_classes=num_classes, pretrained=False)
    out_swin = swin(x)
    assert out_swin.shape == (2, num_classes), f"Swin-T output shape mismatch: {out_swin.shape}"
    print("[SUCCESS] Swin-T forward pass verified.")
