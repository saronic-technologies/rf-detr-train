# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Type, Tuple
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class ModelConfig(BaseModel):
    # cat: Backbone parameters (from argparse defaults overriding Model code)
    vit_encoder_num_layers: int = 12
    window_block_indexes: Optional[List[int]] = None # argparse nargs='+' defaults to None
    position_embedding: str = 'sine'
    freeze_encoder: bool = False # argparse action='store_true' defaults to False
    rms_norm: bool = False # argparse action='store_true' defaults to False
    backbone_lora: bool = False # argparse action='store_true' defaults to False
    force_no_pretrain: bool = False # argparse action='store_true' defaults to False
    pretrained_encoder: Optional[str] = None
    encoder_only: bool = False # argparse action='store_true' defaults to False
    backbone_only: bool = False # argparse action='store_true' defaults to False

    # cat: Transformer parameters (from argparse defaults overriding Model code)
    dim_feedforward: int = 2048
    hidden_dim: int = 256
    sa_nheads: int = 8
    ca_nheads: int = 8
    num_queries: int = 300
    num_select: int = 300
    decoder_norm: str = 'LN'
    freeze_batch_norm: bool = False # argparse action='store_true' defaults to False
    use_cls_token: bool = False # argparse action='store_true' defaults to False

    # cat: Matcher parameters
    set_cost_class: float = 2.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0

    # cat: Loss coefficients
    cls_loss_coef: float = 1.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    aux_loss: bool = True # argparse action='store_false', dest='aux_loss' defaults to True
    sum_group_losses: bool = False # argparse action='store_true' defaults to False

    ia_bce_loss: bool = True
    use_varifocal_loss: bool = False # argparse action='store_true' defaults to False
    use_position_supervised_loss: bool = True # argparse action='store_true' defaults to False

    # cat: Training parameters
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    lr_component_decay: float = 0.7
    lr_vit_layer_decay: float = 0.8
    weight_decay: float = 1e-4
    out_feature_indexes: List[int] = [2, 5, 8, 11] # Taken from model params!

    # cat: Misc parameters
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    num_feature_levels: int = -1 # Filled in by Model code

    # cat: Drop parameters
    dropout: float = 0.0
    drop_path: float = 0.0
    drop_mode: str = 'standard'
    drop_schedule: str = 'constant'
    cutoff_epoch: int = 0

    # Original parameters from Roboflow repo
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"]
    dec_layers: int = 3
    two_stage: bool = True
    projector_scale: List[Literal["P3", "P4", "P5"]]
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = 90
    pretrain_weights: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    shape: Tuple[int, int] = (784, 784)
    group_detr: int = 13
    gradient_checkpointing: bool = False

    model_config = {
        "arbitrary_types_allowed": True
    }

class RFDETRBaseConfig(ModelConfig):
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_small"
    hidden_dim: int = 256
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    pretrain_weights: Optional[str] = "rf-detr-base.pth"

class RFDETRLargeConfig(RFDETRBaseConfig):
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_base"
    hidden_dim: int = 384
    sa_nheads: int = 12
    ca_nheads: int = 24
    dec_n_points: int = 4
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P5"]
    pretrain_weights: Optional[str] = "rf-detr-large.pth"
