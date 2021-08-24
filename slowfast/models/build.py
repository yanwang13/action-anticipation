#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    return model

def set_finetune_mode(model, mode):
    if mode == 'all':
        for param in model.parameters():
            param.requires_grad = True
    elif mode == 'fc':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters(): # Only update the fc layer's params
            param.requires_grad = True
    elif mode == 'freeze_s1_s2_s3':
        for param in model.parameters():
            param.requires_grad = True
        for param in model.s1.parameters(): # Only freeze lower level conv block
            param.requires_grad = False
        for param in model.s2.parameters(): # Only freeze lower level conv block
            param.requires_grad = False
        for param in model.s3.parameters(): # Only freeze lower level conv block
            param.requires_grad = False
    elif mode == 's5_fc':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.s5.parameters(): # Only freeze lower level conv block
            param.requires_grad = True
        for param in model.head.parameters(): # Only update the fc layer's params
            param.requires_grad = True
    else:
        raise ValueError(f'finetune mode {mode} not supported.')
