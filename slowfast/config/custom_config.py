#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.TRAIN.MULTI_TASK = False
    _C.TRAIN.FINETUNE = False
    _C.TRAIN.FINETUNE_EPOCH = 3
    return _C
