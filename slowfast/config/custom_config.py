#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.MULTI_TASK = False
    _C.UNCERTAINTY = False
    _C.LOG_VERB_NOUN = True #False

    _C.TRAIN.FINETUNE = False
    _C.TRAIN.FINETUNE_EPOCH = 3
    _C.TRAIN.SAVE_CKPT_THRES = 75.0

    # only specify when useing Epic-kitchen datasets
    _C.TRAIN.EK_VERSION = 55 # 55, 100
    _C.TEST.EK_VERSION = 55 # 55, 100

    # breakfast split selection
    _C.DATA.BREAKFAST_SPLIT = 3 # 1, 2, 3, 4

    _C.SAVE_FEATURE = False

    return _C
