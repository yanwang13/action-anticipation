#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


class MarginalCrossEntropyLoss(_Loss):
    """Implements the "Marginal Cross Entropy Loss" from the paper:
        A. Furnari, S. Battiato, G. M. Farinella (2018).
        Leveraging Uncertainty to Rethink Loss Functions and Evaluation Measures for Egocentric Action Anticipation .
        In International Workshop on Egocentric Perception, Interaction and Computing (EPIC) in conjunction with ECCV ."

        This is implementation is referenced from
        https://github.com/fpv-iplab/action-anticipation-losses/blob/master/MarginalCrossEntropyLoss/MarginalCrossEntropyLoss.py
    """
    def __init__(self, reduction='mean'):
        """Marginal Cross Entropy Loss
        Input:
            marginal_indexes: list of indexes for each of the marginal probabilities
            numclass: number of classes (e.g., number of actions)
            size_average: whether to average the losses in the batch (if False, they are summed)
            reduction: if None, all individual losses are returned, if 'mean' they are averaged, if 'sum' they are summed
            """
        super().__init__(reduction=reduction)

    def add_marginal_masks(self, marginal_indexes, numclass):
        marginal_masks = []
        for mi in marginal_indexes:
            masks = []
            for i in mi:
                masks.append(self.__indexes_to_masks(i, numclass))
            marginal_masks.append(torch.stack(masks))

        self.marginal_masks = marginal_masks

    def __indexes_to_masks(self, indexes, maxlen):
        mask = np.zeros(maxlen)
        for i in indexes:
            mask[i] = 1
        return torch.from_numpy(mask)

    def __sum_exps(self, exps):
        return exps.sum(1)

    def __build_marginal_loss(self, input, marginal_target, marginal_masks):
        mask = torch.Tensor(marginal_masks[marginal_target.cpu().data].float())
        mask = mask.to(input.device)
        exps = torch.exp(input)
        sum_all = self.__sum_exps(exps)
        sum_marginal = exps.mul(mask).sum(1)
        return torch.log(sum_all) - torch.log(sum_marginal)

    def forward(self, input, marginal_targets):
        """  input: predicted scores
             marginal_targets: list of targets of the marginal probabilities"""
        #input: bs x nc
        #marginal_targets: list of marginal targets
        loss = torch.Tensor(torch.zeros(input.shape[0])).to(input.device)
        for i, mm in enumerate(self.marginal_masks):
            l = self.__build_marginal_loss(input, marginal_targets[:, i], mm)
            loss += l

        #sum cross entropy on actions
        loss += F.cross_entropy(input, marginal_targets[:, -1], reduction='none')

        if self.reduction is not None:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        return loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "marginal_cross_entropy": MarginalCrossEntropyLoss, # VNMCE
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
