#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import os
import numpy as np
import pandas as pd
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model, set_finetune_mode
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter
from slowfast.utils.mtl_meters import MTLTrainMeter, MTLValMeter, Recognition_MTLTrainMeter, Recognition_MTLValMeter
from slowfast.utils.vna_meters import Verb_Noun_Action_ValMeter
from slowfast.utils.multigrid import MultigridSchedule
from slowfast.models.uncertainty_loss import Uncertaintyloss

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None, criterion=None, mtl_meter=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # check label type goes to right cuda execution
            if isinstance(labels, (list,)):
                for i in range(len(labels)):
                    labels[i] = labels[i].cuda(non_blocking=True)
            elif isinstance(labels, (dict, )):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()

            if cfg.DETECTION.ENABLE:
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        #logger.info(f'inputs length: {len(inputs)}')
        #logger.info(f'inputs.shape: {inputs[0].size()}, {inputs[1].size()}')
        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

        else:
            # Perform the forward pass.
            preds = model(inputs)
        # Explicitly declare reduction to mean.
        if criterion is None:
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # Compute the loss.
        # print(f'preds shape: {preds.shape}, labels shape: {labels.shape}')
        if cfg.MULTI_TASK:
            if cfg.UNCERTAINTY: # Use uncertainty loss
                loss, loss_verb, loss_noun = criterion(preds, [labels['verb'], labels['noun']])
            else:
                loss_verb = loss_fun(preds[0], labels['verb'])
                loss_noun = loss_fun(preds[1], labels['noun'])
                loss = 0.5*(loss_verb+loss_noun)
        elif cfg.RECOGNITION_MTL:
            if cfg.UNCERTAINTY: # Use uncertainty loss
                loss, loss_verb, loss_noun = criterion(preds, [labels['action'], labels['observed_action']])
            else:
                loss_predict = loss_fun(preds[0], labels['action'])
                #logger.info(f'before filter: {preds[1].size()}')
                #logger.info(labels['observed_action'])
                recog_preds, recog_labels = misc.filter_none_observed_action_samples(preds[1], labels['observed_action'])
                #logger.info(f'after filter: {recog_preds.size()}')
                #logger.info(recog_labels)
                loss_recog = loss_fun(recog_preds, recog_labels)
                loss = 0.5*(loss_predict+loss_recog)
                preds = preds[0]
                #if labels['observed_action'].item() != -1:
                #    loss_recog = loss_fun(preds[1], labels['observed_action'])
                #    loss = 0.5*(loss_predict+loss_recog)
                #    log_observed_action = True
                #    preds = preds[0]
                #    recog_preds = preds[1]
                #else:
                #    loss = loss_predict
                #    log_observed_action = False
        elif cfg.CAUSAL_INTERVENTION.ENABLE:
            if cfg.CAUSAL_INTERVENTION.CAUSAL_ONLY:
                loss = loss_fun(preds, labels['action'])
            else:
                loss = loss_fun(preds[0], labels['action']) + loss_fun(preds[1], labels['action'])
                preds = preds[0]
        elif cfg.MODEL.LOSS_FUNC == 'marginal_cross_entropy':
            loss = criterion(preds, torch.stack([labels['verb'], labels['noun'], labels['action']], 1))
        else:
            loss = loss_fun(preds, labels['action'])

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                if cfg.MULTI_TASK:
                    # compute verb accuracies
                    verb_topks_correct = metrics.topks_correct(preds[0], labels['verb'], (1, 5))
                    verb_top1_err, verb_top5_err = [
                        (1.0 - x / preds[0].size(0)) * 100.0 for x in verb_topks_correct
                    ]
                    # verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels['verb'], (1, 5))

                    # compute noun accuracies
                    noun_topks_correct = metrics.topks_correct(preds[1], labels['noun'], (1, 5))
                    noun_top1_err, noun_top5_err = [
                        (1.0 - x / preds[1].size(0)) * 100.0 for x in noun_topks_correct
                    ]

                    # compute action accuracies
                    action_topks_correct = metrics.multitask_topks_correct((preds[0], preds[1]),
                                                                           (labels['verb'], labels['noun']),
                                                                           (1, 5))
                    top1_err, top5_err = [
                        (1.0 - x / preds[1].size(0)) * 100.0 for x in action_topks_correct
                    ]

                    # Gather all the predictions across all the devices.
                    if cfg.NUM_GPUS > 1:
                        loss_verb, verb_top1_err, verb_top5_err = du.all_reduce(
                            [loss_verb, verb_top1_err, verb_top5_err]
                        )
                        loss_noun, noun_top1_err, noun_top5_err = du.all_reduce(
                            [loss_noun, noun_top1_err, noun_top5_err]
                        )
                        loss, top1_err, top5_err = du.all_reduce(
                            [loss, top1_err, top5_err]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    loss_verb, verb_top1_err, verb_top5_err = (
                        loss_verb.item(),
                        verb_top1_err.item(),
                        verb_top5_err.item(),
                    )
                    loss_noun, noun_top1_err, noun_top5_err = (
                        loss_noun.item(),
                        noun_top1_err.item(),
                        noun_top5_err.item(),
                    )
                    loss, top1_err, top5_err = (
                        loss.item(),
                        top1_err.item(),
                        top5_err.item(),
                    )

                else:
                    num_topks_correct = metrics.topks_correct(preds, labels['action'], (1, 5))
                    top1_err, top5_err = [
                        (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                    ]

                    # Gather all the predictions across all the devices.
                    if cfg.NUM_GPUS > 1:
                        loss, top1_err, top5_err = du.all_reduce(
                            [loss, top1_err, top5_err]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    loss, top1_err, top5_err = (
                        loss.item(),
                        top1_err.item(),
                        top5_err.item(),
                    )

                if cfg.RECOGNITION_MTL: # log observed action acc
                    recog_topks_correct = metrics.topks_correct(recog_preds, recog_labels, (1, 5))
                    recog_top1_err, recog_top5_err = [
                        (1.0 - x / recog_preds.size(0)) * 100.0 for x in recog_topks_correct
                    ]

                    # Gather all the predictions across all the devices.
                    if cfg.NUM_GPUS > 1:
                        loss_recog, recog_top1_err, recog_top5_err = du.all_reduce(
                            [loss_recog, recog_top1_err, recog_top5_err]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    loss_recog, recog_top1_err, recog_top5_err = (
                        loss_recog.item(),
                        recog_top1_err.item(),
                        recog_top5_err.item(),
                    )

            train_meter.iter_toc()
            # Update and log stats.
            if cfg.MULTI_TASK:
                train_meter.update_stats(
                    (verb_top1_err, noun_top1_err, top1_err),
                    (verb_top5_err, noun_top5_err, top5_err),
                    (loss_verb, loss_noun, loss),
                    lr,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
            else:
                train_meter.update_stats(
                    top1_err,
                    top5_err,
                    loss,
                    lr,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
            if mtl_meter is not None:
                mtl_meter.iter_toc()
                mtl_meter.update_stats(
                    recog_top1_err,
                    recog_top5_err,
                    loss_recog,
                    recog_labels.size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                mtl_meter.log_iter_stats(cur_epoch, cur_iter)
                mtl_meter.iter_tic()
            # write to tensorboard format if available.
            if writer is not None:
                if cfg.DATA.MULTI_LABEL:
                    writer.add_scalars(
                        {
                            "Train/loss": loss,
                            "Train/lr": lr,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    if cfg.UNCERTAINTY and writer is not None:
        verb_weight, noun_weight = criterion.get_weights()
        writer.add_scalars(
            {
                "Train/verb_weight": verb_weight,
                "Train/noun_weight": noun_weight,
            },
            global_step = cur_epoch,
        )
    train_meter.log_epoch_stats(cur_epoch, writer)
    train_meter.reset()
    if mtl_meter is not None:
        mtl_meter.log_epoch_stats(cur_epoch, writer)
        mtl_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None, mtl_meter=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            if isinstance(labels, (list,)):
                for i in range(len(labels)):
                    labels[i] = labels[i].cuda(non_blocking=True)
            elif isinstance(labels, (dict, )):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()

            if cfg.DETECTION.ENABLE:
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            preds = model(inputs)
            if cfg.CAUSAL_INTERVENTION.ENABLE:
                if cfg.CAUSAL_INTERVENTION.CAUSAL_ONLY == False:
                    preds = preds[0]
            if cfg.RECOGNITION_MTL:
                recog_preds = preds[1]
                recog_preds, recog_labels = misc.filter_none_observed_action_samples(preds[1], labels['observed_action'])
                preds = preds[0]

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                if cfg.MULTI_TASK:

                    # Compute the verb accuracies
                    verb_topks_correct = metrics.topks_correct(preds[0], labels['verb'], (1, 5))
                    verb_top1_err, verb_top5_err = [
                        (1.0 - x / preds[0].size(0)) * 100.0 for x in verb_topks_correct
                    ]

                    # Compute the noun accuracies
                    noun_topks_correct = metrics.topks_correct(preds[1], labels['noun'], (1, 5))
                    noun_top1_err, noun_top5_err = [
                        (1.0 - x / preds[1].size(0)) * 100.0 for x in noun_topks_correct
                    ]

                    # Compute the acion accuracies
                    action_topks_correct = metrics.multitask_topks_correct((preds[0], preds[1]),
                                                                           (labels['verb'], labels['noun']),
                                                                           (1, 5))
                    top1_err, top5_err = [
                        (1.0 - x / preds[1].size(0)) * 100.0 for x in action_topks_correct
                    ]

                    # Gather all the predictions across all the devices.
                    if cfg.NUM_GPUS > 1:
                        verb_top1_err, verb_top5_err = du.all_reduce([verb_top1_err, verb_top5_err])
                        noun_top1_err, noun_top5_err = du.all_reduce([noun_top1_err, noun_top5_err])
                        top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                    # Copy the errors from GPU to CPU (sync point).
                    verb_top1_err, verb_top5_err = verb_top1_err.item(), verb_top5_err.item()
                    noun_top1_err, noun_top5_err = noun_top1_err.item(), noun_top5_err.item()

                else: # model with only one output
                    num_topks_correct = metrics.topks_correct(preds, labels['action'], (1, 5))
                    #num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                    # Compute the errors across the GPUs
                    top1_err, top5_err = [
                        (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                    ]

                    # Gather all the predictions across all the devices.
                    if cfg.NUM_GPUS > 1:
                        top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                    # Copy the errors from GPU to CPU (sync point).
                    top1_err, top5_err = top1_err.item(), top5_err.item()

                if cfg.RECOGNITION_MTL: #and log_observed_action:
                    recog_topks_correct = metrics.topks_correct(recog_preds, recog_labels, (1, 5))
                    recog_top1_err, recog_top5_err = [
                        (1.0 - x / recog_preds.size(0)) * 100.0 for x in recog_topks_correct
                    ]

                    # Gather all the predictions across all the devices.
                    if cfg.NUM_GPUS > 1:
                        recog_top1_err, recog_top5_err = du.all_reduce(
                            [recog_top1_err, recog_top5_err]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    recog_top1_err, recog_top5_err = (
                        recog_top1_err.item(),
                        recog_top5_err.item(),
                    )

            val_meter.iter_toc()
            # Update and log stats.
            if cfg.MULTI_TASK:
                val_meter.update_stats(
                    (verb_top1_err, noun_top1_err, top1_err),
                    (verb_top5_err, noun_top5_err, top5_err),
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
            else:
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )

            if mtl_meter is not None:
                mtl_meter.iter_toc()
                mtl_meter.update_stats(
                    recog_top1_err,
                    recog_top5_err,
                    recog_labels.size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                mtl_meter.log_iter_stats(cur_epoch, cur_iter)
                mtl_meter.iter_tic()

            #if cfg.MULTI_TASK:
            #    preds = preds[0]
            #    labels = labels[0]
            if not cfg.MULTI_TASK:
                if cfg.LOG_VERB_NOUN:
                    val_meter.update_predictions(preds, labels)
                else:
                    val_meter.update_predictions(preds, labels['action'])
            else:
                pass

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    save_current_ckpts, best_err = val_meter.log_epoch_stats(cur_epoch, writer)
    if mtl_meter is not None:
        mtl_meter.log_epoch_stats(cur_epoch, writer)
        mtl_meter.reset()

    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else: # plot confusion matrix
            if not cfg.MULTI_TASK:
                all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
                all_labels = [
                    label.clone().detach() for label in val_meter.all_labels
                ]
                if cfg.NUM_GPUS:
                    all_preds = [pred.cpu() for pred in all_preds]
                    all_labels = [label.cpu() for label in all_labels]
                writer.plot_eval(
                    preds=all_preds, labels=all_labels, global_step=cur_epoch
                )
            else:
                pass
                #if preds more than one
                #for zip(preds, labels)
                #writer.plot_eval(preds=all_preds[i], labels=all_labels[i])

    val_meter.reset()

    return save_current_ckpts, best_err


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, _, _, _ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg=cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # when using uncertainty loss build the module first,
    # than add it to the optimizer
    if cfg.UNCERTAINTY:
        #criterion = Uncertaintyloss().to(torch.cuda.current_device())
        logger.info(f'Building Uncertaintyloss')
        if cfg.NUM_GPUS <= 1:
            criterion = Uncertaintyloss().to(next(model.parameters()).device)
        else:
            raise ValueError
        optimizer = optim.construct_optimizer(model, criterion, cfg)
    elif cfg.MODEL.LOSS_FUNC == 'marginal_cross_entropy':
        logger.info(f'Building Verb Noun Marginal Cross Entropy Loss')
        criterion = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        actions = pd.read_csv(os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'actions.csv'))
        if cfg.TEST.DATASET=="breakfast":
            vi = misc.get_marginal_indexes(actions, 'verb')
            ni = misc.get_marginal_indexes(actions, 'noun')
        else:
            vi = misc.get_marginal_indexes(actions, 'verb_class')
            ni = misc.get_marginal_indexes(actions, 'noun_class')
        criterion.add_marginal_masks([vi, ni], cfg.MODEL.NUM_CLASSES[0])

        optimizer = optim.construct_optimizer(model, cfg=cfg)
    else:
        criterion = None
        # Construct the optimizer.
        optimizer = optim.construct_optimizer(model, cfg=cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        if cfg.MULTI_TASK:
            train_meter = MTLTrainMeter(len(train_loader), cfg)
            val_meter = MTLValMeter(len(val_loader), cfg)
        elif cfg.LOG_VERB_NOUN:
            action_csv_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'actions.csv')
            actions = pd.read_csv(os.path.join(action_csv_path))
            logger.info(f'Reading action info from {action_csv_path}')
            if cfg.TEST.DATASET=="breakfast":
                vi = misc.get_marginal_indexes(actions, 'verb')
                ni = misc.get_marginal_indexes(actions, 'noun')
            else:
                vi = misc.get_marginal_indexes(actions, 'verb_class')
                ni = misc.get_marginal_indexes(actions, 'noun_class')

            train_meter = TrainMeter(len(train_loader), cfg)
            val_meter = Verb_Noun_Action_ValMeter(len(val_loader), cfg, vi, ni)
        else:
            train_meter = TrainMeter(len(train_loader), cfg)
            val_meter = ValMeter(len(val_loader), cfg)
    if cfg.RECOGNITION_MTL:
        mtl_train_meter = Recognition_MTLTrainMeter(len(train_loader), cfg)
        mtl_val_meter = Recognition_MTLValMeter(len(val_loader), cfg)
    else:
        mtl_train_meter = None
        mtl_val_meter = None

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # set finetune params if needed
        if cfg.TRAIN.FINETUNE:
            if cur_epoch < cfg.TRAIN.FINETUNE_EPOCH:
                #set_finetune_mode(model, 'freeze_s1_s2_s3')
                #set_finetune_mode(model, 's5_fc')
                set_finetune_mode(model, 'fc')
            else:
                set_finetune_mode(model, 'all')

        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer, criterion=criterion, mtl_meter=mtl_train_meter
        )

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        ):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        ):
            save_current_ckpts, best_err = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer, mtl_meter=mtl_val_meter)

            # Save the checkpoint of the best result
            if save_current_ckpts:
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, best_err)

    if writer is not None:
        writer.close()
