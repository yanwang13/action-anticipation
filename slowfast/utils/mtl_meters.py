#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.utils.meters import ScalarMeter

logger = logging.get_logger(__name__)

class DictScalarMeter(object):

    def __init__(self, window_size):

        self.dict_deques = {
            'verb': ScalarMeter(window_size),
            'noun': ScalarMeter(window_size),
            'action': ScalarMeter(window_size)
        }

    def reset(self):
        """
        reset each deque
        """
        self.dict_deques['verb'].reset()
        self.dict_deques['noun'].reset()
        self.dict_deques['action'].reset()

    def add_value(self, values):
        """
        Add new value to correspond deque
        Args:
            values (tuple, list): the value sequence should be (verb, noun, action)
        """
        self.dict_deques['verb'].add_value(values[0])
        self.dict_deques['noun'].add_value(values[1])
        self.dict_deques['action'].add_value(values[2])

    def get_win_median(self):
        """
        Calculate the current median value of each deque
        Return:
            win_median (dict)
        """
        win_median = {
            'verb': self.dict_deques['verb'].get_win_median(),
            'noun': self.dict_deques['noun'].get_win_median(),
            'action': self.dict_deques['action'].get_win_median(),
        }

        return win_median

    def get_win_avg(self):
        """
        Calculate the current average value of each deque
        Return:
            win_avg (dict)
        """
        win_avg = {
            'verb': self.dict_deques['verb'].get_win_avg(),
            'noun': self.dict_deques['noun'].get_win_avg(),
            'action': self.dict_deques['action'].get_win_avg(),
        }

        return win_avg

    def get_global_avg(self):
        """
        Calculate the global mean value of each deque
        Return:
            global_avg (dict)
        """
        win_avg = {
            'verb': self.dict_deques['verb'].get_global_avg(),
            'noun': self.dict_deques['noun'].get_global_avg(),
            'action': self.dict_deques['action'].get_global_avg(),
        }

        return global_avg

class MTLTestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
        multitask=False,
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        #self.ensemble_method = ensemble_method # use sum in this case
        # Initialize tensors.
        #self.video_preds = torch.zeros((num_videos, num_cls))
        self.verb_video_preds = torch.zeros((num_videos, num_cls[0]))
        self.noun_video_preds = torch.zeros((num_videos, num_cls[1]))

        self.verb_video_labels = torch.zeros((num_videos)).long()
        self.noun_video_labels = torch.zeros((num_videos)).long()
        #self.metadata = np.zeros(num_videos, dtype=object)
        self.clip_count = torch.zeros((num_videos)).long()

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.verb_video_preds.zero_()
        self.verb_video_labels.zero_()
        self.noun_video_preds.zero_()
        self.noun_video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids, metadata=None):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        assert preds[0].shape[0] == preds[1].shape[0]

        for ind in range(preds[0].shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips

            if self.verb_video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.verb_video_labels[vid_id].type(torch.FloatTensor),
                    labels[0][ind].type(torch.FloatTensor),
                )
            if self.noun_video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.noun_video_labels[vid_id].type(torch.FloatTensor),
                    labels[1][ind].type(torch.FloatTensor),
                )

            self.verb_video_labels[vid_id] = labels[0][ind]
            self.verb_video_preds[vid_id] += preds[0][ind]

            self.noun_video_labels[vid_id] = labels[1][ind]
            self.noun_video_preds[vid_id] += preds[1][ind]

            #self.metadata[vid_id] = metadata['narration_id'][ind]
            self.clip_count[vid_id] += 1


    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "test_final"}

        verb_topks = metrics.topk_accuracies(self.verb_video_preds, self.verb_video_labels, ks)
        noun_topks = metrics.topk_accuracies(self.noun_video_preds, self.noun_video_labels, ks)
        topks = metrics.multitask_topk_accuracies((self.verb_video_preds, self.noun_video_preds),
                                                  (self.verb_video_labels, self.noun_video_labels),
                                                  ks, use_cuda=False)

        assert len({len(ks), len(verb_topks)}) == 1
        assert len({len(ks), len(noun_topks)}) == 1
        assert len({len(ks), len(topks)}) == 1

        for k, verb_topk in zip(ks, verb_topks):
            stats["verb_top{}_acc".format(k)] = "{:.{prec}f}".format(verb_topk, prec=2)
        for k, noun_topk in zip(ks, noun_topks):
            stats["noun_top{}_acc".format(k)] = "{:.{prec}f}".format(noun_topk, prec=2)
        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)
        logging.log_json_stats(stats)
       # return (self.verb_video_preds.numpy().copy(), self.noun_video_preds.numpy().copy()), \
       #        (self.verb_video_labels.numpy().copy(), self.noun_video_labels.numpy().copy()), \
       #        self.metadata.copy()


class MTLTrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = DictScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = { 'verb': 0.0, 'noun': 0.0, 'action': 0.0}
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = DictScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = DictScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = { 'verb':0, 'noun': 0, 'action': 0 }
        self.num_top5_mis = { 'verb':0, 'noun': 0, 'action': 0 }
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = { 'verb': 0.0, 'noun': 0.0, 'action': 0.0}
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = { 'verb':0, 'noun': 0, 'action': 0 }
        self.num_top5_mis = { 'verb':0, 'noun': 0, 'action': 0 }
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size, int_top1_err=None, int_top5_err=None):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total['verb'] += loss[0] * mb_size
        self.loss_total['noun'] += loss[1] * mb_size
        self.loss_total['action'] += loss[2] * mb_size
        self.num_samples += mb_size

        # Current minibatch stats
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        # Aggregate stats
        self.num_top1_mis['verb'] += top1_err[0] * mb_size
        self.num_top1_mis['noun'] += top1_err[1] * mb_size
        self.num_top1_mis['action'] += top1_err[2] * mb_size
        self.num_top5_mis['verb'] += top5_err[0] * mb_size
        self.num_top5_mis['noun'] += top5_err[1] * mb_size
        self.num_top5_mis['action'] += top5_err[2] * mb_size


    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        loss_median = self.loss.get_win_median()
        top1_err_median = self.mb_top1_err.get_win_median()
        top5_err_median = self.mb_top5_err.get_win_median()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "verb_top1_err": top1_err_median['verb'],
            "verb_top5_err": top5_err_median['verb'],
            "noun_top1_err": top1_err_median['noun'],
            "noun_top5_err": top5_err_median['noun'],
            "top1_err": top1_err_median['action'],
            "top5_err": top5_err_median['action'],
            "loss_verb": loss_median['verb'],
            "loss_noun": loss_median['noun'],
            "loss": loss_median['action'],
            "lr": self.lr,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch, writer=None):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        }
        stats["verb_top1_err"] = self.num_top1_mis['verb'] / self.num_samples
        stats["verb_top5_err"] = self.num_top5_mis['verb'] / self.num_samples

        stats["noun_top1_err"] = self.num_top1_mis['noun'] / self.num_samples
        stats["noun_top5_err"] = self.num_top5_mis['noun'] / self.num_samples

        stats["top1_err"] = self.num_top1_mis['action'] / self.num_samples
        stats["top5_err"] = self.num_top5_mis['action'] / self.num_samples

        stats["loss_verb"] = self.loss_total['verb'] / self.num_samples # avg_loss
        stats["loss_noun"] = self.loss_total['noun'] / self.num_samples # avg_loss
        stats["loss"] = self.loss_total['action'] / self.num_samples # avg_loss

        logging.log_json_stats(stats)

        if writer is not None:
            writer.add_scalars(
                {
                    "Train/Avg_loss_verb": stats["loss_verb"],
                    "Train/Avg_loss_noun": stats["loss_noun"],
                    "Train/Avg_loss": stats["loss"],
                    "Train/Epoch_Verb_Top1_err": stats["verb_top1_err"],
                    "Train/Epoch_Verb_Top5_err": stats["verb_top5_err"],
                    "Train/Epoch_Noun_Top1_err": stats["noun_top1_err"],
                    "Train/Epoch_Noun_Top5_err": stats["noun_top5_err"],
                    "Train/Epoch_Top1_err": stats["top1_err"],
                    "Train/Epoch_Top5_err": stats["top5_err"],
                    "Train/Epoch_lr": self.lr,
                },
                global_step = cur_epoch,
            )


class MTLValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = DictScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = DictScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = {'verb':100.0, 'noun':100.0, 'action':100.0}
        self.min_top5_err = {'verb':100.0, 'noun':100.0, 'action':100.0}
        # Number of misclassified examples.
        self.num_top1_mis = { 'verb':0, 'noun': 0, 'action': 0 }
        self.num_top5_mis = { 'verb':0, 'noun': 0, 'action': 0 }
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

        self.save_ckpt_thres = cfg.TRAIN.SAVE_CKPT_THRES

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = { 'verb':0, 'noun': 0, 'action': 0 }
        self.num_top5_mis = { 'verb':0, 'noun': 0, 'action': 0 }
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, mb_size, int_top1_err=None, int_top5_err=None):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        # Aggregate stats
        self.num_top1_mis['verb'] += top1_err[0] * mb_size
        self.num_top5_mis['verb'] += top5_err[0] * mb_size
        self.num_top1_mis['noun'] += top1_err[1] * mb_size
        self.num_top5_mis['noun'] += top5_err[1] * mb_size
        self.num_top1_mis['action'] += top1_err[2] * mb_size
        self.num_top5_mis['action'] += top5_err[2] * mb_size
        self.num_samples += mb_size

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        top1_err_median = self.mb_top1_err.get_win_median()
        top5_err_median = self.mb_top5_err.get_win_median()
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "verb_top1_err": top1_err_median['verb'],
            "verb_top5_err": top5_err_median['verb'],
            "noun_top1_err": top1_err_median['noun'],
            "noun_top5_err": top5_err_median['noun'],
            "top1_err": top1_err_median['action'],
            "top5_err": top5_err_median['action'],
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch, writer=None):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        verb_top1_err = self.num_top1_mis['verb'] / self.num_samples
        verb_top5_err = self.num_top5_mis['verb'] / self.num_samples

        noun_top1_err = self.num_top1_mis['noun'] / self.num_samples
        noun_top5_err = self.num_top5_mis['noun'] / self.num_samples

        top1_err = self.num_top1_mis['action'] / self.num_samples
        top5_err = self.num_top5_mis['action'] / self.num_samples

        #self.min_top1_err = min(self.min_top1_err, top1_err)
        if top1_err < self.min_top1_err['action']:
            self.min_top1_err['action'] = top1_err
            save_ckpts = True if self.min_top1_err['action'] < self.save_ckpt_thres else False
        else:
            save_ckpts = False
        self.min_top5_err['action'] = min(self.min_top5_err['action'], top5_err)

        self.min_top1_err['verb'] = min(self.min_top1_err['verb'], verb_top1_err)
        self.min_top5_err['verb'] = min(self.min_top5_err['verb'], verb_top5_err)
        self.min_top1_err['noun'] = min(self.min_top1_err['noun'], noun_top1_err)
        self.min_top5_err['noun'] = min(self.min_top5_err['noun'], noun_top5_err)

        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
            "verb_top1_err": verb_top1_err,
            "verb_top5_err": verb_top5_err,
            "noun_top1_err": noun_top1_err,
            "noun_top5_err": noun_top5_err,
            "top1_err": top1_err,
            "top5_err": top5_err,
            "min_verb_top1_err": self.min_top1_err['verb'],
            "min_verb_top5_err": self.min_top5_err['verb'],
            "min_noun_top1_err": self.min_top1_err['noun'],
            "min_noun_top5_err": self.min_top5_err['noun'],
            "min_top1_err": self.min_top1_err['action'],
            "min_top5_err": self.min_top5_err['action'],
        }

#        stats["verb_top1_err"] = verb_top1_err
#        stats["verb_top5_err"] = verb_top5_err

 #       stats["noun_top1_err"] = noun_top1_err
 #       stats["noun_top5_err"] = noun_top5_err

#        stats["top1_err"] = top1_err
#        stats["top5_err"] = top5_err

#        stats["min_verb_top1_err"] = self.min_top1_err['verb']
#        stats["min_verb_top5_err"] = self.min_top5_err['verb']
#        stats["min_noun_top1_err"] = self.min_top1_err['noun']
#        stats["min_noun_top5_err"] = self.min_top5_err['noun']
#        stats["min_top1_err"] = self.min_top1_err['action']
#        stats["min_top5_err"] = self.min_top5_err['acion']

        if writer is not None:
            writer.add_scalars(
                {
                    "Val/Epoch_Verb_Top1_err": stats["verb_top1_err"],
                    "Val/Epoch_Verb_Top5_err": stats["verb_top5_err"],
                    "Val/Epoch_Noun_Top1_err": stats["noun_top1_err"],
                    "Val/Epoch_Noun_Top5_err": stats["noun_top5_err"],
                    "Val/Epoch_Top1_err": stats["top1_err"],
                    "Val/Epoch_Top5_err": stats["top5_err"],
                    "Val/Min_Verb_Top1_err": stats["min_verb_top1_err"],
                    "Val/Min_Verb_Top5_err": stats["min_verb_top5_err"],
                    "Val/Min_Noun_Top1_err": stats["min_noun_top1_err"],
                    "Val/Min_Noun_Top5_err": stats["min_noun_top5_err"],
                    "Val/Min_Top1_err": stats["min_top1_err"],
                    "Val/Min_Top5_err": stats["min_top5_err"],
                },
                global_step = cur_epoch,
            )

        logging.log_json_stats(stats)
        return save_ckpts, self.min_top1_err['action']

