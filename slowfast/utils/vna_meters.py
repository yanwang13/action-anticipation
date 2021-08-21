
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

class Verb_Noun_Action_ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg, vi, ni):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # For each verb/noun retrieve the list of actions that containes the corresponding verb/noun
        self.vi = vi
        self.ni = ni
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        #self.min_top1_err = 100.0
        #self.min_top5_err = 100.0
        self.min_top1_err = {'verb':100.0, 'noun':100.0, 'action':100.0}
        self.min_top5_err = {'verb':100.0, 'noun':100.0, 'action':100.0}
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.all_verb_labels = []
        self.all_noun_labels = []

        self.save_ckpt_thres = cfg.TRAIN.SAVE_CKPT_THRES

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.all_verb_labels = []
        self.all_noun_labels = []

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
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
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
        self.all_labels.append(labels['action'])
        self.all_verb_labels.append(labels['verb'])
        self.all_noun_labels.append(labels['noun'])

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
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch, writer=None):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        }

        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        #self.min_top1_err = min(self.min_top1_err, top1_err)
        if top1_err < self.min_top1_err['action']:
            self.min_top1_err['action'] = top1_err
            save_ckpts = True if self.min_top1_err['action'] < self.save_ckpt_thres else False
            #logger.info(f"save_ckpts:{save_ckpts}, min_top1_err:{self.min_top1_err}, save_ckpt_thres:{self.save_ckpt_thres}")
        else:
            save_ckpts = False
        self.min_top5_err['action'] = min(self.min_top5_err['action'], top5_err)

        stats["top1_err"] = top1_err
        stats["top5_err"] = top5_err
        stats["min_top1_err"] = self.min_top1_err['action']
        stats["min_top5_err"] = self.min_top5_err['action']

        # calcuate verb/noun errors
        verb_top1_err, verb_top5_err, noun_top1_err, noun_top5_err = self.marginal_topk_errors()
        self.min_top1_err['verb'] = min(self.min_top1_err['verb'], verb_top1_err)
        self.min_top5_err['verb'] = min(self.min_top5_err['verb'], verb_top5_err)
        self.min_top1_err['noun'] = min(self.min_top1_err['noun'], noun_top1_err)
        self.min_top5_err['noun'] = min(self.min_top5_err['noun'], noun_top5_err)

        stats["verb_top1_err"] = verb_top1_err
        stats["verb_top5_err"] = verb_top5_err
        stats["noun_top1_err"] = noun_top1_err
        stats["noun_top5_err"] = noun_top5_err

        stats["min_verb_top1_err"] = self.min_top1_err['verb']
        stats["min_verb_top5_err"] = self.min_top5_err['verb']
        stats["min_noun_top1_err"] = self.min_top1_err['noun']
        stats["min_noun_top5_err"] = self.min_top5_err['noun']

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

    def cat_all(self, x):

        if isinstance(x, (list, )):
            return torch.cat(x, dim=0)
        else:
            raise NotImplementedError
        #if isinstance(self.all_preds, (list, )):
        #    probs = torch.cat(self.all_preds, dim=0)
        #else:
        #    raise NotImplementedError

    def marginal_topk_errors(self):

        probs = self.cat_all(self.all_preds)

        verb_preds = misc.marginalize(probs, self.vi)
        noun_preds = misc.marginalize(probs, self.ni)

        verb_labels = self.cat_all(self.all_verb_labels)
        noun_labels = self.cat_all(self.all_noun_labels)

        #logger.info(f'verb_labels shape: {verb_labels.size()}, device: {verb_labels.device}')
        #logger.info(f'noun_labels shape: {noun_labels.size()}, device: {verb_labels.device}')
        #logger.info(f'verb_preds shape: {verb_preds.size()}, device: {verb_preds.device} ')
        #logger.info(f'noun_preds shape: {noun_preds.size()}, device: {noun_preds.device} ')
        # Compute the verb accuracies
        verb_topks_correct = metrics.topks_correct(verb_preds, verb_labels, (1, 5))
        verb_top1_err, verb_top5_err = [
            (1.0 - x / verb_preds.size(0)) * 100.0 for x in verb_topks_correct
        ]

        # Compute the noun accuracies
        noun_topks_correct = metrics.topks_correct(noun_preds, noun_labels, (1, 5))
        noun_top1_err, noun_top5_err = [
            (1.0 - x / noun_preds.size(0)) * 100.0 for x in noun_topks_correct
        ]

        # Gather all the predictions across all the devices.
        if self._cfg.NUM_GPUS > 1:
            verb_top1_err, verb_top5_err = du.all_reduce([verb_top1_err, verb_top5_err])
            noun_top1_err, noun_top5_err = du.all_reduce([noun_top1_err, noun_top5_err])

        # Copy the errors from GPU to CPU (sync point).
        verb_top1_err, verb_top5_err = verb_top1_err.item(), verb_top5_err.item()
        noun_top1_err, noun_top5_err = noun_top1_err.item(), noun_top5_err.item()

        return verb_top1_err, verb_top5_err, noun_top1_err, noun_top5_err


class Verb_Noun_Action_TestMeter(object):
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
        vi,
        ni,
        #ensemble_method="sum",
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
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        #self.ensemble_method = ensemble_method
        # For each verb/noun retrieve the list of actions that containes the corresponding verb/noun
        self.vi = vi
        self.ni = ni
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))

        self.video_labels = torch.zeros((num_videos)).long()

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
        self.video_preds.zero_()
        self.video_labels.zero_()

        self.verb_video_labels.zero_()
        self.noun_video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids, int_preds=None, int_labels=None):
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
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips

            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels['action'][ind].type(torch.FloatTensor),
                )
            if self.verb_video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.verb_video_labels[vid_id].type(torch.FloatTensor),
                    labels['verb'][ind].type(torch.FloatTensor),
                )
            if self.noun_video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.noun_video_labels[vid_id].type(torch.FloatTensor),
                    labels['noun'][ind].type(torch.FloatTensor),
                )

            self.video_labels[vid_id] = labels['action'][ind]
            self.video_preds[vid_id] += preds[ind]

            self.verb_video_labels[vid_id] = labels['verb'][ind]
            self.noun_video_labels[vid_id] = labels['noun'][ind]
            #import pdb
            #pdb.set_trace()

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

    def finalize_metrics(self, ks=(1, 3, 5)):
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

        #num_topks_correct = metrics.topks_correct(
        #    self.video_preds, self.video_labels, ks
        #)
        #topks = [
        #    (x / self.video_preds.size(0)) * 100.0
        #    for x in num_topks_correct
        #]
        # Compute the action accuracies
        topks = metrics.topk_accuracies(self.video_preds, self.video_labels, ks)

        verb_preds = misc.marginalize(self.video_preds, self.vi)
        noun_preds = misc.marginalize(self.video_preds, self.ni)

        # Compute the verb accuracies
        verb_topks = metrics.topk_accuracies(verb_preds, self.verb_video_labels, ks)
        # Compute the noun accuracies
        noun_topks = metrics.topk_accuracies(noun_preds, self.noun_video_labels, ks)


        #logger.info(f'verb_labels shape: {self.verb_video_labels.size()}, device: {self.verb_video_labels.device}')
        #logger.info(f'noun_labels shape: {self.noun_video_labels.size()}, device: {self.noun_video_labels.device}')
        #logger.info(f'verb_preds shape: {verb_preds.size()}, device: {verb_preds.device} ')
        #logger.info(f'noun_preds shape: {noun_preds.size()}, device: {noun_preds.device} ')

        assert len({len(ks), len(verb_topks)}) == 1
        assert len({len(ks), len(noun_topks)}) == 1
        assert len({len(ks), len(topks)}) == 1

        for k, verb_topk in zip(ks, verb_topks):
            stats["verb_top{}_acc".format(k)] = "{:.{prec}f}".format(verb_topk, prec=2)
        for k, noun_topk in zip(ks, noun_topks):
            stats["noun_top{}_acc".format(k)] = "{:.{prec}f}".format(noun_topk, prec=2)
        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)

        # Compute the recalls
        for k in ks:
            action_recall = metrics.topk_recall(self.video_preds, self.video_labels, k)
            stats["top{}_recall".format(k)] = "{:.{prec}f}".format(action_recall, prec=2)

            verb_recall = metrics.topk_recall(verb_preds, self.verb_video_labels, k)
            stats["verb_top{}_recall".format(k)] = "{:.{prec}f}".format(verb_recall, prec=2)

            noun_recall = metrics.topk_recall(noun_preds, self.noun_video_labels, k)
            stats["noun_top{}_recall".format(k)] = "{:.{prec}f}".format(noun_recall, prec=2)

        logging.log_json_stats(stats)
