#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import pandas as pd
import os
import glob
import pickle
import lmdb
import torch
from fvcore.common.file_io import PathManager

from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
from slowfast.utils.mtl_meters import MTLTestMeter
from slowfast.utils.vna_meters import Verb_Noun_Action_TestMeter

logger = logging.get_logger(__name__)

def generate_action_feature_dic(cfg):
    logger.info('Generating action feature dic .....')
    # How many actions are there in the dataset
    action_csv_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'actions.csv')
    actions = pd.read_csv(os.path.join(action_csv_path))
    logger.info(f'Reading action info from {action_csv_path}')

    feature_base_dir = os.path.join(cfg.OUTPUT_DIR, "features")
    features_dir = os.listdir(feature_base_dir)

    features_dic = []
    for a in range(len(actions)):
        if str(a) in features_dir:
            features_path = glob.glob(os.path.join(feature_base_dir, str(a))+'/*.npy')
            features = []
            for feature_path in features_path:
                cur_feature = np.load(feature_path)
                features.append(cur_feature.reshape((cur_feature.shape[-1],)))
            features = np.stack(features, axis=0)
            logger.info(features.shape)
            action_feature = np.mean(features, axis=0)
            features_dic.append(action_feature)
        else:
            logger.info(f'set zeros for action {a}')
            features_dic.append(np.zeros(features_dic[0].shape))
    features_dic = np.stack(features_dic, axis=0)

    save_path = os.path.join(cfg.OUTPUT_DIR, 'action_dic.npy')
    np.save(save_path, features_dic)
    logger.info(f'Save action feature dic to {save_path}')
    return

def save_features_to_dir(features, labels, save_feature_dir):
    for (feature, label) in zip(features, labels):
        label = label.item()
        label_dir = os.path.join(save_feature_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        filename = str(len(glob.glob(label_dir+'/*.npy'))) + '.npy'
        np.save(os.path.join(label_dir, filename), feature.cpu().detach().numpy())
    return

def save_features_to_lmdb(features, env, base_idx):
    with env.begin(write=True) as txn:
        for idx, feature in enumerate(features):
            key = base_idx + idx
            #logger.info(key)
            #logger.info(feature.shape)
            #logger.info(feature.dtype) #float32
            txn.put(str(key).encode(), feature.tobytes())


@torch.no_grad()
def extract_features(test_loader, model, test_meter, cfg):
    """
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    save_feature_dir = os.path.join(cfg.OUTPUT_DIR, "features")
    if not os.path.exists(save_feature_dir):
        os.makedirs(save_feature_dir)

    env = lmdb.open(save_feature_dir, map_size=1099511627776)
    logger.info(f'save features to: {save_feature_dir}')
    #logger.info(f'save features to: {save_feature_dir}')
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            if isinstance(labels, (list,)):
                for i in range(len(labels)):
                    labels[i] = labels[i].cuda(non_blocking=True)
            elif isinstance(labels, (dict, )):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()
            video_idx = video_idx.cuda()
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

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            #logger.info(f'inputs length: {len(inputs)}')
            #logger.info(f'inputs.shape: {inputs[0].size()}, {inputs[1].size()}')
            preds, features = model(inputs)

            if not cfg.MULTI_TASK:
                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    preds, labels, video_idx = du.all_gather(
                        [preds, labels, video_idx]
                    )
                if cfg.NUM_GPUS:
                    preds = preds.cpu()
                    if isinstance(labels, (dict, )):
                        labels = {k: v.cpu() for k, v in labels.items()}
                    else:
                        labels = labels.cpu()
                    video_idx = video_idx.cpu()

                test_meter.iter_toc()
                # Update and log stats.
                if cfg.LOG_VERB_NOUN:
                    labels = {k: v.detach() for k, v in labels.items()}
                    test_meter.update_stats(
                        preds.detach(), labels, video_idx.detach()
                    )
                else:
                    test_meter.update_stats(
                        preds.detach(), labels['action'].detach(), video_idx.detach()
                    )
                test_meter.log_iter_stats(cur_iter)

            else: # multitask
                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    verb_preds, verb_labels, video_idx = du.all_gather(
                        [preds[0], labels['verb'], video_idx]
                    )
                    noun_preds, noun_labels = du.all_gather(
                        [preds[1], labels['noun']]
                    )

                    #meta = du.all_gather_unaligned(meta)
                    #du.all_gather_unaligned(meta)
                    #for i in range(len(meta)):
                    #    metadata['narration_id'].extend(meta[i]['narration_id'])

                if cfg.NUM_GPUS:
                    verb_preds = preds[0].cpu()
                    verb_labels = labels['verb'].cpu()
                    noun_preds = preds[1].cpu()
                    noun_labels = labels['noun'].cpu()
                    video_idx = video_idx.cpu()
                    #metatdata = meta.cpu()

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    (verb_preds.detach(), noun_preds.detach()),
                    (verb_labels.detach(), noun_labels.detach()),
                    video_idx.detach(),
                    #metadata = metatdata.detach(),
                )
                test_meter.log_iter_stats(cur_iter)

            #save_features_to_dir(features, labels['action'], save_feature_dir)
            logger.info(features.size())
            save_features_to_lmdb(features.squeeze().detach().cpu().numpy(), env, cur_iter*cfg.TEST.BATCH_SIZE)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:

        if not cfg.MULTI_TASK:
            all_preds = test_meter.video_preds.clone().detach()
            all_labels = test_meter.video_labels

            if cfg.NUM_GPUS:
                all_preds = all_preds.cpu()
                all_labels = all_labels.cpu()

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            with PathManager.open(save_path, "wb") as f:
                pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics()
    test_meter.reset()


def main():
    """
    Save the features using pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    args = parse_args()
    cfg = load_config(args, save_config=False)

    cfg.SAVE_FEATURE = True
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = False

    #cfg.TEST.NUM_ENSEMBLE_VIEWS = 10
    cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
    cfg.TEST.NUM_SPATIAL_CROPS = 1

    #cfg.DATA.TEST_CROP_SIZE = cfg.DATA.TRAIN_CROP_SIZE # expected to be 224
    cfg.DATA.RANDOM_FLIP = False
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Save features with config:")
    logger.info(cfg)
    with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    print(f"Save config to {os.path.join(cfg.OUTPUT_DIR, 'config.yaml')}")

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        if cfg.MULTI_TASK:
            test_meter = MTLTestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )
        elif cfg.LOG_VERB_NOUN:
            action_csv_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'actions.csv')
            actions = pd.read_csv(os.path.join(action_csv_path))
            logger.info(f'Reading action info from {action_csv_path}')
            vi = misc.get_marginal_indexes(actions, 'verb')
            ni = misc.get_marginal_indexes(actions, 'noun')
            logger.info(f'Get marginal indexes for verb & noun')

            test_meter = Verb_Noun_Action_TestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES[0],
                len(test_loader),
                vi,
                ni,
            )
        else:
            test_meter = TestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES[0],
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )

    # # Perform multi-view test on the entire dataset.
    extract_features(test_loader, model, test_meter, cfg)
    #generate_action_feature_dic(cfg)

if __name__ == "__main__":
    main()
