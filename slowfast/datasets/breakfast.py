import os
import csv
import glob
from PIL import Image
import numpy as np
import pandas as pd
import random
from itertools import chain as chain
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import slowfast.utils.logging as logging
from slowfast.datasets.transform import random_short_side_scale_jitter, random_crop, horizontal_flip, uniform_crop

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def extract_frame_from_dir(images_dir, start_frame, end_frame, sample_idx, num_frames, fps, num_clips):
    images_data = glob.glob(images_dir+"/*.jpg")

    images = []
    image_path = []

    for i in range(0, len(images_data)):
        if start_frame <= i <= end_frame:
            image_path.append(f'{images_dir}/{images_dir.split("/")[-1]}_{i+1:06d}.jpg')

    start_idx = 0
    delta = len(image_path) - num_frames
    if sample_idx == -1:
        start_idx = random.uniform(0, delta)
    else:
        start_idx = delta * sample_idx / num_clips
    #print('start idx: ', start_idx, ' delta: ', delta, ' sample_idx: ', sample_idx, ' num_clips: ', num_clips)
    #temporal sampling
    index = np.linspace(start_idx, len(image_path), num_frames, endpoint=False, dtype=int)
    image_path = np.take(np.array(image_path), index)

    for img in image_path:
        images.append(default_loader(img))

    return [np.array(img) for img in images]

@DATASET_REGISTRY.register()
class Breakfast(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):

        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Breakfast".format(mode)
        self.mode = mode
        self.cfg = cfg
        self._prediction_timesteps = cfg.PREDICTION_TIMESTEPS
        self._observe_time = cfg.OBSERVED_TIME
        self.fps = cfg.DATA.TARGET_FPS

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )
        logger.info("Constructing Breakfast {}...".format(mode))

        action_csv_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'actions.csv')
        self.actions = pd.read_csv(action_csv_path)
        logger.info(f'Reading action info from {action_csv_path}')

        self._construct_loader()
    def _construct_loader(self):
        """
        construct the charades label loader
        """
        path_to_labelfile = os.path.join(
            #self.cfg.DATA.PATH_TO_DATA_DIR, "breakfast_{}_01.csv".format(self.mode)
            self.cfg.DATA.PATH_TO_DATA_DIR, "bf_intention_{}.csv".format(self.mode)
        )
        assert os.path.exists(path_to_labelfile), "{} dir not found".format(
            path_to_labelfile
        )

        data_path_prefix = self.cfg.DATA.PATH_PREFIX
        assert os.path.exists(data_path_prefix), "{} dir not found".format(
            data_path_prefix
        )

        self._path_to_images = []
        self._timestamps = []
        self._labels = []
        self._spatial_temporal_idx = []

        with open(path_to_labelfile) as f:
            reader = csv.DictReader(f)
            for clip_idx, row in enumerate(reader):
                for idx in range(self._num_clips):
                    if self.cfg.PREDICT_MODE:
                        if float(row['start'])/self.fps - self._prediction_timesteps < self._observe_time:
                            continue
                        else:
                            self._timestamps.append(float(row['start'])-self._prediction_timesteps*self.fps)
                    else:
                        if float(row['end'])-float(row['start']) < self.cfg.DATA.NUM_FRAMES:
                            continue
                        else:
                            self._timestamps.append([float(row['start']), float(row['end'])])

                    self._path_to_images.append(
                        os.path.join(data_path_prefix, row["path"])
                    )

                    action = int(row['label'])-1
                    label = {'verb': self.actions.iloc[action]['verb'],
                             'noun': self.actions.iloc[action]['noun'],
                             'action': action}
                    self._labels.append(label)
                    #if self.vn_label:
                    #    vn = self.actions.iloc[action][['verb', 'noun']].values.astype(int)

                    #    if self.cfg.MODEL.LOSS_FUNC == 'marginal_cross_entropy':
                    #        self._labels.append(np.append(vn, action))
                    #    elif self.cfg.MULTI_TASK:
                    #        self._labels.append({'verb': vn[0], 'noun': vn[1]})
                    #else:
                    #    self._labels.append(action)

                    self._spatial_temporal_idx.append(idx)
        assert (
            len(self._path_to_images) > 0
        ), "Failed to load Breakfast split {} from {}".format(
            self.mode, path_to_labelfile
        )
        logger.info(
            "Constructing Breakfast dataloader (size: {}) from {}".format(
                len(self._path_to_images), path_to_labelfile
            )
        )

    def __getitem__(self, index):

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]#256
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]#320
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE#224
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        images_dir = self._path_to_images[index]

        if self.cfg.PREDICT_MODE:
            observed_end = self._timestamps[index]
            start = observed_end - self._observe_time*self.fps
            frames = extract_frame_from_dir(
                images_dir,
                start_frame = start,
                end_frame = observed_end,
                sample_idx = temporal_sample_index,
                num_frames = self.cfg.DATA.NUM_FRAMES,
                fps = self.fps,
                num_clips = self.cfg.TEST.NUM_ENSEMBLE_VIEWS
            )
        else:
            start, end = self._timestamps[index]
            frames = extract_frame_from_dir(
                images_dir,
                start,
                end,
                temporal_sample_index,
                self.cfg.DATA.NUM_FRAMES,
                fps = self.fps,
                num_clips = self.cfg.TEST.NUM_ENSEMBLE_VIEWS
            )

        frames = torch.as_tensor(np.stack(frames))
        #print(f'frame shape: {frames.shape}')


        #frames = decoder.temporal_sampling(frames, 0, len(frames), self.cfg.DATA.NUM_FRAMES)
        #frames = temporal_sampling(frames, start_inx, end_inx, num_frames) #call function from slowfast/datasets/decoder.py
        # Perform color normalization.
        #frames = frames.float()
        #frames = frames / 255.0
        #frames = frames - torch.tensor([0.485, 0.456, 0.406])
        #frames = frames / torch.tensor([0.229, 0.224, 0.225])
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )

        label = self._labels[index]
        frames = utils.pack_pathway_output(self.cfg, frames)
        return frames, label, index, {}

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_images)

