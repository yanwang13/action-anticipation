"""
 This is implementation is referenced from
 https://github.com/epic-kitchens/epic-kitchens-slowfast/blob/master/slowfast/datasets/epickitchens.py
"""
import os
import pandas as pd
import torch
import torch.utils.data

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY
from .epickitchens_record import EpicKitchensVideoRecord

from . import transform as transform
from . import utils as utils
#from .frame_loader import pack_frames_to_video_clip
from .decoder import get_start_end_idx

logger = logging.get_logger(__name__)

def temporal_sampling(num_frames, start_idx, end_idx, num_samples, start_frame=0):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        num_frames (int): number of frames of the trimmed action clip
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
        start_frame (int): starting frame of the action clip in the untrimmed video
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, num_frames - 1).long()
    return start_frame + index

def pack_frames_to_video_clip(cfg, ek_version,mode, video_record, temporal_sample_index, target_fps=60):
    # Load video by loading its extracted frames
    #path_to_video = '{}/{}/rgb_frames/{}'.format(cfg.EPICKITCHENS.VISUAL_DATA_DIR,
    #                                             video_record.participant,
    #                                             video_record.untrimmed_video_name)
    #/work/r08944003/EPIC_KITCHENS_55/frames_rgb_flow/rgb/train/P01/P01_01

    if ek_version == "55":
        mode = mode if mode=='test' else 'train'
        path_to_video = f'{cfg.DATA.PATH_PREFIX}/{mode}/{video_record.participant}/{video_record.untrimmed_video_name}'
    else:
        raise NotImplementedError()

    img_tmpl = "frame_{:010d}.jpg"
    fps, sampling_rate, num_samples = video_record.fps, cfg.DATA.SAMPLING_RATE, cfg.DATA.NUM_FRAMES
    start_idx, end_idx = get_start_end_idx(
        video_record.num_frames,
        num_samples * sampling_rate * fps / target_fps,
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
    )
    start_idx, end_idx = start_idx + 1, end_idx + 1
    frame_idx = temporal_sampling(video_record.num_frames,
                                  start_idx, end_idx, num_samples,
                                  start_frame=video_record.start_frame)
    img_paths = [os.path.join(path_to_video, img_tmpl.format(idx.item())) for idx in frame_idx]
    frames = utils.retry_load_images(img_paths)
    return frames

@DATASET_REGISTRY.register()
class Epickitchens(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.cfg = cfg
        self.mode = mode
        self.target_fps = 60

        # for action anticipation task
        self._prediction_timesteps = cfg.PREDICTION_TIMESTEPS
        self._observed_time = cfg.OBSERVED_TIME
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.

        assert (
            cfg.TRAIN.EK_VERSION == cfg.TEST.EK_VERSION
        ), "Epic-Kitchen train version:{} and test version:{} should be the same.".format(
            cfg.TRAIN.EK_VERSION, cfg.TEST.EK_VERSION
        )

        assert cfg.TRAIN.EK_VERSION in [55, 100], "Epic-Kitchen version {} not supported.".format(cfg.TRAIN.EK_VERSION)
        self.ek_version = str(cfg.TRAIN.EK_VERSION)

        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing EPIC-KITCHENS {}...".format(mode))

        #if cfg.MODEL.LOSS_FUNC == 'marginal_cross_entropy' or cfg.MULTI_TASK:
        #    self.vn_label = True
        #else:
        #    self.vn_label = False
        #    action_csv_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'actions.csv')
        #    self.actions = pd.read_csv(os.path.join(action_csv_path))
        #    logger.info(f'Reading action info from {action_csv_path}')

        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train+val":
            path_annotations_pickle = [os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, file)
                                       for file in [f'ek{self.ek_version}_train.pkl', f'ek{self.ek_version}_val.pkl']]
        else:
            path_annotations_pickle = [os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'ek{self.ek_version}_{self.mode}.pkl')]

        #if self.mode == "train":
        #    path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TRAIN_LIST)]
        #    _C.EPICKITCHENS.TRAIN_LIST = "EPIC_100_train.pkl"
        #elif self.mode == "val":
        #    path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.VAL_LIST)]
        #    _C.EPICKITCHENS.VAL_LIST = "EPIC_100_validation.pkl"
        #elif self.mode == "test":
        #    path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]
        #    _C.EPICKITCHENS.TEST_LIST = "EPIC_100_validation.pkl"
        #else:
        #    path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
        #                               for file in [self.cfg.EPICKITCHENS.TRAIN_LIST, self.cfg.EPICKITCHENS.VAL_LIST]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )
            logger.info(f'Reading annotations from {file}')

        # read verb noun action pair csv
        action_csv_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'actions.csv')
        self.actions = pd.read_csv(action_csv_path)
        logger.info(f'Reading action info from {action_csv_path}')

        if self.ek_version=="55":
            self.video_info_csv = pd.read_csv(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'EPIC_video_info.csv'))
            self.video_id_to_info_idx_dict = {video_id: i for i, video_id in enumerate(self.video_info_csv['video'])}

        self._video_records = []
        self._spatial_temporal_idx = []
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                for idx in range(self._num_clips):

                    if self.ek_version=="55": # set the fps by the video info csv
                        fps = int(round(self.video_info_csv['fps'][self.video_id_to_info_idx_dict[tup[1]['video_id']]]))
                        record = EpicKitchensVideoRecord(tup, fps)
                    else:
                        record = EpicKitchensVideoRecord(tup)
                    # Check if the observed time is enough before adding to dataset when doing anticipation
                    if self.cfg.PREDICT_MODE:
                        if (record.start_time - self._prediction_timesteps) < self._observed_time:
                            continue
                        else:
                            # Only access the frame before anticipation time
                            record.set_frame_for_anticipation(self._prediction_timesteps, self._observed_time)
                    #if not self.cfg.MULTI_TASK:
                    # Add action label to the record
                    a = self.actions[(self.actions['verb_class']==record.label['verb'])&(self.actions['noun_class']==record.label['noun'])]
                    record.set_action_label(a['action_class'].values)

                    self._video_records.append(record)
                    self._spatial_temporal_idx.append(idx)
        assert (
                len(self._video_records) > 0
        ), "Failed to load EPIC-KITCHENS{} split {} from {}".format(
            self.ek_version, self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing epic-kitchens{} dataloader (size: {}) from {}".format(
                self.ek_version, len(self._video_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        frames = pack_frames_to_video_clip(self.cfg, self.ek_version, self.mode, self._video_records[index], temporal_sample_index)

        # Perform color normalization.
        frames = frames.float()
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames = self.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )

        #if self.cfg.MULTI_TASK:
        #    label = self._video_records[index].label
        #elif cfg.MODEL.LOSS_FUNC == 'marginal_cross_entropy':
            #label = [(verb, noun, action)]
        #else:
        #    label = self._video_records[index].label['action']
        label = self._video_records[index].label
        frames = utils.pack_pathway_output(self.cfg, frames)
        metadata = self._video_records[index].metadata
        return frames, label, index, metadata


    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
