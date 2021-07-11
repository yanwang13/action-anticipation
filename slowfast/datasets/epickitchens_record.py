"""
This is implementation is referenced from
https://github.com/epic-kitchens/epic-kitchens-slowfast/blob/master/slowfast/datasets/epickitchens_record.py
"""
from .video_record import VideoRecord
from datetime import timedelta
import time


def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(hours=x.tm_hour,
                          minutes=x.tm_min,
                          seconds=x.tm_sec).total_seconds()) + float(
        timestamp.split('.')[-1]) / 100
    return sec

def get_frame_index(timestamp, fps):
    """
    Turn timestamps to correspond frame index.
    """
    if type(timestamp) == float:
        return int(round(timestamp) * fps)
    else:
        return int(round(timestamp_to_sec(timestamp) * fps))


class EpicKitchensVideoRecord(VideoRecord):
    def __init__(self, tup, video_fps=None):
        self._index = str(tup[0])
        self._series = tup[1]

        self._label = {'verb': self._series['verb_class'] if 'verb_class' in self._series else -1,
                'noun': self._series['noun_class'] if 'noun_class' in self._series else -1}
        if self._label['verb']==-1 or self._label['noun'] == -1:
            print(self._series)

        self._start_time = timestamp_to_sec(self._series['start_timestamp'])

        if video_fps is None:
            self._fps = 50 if len(self._series['video_id'].split('_')[1]) == 3 else 60
        else:
            self._fps = video_fps
        self._start_frame =  get_frame_index(self._series['start_timestamp'],  self._fps)
        # debug check get_frame_index works
        #assert self._start_frame == int(round(timestamp_to_sec(self._series['start_timestamp']) * self._fps))

        self._end_frame =  get_frame_index(self._series['stop_timestamp'],  self._fps)

    @property
    def participant(self):
        return self._series['participant_id']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def start_frame(self):
        return self._start_frame
        #return int(round(timestamp_to_sec(self._series['start_timestamp']) * self.fps))

    @property
    def end_frame(self):
        return self._end_frame
        #return int(round(timestamp_to_sec(self._series['stop_timestamp']) * self.fps))

    @property
    def start_time(self): # sec
        return self._start_time

    def set_frame_for_anticipation(self, prediction_timesteps, observed_time):
        end_time = self._start_time - prediction_timesteps
        self._end_frame = get_frame_index(end_time, self._fps)
        self._start_time = end_time - observed_time
        self._start_frame = get_frame_index(self._start_time, self._fps)
        return

    @property
    def fps(self):
        return self._fps
        #is_100 = len(self.untrimmed_video_name.split('_')[1]) == 3
        #return 50 if is_100 else 60

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame

    @property
    def label(self):
        return self._label
        #return {'verb': self._series['verb_class'] if 'verb_class' in self._series else -1,
        #        'noun': self._series['noun_class'] if 'noun_class' in self._series else -1}

    def set_action_label(self, action):
        if len(action) > 1:
            print(f'action label more than one: {action}')
        self._label['action'] = action[0]
        return

    @property
    def metadata(self):
        return {'narration_id': self._index}
