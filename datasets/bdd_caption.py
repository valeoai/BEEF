import json
import pickle

from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

from tqdm import tqdm
from bootstrap.lib.logger import Logger

try:
    from .bdd import BDDDrive
except:
    from bdd import BDDDrive

def read_text_data(text_path):
    with open(text_path, "r") as f:
        data = f.read().splitlines()
    return data

def pad_video(video_feature, dimension):
    '''
    Fill pad to video to have same length.
    Pad in Left.
    video = [pad,..., pad, frm1, frm2, ..., frmN]
    '''
    padded_feature  = np.zeros(dimension).astype(np.int32)
    max_length      = dimension[0]
    current_length  = video_feature.shape[0]
    num_padding     = max_length - current_length

    if num_padding == 0:
        padded_feature  = video_feature
    elif num_padding < 0:
        steps           = np.linspace(0, current_length, num=max_length, endpoint=False, dtype=np.int32)
        padded_feature  = video_feature[steps]
    else:
        padded_feature[num_padding:] = video_feature

    return padded_feature


def fill_mask(max_length, current_length, zero_location='LEFT'):
    num_padding = max_length - current_length
    if num_padding <= 0:
        mask = np.ones(max_length)
    elif zero_location == 'LEFT':
        mask = np.ones(max_length)
        for i in range(num_padding):
            mask[i] = 0
    elif zero_location == 'RIGHT':
        mask = np.zeros(max_length)
        for i in range(current_length):
            mask[i] = 1

    return mask.astype(np.float32)


class BDDCaption(BDDDrive):
    def __init__(self,
                 dir_data,
                 split,
                 max_length=20,
                 sample_interval=5,
                 n_before=20,
                 batch_size=2,
                 features_dir="",
                 debug=False,
                 shuffle=False,
                 pin_memory=False,
                 nb_threads=0):

        super(BDDCaption, self).__init__(dir_data=dir_data,
                                  split=split,
                                  n_before=n_before,
                                  batch_size=batch_size,
                                  debug=debug,
                                  shuffle=shuffle,
                                  pin_memory=pin_memory,
                                  nb_threads=nb_threads)

        # Load id to word dictionary
        # It is contained in the train directory
        with open(self.dir_processed.joinpath("cap", "train", "idx_to_word.pkl"), "rb") as f:
            self.idx_to_word = pickle.load(f)

        # Hypermarameters
        self.max_length = max_length
        self.sample_interval = sample_interval
        self.features_dir = features_dir # name of the driver model directory under which the features are extracted

    def __len__(self):
        return self.nb_examples

    def build_index(self):
        Logger()('Building index for %s split...' % self.split)

        # Load all raw data
        self.vid_names = read_text_data(self.dir_cap.joinpath("vidName.txt"))
        self.vid_ids = np.load(self.dir_cap.joinpath("video_id.npy"))
        self.actions = np.load(self.dir_cap.joinpath("actions.npy")).astype(np.int64) # Convert to int64 for pytorch
        self.justifications = np.load(self.dir_cap.joinpath("justifications.npy")).astype(np.int64) # Convert to int64 for pytorch
        self.captions = np.load(self.dir_cap.joinpath("captions.npy")).astype(np.int64) # Convert to int64 for pytorch
        self.captions_id = np.load(self.dir_cap.joinpath("caption_id.npy"))
        self.s_times = np.load(self.dir_cap.joinpath("sTime.npy"))
        self.e_times = np.load(self.dir_cap.joinpath("eTime.npy"))
        self.captions_text = read_text_data(self.dir_cap.joinpath("caption.txt"))
        self.justification_text = read_text_data(self.dir_cap.joinpath("justification.txt"))
        self.action_text = read_text_data(self.dir_cap.joinpath("action.txt"))
        self.all_captions_data = [{ "vid_name": "%s_%s" % (vid_id, vid_name),
                                    "action": action,
                                    "justification": justification,
                                    "caption": caption,
                                    "caption_id": caption_id,
                                    "s_time": s_time,
                                    "e_time": e_time,
                                    "caption_text": caption_text,
                                    "justification_text": justification_text,
                                    "action_text": action_text,
                                    } for vid_name, vid_id, action, justification, caption, caption_id, s_time, e_time, caption_text, justification_text, action_text \
                                            in zip(self.vid_names, self.vid_ids, self.actions, self.justifications, self.captions, self.captions_id, self.s_times, self.e_times, self.captions_text, self.justification_text, self.action_text)
                                    ]

        # Load frames of interest
        path_frames_of_interest = self.dir_processed.joinpath("%s_frames_of_interest.json" % self.split)
        with open(path_frames_of_interest, "r") as f:
            vid_name_to_frames_of_interest = json.load(f)
        # Compute correspondance between frame_of_interest and position in the list
        self.vid_name_to_frames_of_interest_position = {vid_name: {frame_interest: i for i, frame_interest in enumerate(list_of_frames)} for vid_name, list_of_frames in vid_name_to_frames_of_interest.items()}

        # Build index
        index = []
        not_in_data = []
        self.nb_examples = 0
        for i, caption in tqdm(list(enumerate(self.all_captions_data))):

            # check if video is ok
            log_path = self.dir_processed_log.joinpath("%s.json" % caption["vid_name"])
            if not log_path.exists():
                not_in_data.append([caption["caption_id"], caption["vid_name"]])
                continue
            with open(log_path, "r") as f:
                log = json.load(f)

            # update index
            index.append(caption)
            self.nb_examples += 1

            if self.debug and i >= 64:
                break

        Logger()('Done')
        return index, not_in_data


    def __getitem__(self, idx):
        item = self.index[idx]

        # Load data for the full video (at 10 Hz)
        json_path = self.dir_processed_log.joinpath(item["vid_name"]+'.json')
        with open(json_path) as f:
            logs = json.load(f)

        # Get the id of the frames thare are needed (at 10Hz)
        frames_id, normal_len, _ = self.get_frames_of_interest(item, logs)

        # Extract segments of interest (becomes 20 frames at 2Hz)
        item["speed"] = np.array(logs["speed_value"]).flatten()[frames_id]
        item["course"] = np.array(logs["course_value"]).flatten()[frames_id]
        item["accelerator"] = np.array(logs["accelerator_value"]).flatten()[frames_id]
        item["curvature"] = np.array(logs["curvature_value"]).flatten()[frames_id] # unused
        item["goaldir"] = np.array(logs["goaldir_value"]).flatten()[frames_id]
        item["timestamp"] = np.array(logs["timestamp_value"]).flatten()[frames_id]
        item["mask"] = fill_mask(self.max_length, normal_len, zero_location='LEFT')

        # Load pred_accel, pred_courses and precomputed features
        precomputed = {}
        for name in ["layer1", "layer2", "layer3", "layer4", "output"]:
            precomputed[name] = np.load(self.dir_processed.joinpath("extracted_driver", self.features_dir, "%s_%s.npy" % (item["vid_name"], name)))
        accels = np.load(self.dir_processed.joinpath("extracted_driver", self.features_dir, "%s_accels.npy" % item["vid_name"]))
        courses = np.load(self.dir_processed.joinpath("extracted_driver", self.features_dir, "%s_courses.npy" % item["vid_name"]))

        # Get row number for the frame_id
        row_numbers = np.array([self.vid_name_to_frames_of_interest_position[item["vid_name"]][frame_id] for frame_id in frames_id])

        # Extract features, a_pred and c_pred for the selected frames
        for name in precomputed.keys():
            item[name] = precomputed[name][row_numbers]

        # Add predicted controls
        a_pred = accels[row_numbers]
        c_pred = courses[row_numbers]
        item["prediction"] = np.concatenate((a_pred[:, None], c_pred[:, None]), axis=1)

        # Convert numpy arrays to torch
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                item[k] = torch.from_numpy(v)
        return item

    def get_frames_of_interest(self, item, logs):

        # Pad +/- 1 second and extract the frame id of interest (at 10 Hz)
        timestamp = np.array(logs["timestamp_value"]).flatten()
        start_stamp = timestamp[0] + float((int(item["s_time"]) - 1)) * 1000
        end_stamp   = timestamp[0] + float((int(item["e_time"]) + 1)) * 1000
        id_interest = np.where(np.logical_and(timestamp >= start_stamp, timestamp <= end_stamp))[0]

        # Reduce FPS from 10 to 2
        id_interest = id_interest[::self.sample_interval]

        # The frames in the original video (at 10 Hz) that are needed
        frames_id = pad_video(id_interest, (self.max_length,))

        return frames_id, len(id_interest), item["vid_name"]

    def save_frames_of_interest(self,):
        frames_to_save_set = defaultdict(set)
        for i in range(self.nb_examples):
            item = self.index[i]
            json_path = self.dir_processed_log.joinpath("%s.json" % item["vid_name"])
            with open(json_path) as f:
                logs = json.load(f)
            frames_id, _, vid_name = self.get_frames_of_interest(item, logs)
            frames_to_save_set[vid_name].update(set(frames_id))

        frames_to_save = {vid_name: sorted(list(map(int, list(values)))) for vid_name, values in frames_to_save_set.items()}
        with open("/datasets_master/BDD-X/processed/%s_frames_of_interest.json" % split, "w") as f:
            f.write(json.dumps(frames_to_save))

    def to_sentence(self, sentence_tokens):
        # decode a sentence
        return [self.idx_to_word[int(token)] for token in sentence_tokens]

