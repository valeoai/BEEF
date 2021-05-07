import json

from pathlib import Path

import h5py
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data

from tqdm import tqdm
from scipy import interpolate

from bootstrap.datasets.dataset import Dataset
from bootstrap.lib.logger import Logger


def preprocess_course(course_value, nImg):
    nRecords = course_value.shape[0]

    for idx in range(1,nRecords):
        if course_value[idx] - course_value[idx-1] > 180:
            course_value[idx:] -= 360
        elif course_value[idx] - course_value[idx-1] < -180:
            course_value[idx:] += 360

    # interpolation
    xaxis         = np.arange(0, nRecords)
    idxs          = np.linspace(0, nRecords-1, nImg).astype("float")  # approximate alignment
    course_interp = interpolate.interp1d(xaxis, course_value)
    course_value  = np.expand_dims(course_interp(idxs),1)

    # exponential smoothing in reverse order
    course_value_smooth = np.flip(exponential_smoothing(np.flip(course_value,0), 0.01),0)
    course_delta        = course_value-course_value_smooth

    return course_delta

def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)


def preprocess_others(series, nImg):
    nRecords = series.shape[0]

    # interpolation
    xaxis         = np.arange(0, nRecords)
    idxs          = np.linspace(0, nRecords-1, nImg).astype("float")  # approximate alignment
    series_interp = interpolate.interp1d(xaxis, series)
    series        = np.expand_dims(series_interp(idxs),1)

    return series

class BDDDrive(Dataset):
    def __init__(self,
                 dir_data,
                 split,
                 n_before=20,
                 batch_size=2,
                 debug=False,
                 shuffle=False,
                 pin_memory=False,
                 nb_threads=0):
        super(BDDDrive, self).__init__(dir_data,
                                  split,
                                  batch_size,
                                  shuffle,
                                  pin_memory,
                                  nb_threads)
        self.debug = debug
        self.dir_data = dir_data
        self.n_before = n_before
        self.dir_processed = self.dir_data.joinpath('processed')
        self.dir_cam = self.dir_processed.joinpath('cam')
        self.dir_log = self.dir_processed.joinpath('log')
        self.dir_cap = self.dir_processed.joinpath("cap", self.split)
        self.dir_processed_log = self.dir_processed.joinpath('processed_log')
        if not self.dir_processed_log.exists():
            self.dir_processed_log.mkdir()

        self.process_all_logs()

        self.index, self._not_in_data = self.build_index()

        self.im_transform = transforms.Compose([transforms.Normalize(mean = [0.43216, 0.394666, 0.37645],
                                                                     std = [0.22803, 0.22145, 0.216989])])

        self.course_stat = {'mean': -0.3337784398579058, 'std': 20.68296022553621} # Both are computed only on train
        self.accel_stat = {'mean': -0.02921719836332804, 'std': 0.8718107758577537} # Both are computed only on train

    def __len__(self):
        return len(self.index)

    def make_batch_loader(self, batch_size=None, shuffle=None, num_samples=200000):
        batch_size = self.batch_size if batch_size is None else batch_size
        shuffle = self.shuffle if shuffle is None else shuffle

        if shuffle:
            sampler = data.RandomSampler(self, replacement=True, num_samples=min(num_samples, len(self)))
            shuffle = None
        else:
            sampler = None

        batch_loader = data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            num_workers=self.nb_threads,
            collate_fn=self.collate_fn,
            sampler=sampler)
        return batch_loader

    def build_index(self):
        Logger()('Building index for %s split...' % self.split)
        split_file = self.dir_processed.joinpath('%s.txt'%self.split)
        index = []
        not_in_data = []
        for idx, vid_id in enumerate(open(split_file)):
            vid_id = vid_id.strip()
            log_path = self.dir_processed_log.joinpath(vid_id+'.json')
            if not log_path.exists():
                not_in_data.append(vid_id)
                continue
            log = json.load(open(log_path))
            n_frames = len(log['course_value'])
            index += [(vid_id, idx) for idx in range(n_frames)]
            if self.debug:
                if idx>=2:
                    break
        Logger()('Done')
        return index, not_in_data


    def process_all_logs(self):
        for log_path in tqdm(list(self.dir_log.iterdir())):
            log_name = log_path.stem
            npy_path = self.dir_processed_log.joinpath(log_name+'.npy')
            json_path = self.dir_processed_log.joinpath(log_name+'.json')
            cam_path = self.dir_cam.joinpath(log_path.name)
            if json_path.exists():
                continue
            try:
                log = h5py.File(log_path,'r')
                cam = h5py.File(cam_path, 'r')
            except:
                Logger()("Could not open %s" % log_name)
                continue
            processed = self.process_log(log, cam)
            X = cam['X'][:]
            np.save(npy_path, X)
            with open(json_path, 'w') as Fjson:
                Fjson.write(json.dumps(processed))
            log.close()
            cam.close()


    def process_log(self, log, cam):
        nImg     = cam['X'].shape[0]
        timestamp_value     = preprocess_others(log["timestamp"][:], nImg)
        curvature_value     = preprocess_others(log["curvature"][:], nImg)
        accelerator_value   = preprocess_others(log["accelerator"][:], nImg)
        speed_value         = preprocess_others(log["speed"][:], nImg)
        course_value        = preprocess_course(log["course"][:], nImg)
        goaldir_value       = preprocess_others(log["goaldir"][:], nImg)
        return {'timestamp_value': timestamp_value.tolist(),
                'curvature_value': curvature_value.tolist(),
                'accelerator_value': accelerator_value.tolist(),
                'speed_value': speed_value.tolist(),
                'course_value': course_value.tolist(),
                'goaldir_value': goaldir_value.tolist()}

    def __getitem__(self, idx):
        vid_name, frame_id = self.index[idx]
        item = {"vid_name": vid_name, "frame_id": frame_id}
        n_before = min(frame_id, self.n_before)

        npy_path = self.dir_processed_log.joinpath(vid_name+'.npy')
        json_path = self.dir_processed_log.joinpath(vid_name+'.json')
        logs = json.load(open(json_path))

        item['course_value'] = torch.Tensor(logs['course_value'][frame_id])
        item['goaldir_value'] = torch.Tensor(logs['goaldir_value'][frame_id])
        item['accelerator_value'] = torch.Tensor(logs['accelerator_value'][frame_id])
        item['speed_value'] = torch.Tensor(logs['speed_value'][frame_id])

        item['course_value_standard'] = (item['course_value'] - self.course_stat['mean']) / self.course_stat['std']
        item['accelerator_value_standard'] = (item['accelerator_value'] - self.accel_stat['mean']) / self.accel_stat['std']

        vid_frames = np.load(npy_path)
        _frames = torch.Tensor(vid_frames[frame_id-n_before:frame_id+1]/255.)
        _frames = _frames.transpose(3,1).transpose(2,3)
        _frames = torch.cat([self.im_transform(f)[None,:] for f in _frames], 0)
        frames = torch.Tensor(self.n_before+1, _frames.size(1), _frames.size(2), _frames.size(3)).zero_()
        frames[-(frame_id+1):] = _frames

        item['frames'] = frames
        return item

