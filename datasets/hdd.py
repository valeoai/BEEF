import os
import re
import sys
import csv
import json
from pathlib import Path
from copy import copy
from collections import Counter
from tqdm import tqdm
import pickle as pkl
import csv
import utm
import subprocess
from subprocess import call
import math

import numpy as np
import pandas as pd
from scipy import interpolate
import skvideo.io
import cv2
from PIL import Image 

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.datasets.dataset import Dataset
from bootstrap.datasets import transforms as bootstrap_tf
from bootstrap.datasets.dataset import ListDatasets

class HDD(Dataset):
    def __init__(self,
                 dir_data, 
                 split,
                 im_size="small", # should be in ["small", "large"]
                 fps=10,
                 horizon=2, # in seconds
                 batch_size=2,
                 debug=False,
                 shuffle=False,
                 pin_memory=False, 
                 nb_threads=0):
        super(HDD, self).__init__(dir_data, 
                                  split,
                                  batch_size,
                                  shuffle,
                                  pin_memory, 
                                  nb_threads)
        self.debug = debug
        self.fps = fps
        self.horizon = horizon
        self.length = self.fps*self.horizon
        self.im_size = im_size


        self.dir_processed = self.dir_data.joinpath('processed')
        name_data = self.dir_data.name
        # name_processed = name_data + "/processed"

        # import ipdb; ipdb.set_trace()
        self.dir_processed_img = self.dir_processed.joinpath('img')

        self.dir_processed_annot = self.dir_processed.joinpath(f'fps,{self.fps}_horizon,{self.horizon}sec')
        cache_path = self.dir_data.joinpath("EAF_parsing/saved_index.pkl")
        self.cache_data = pkl.load(open(cache_path, 'rb'))
        self.events_pd = self.cache_data['events_pd']
        self.ix_to_event = dict((k,re.sub("([^\x00-\x7F])+"," ",v).strip()) \
                              for k,v in self.cache_data['event_type_ix'].items()) # remove chinese caracters
        self.layer_to_ix = dict((re.sub("([^\x00-\x7F])+"," ",k).strip(),v) for k,v in self.cache_data['inv_layer_ix'].items())
        self.ix_to_layer = dict((v,k) for k,v in self.layer_to_ix.items())

        self.ix_to_blinkers = {0:'', 1:'lturn', 2:'rturn'}
        self.blinkers_to_ix = dict((v,k) for k,v in self.ix_to_blinkers.items())


        self.preprocess()
        self.index = self.build_index()


        if self.im_size == "small":
            self.im_h, self.im_w = (90, 160)
        elif self.im_size == "large":
            self.im_h, self.im_w = (224, 224)
        else:
            raise ValueError(self.im_size)

        self.im_transform = transforms.Compose([transforms.Resize((self.im_h, self.im_w)),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
        

        self.collate_fn = bootstrap_tf.Compose([
            bootstrap_tf.ListDictsToDictLists(),
            bootstrap_tf.PadTensors(use_keys=['prev_xy',
                                              'r_prev_xy',
                                              'next_xy',     # !! To be rigorous, we should zero-pad next_xy and r_next_xy at the end and
                                              'r_next_xy']), # not at the begining, but its very rare so we don't care...
            bootstrap_tf.StackTensors()
        ])



    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        annot_path = self.index[idx]
        frame_number, vid_name, im = self.get_im(annot_path)
        annot = json.load(open(annot_path))
        if annot['prev_xy'] == []:
            annot['r_prev_xy'] = [[]]
            annot['prev_xy'] = [[]]
        if annot['next_xy'] == []:
            annot['next_xy'] = [[]]
            annot['r_next_xy'] = [[]]
        item = {'frame':im,
                'xy':torch.Tensor(annot['xy']),
                'prev_xy':torch.Tensor(annot['prev_xy']),
                'next_xy':torch.Tensor(annot['next_xy']),
                'r_prev_xy':torch.Tensor(annot['r_prev_xy']),
                'r_next_xy':torch.Tensor(annot['r_next_xy']),
                'timestamp':torch.Tensor([annot['timestamp']]),
                'frame_number':frame_number,
                'vid_name':vid_name,
                'idx':idx,
                'blinkers':torch.LongTensor([self.blinkers_to_ix[annot['blinkers']]])
                }
        for layer,event in annot['labels'].items():
            item['label_'+layer] = torch.LongTensor([event])
        return item


    def get_im(self, annot_path):
        frame_number = int(annot_path.stem) + 1 # Videos are indexed starting by 1
        vid_name = annot_path.parent.name
        frames_folder = self.dir_processed_img.joinpath(vid_name)
        frame_path = frames_folder.joinpath(f"{frame_number:06d}.jpg")
        im = Image.open(frame_path)
        im = self.im_transform(im)
        return frame_number, vid_name, im


    def build_index(self):
        Logger()('Building index for %s split...' % self.split)
        split_file = self.dir_data.joinpath('%s.txt'%self.split)
        index = []
        session_template = "{0}-{1}-{2}-{3}-{4}" 
        for idx, session_id in enumerate(open(split_file)):
            name = session_template.format(session_id[:4],
                                           session_id[4:6],
                                           session_id[6:8],
                                           session_id[8:10],
                                           session_id[10:12])
            annot_paths = list(filter(lambda x:name in x.as_posix(), 
                                      self.dir_processed_annot.iterdir()))

            if len(annot_paths) == 0:
                continue
                
            assert len(annot_paths) == 1
            annot_path = annot_paths[0]

            if annot_path.exists:
                frame_annots = sorted(annot_path.iterdir())
                index += frame_annots
            if self.debug and idx>1:
                break
        Logger()('Done')
        return index

    def subsample(self, value, times):
        current_fps = len(times) / (times[-1] - times[0])
        sample_every = int(round(current_fps / self.fps))
        return value[::sample_every], times[::sample_every]

    def preprocess(self):
        dir_folders = self.dir_data.joinpath('release_2019_07_08')
        vid_folders = []
        for folder in dir_folders.iterdir():
            if folder.stem == "EAF":
                continue
            vid_folders += list(folder.iterdir())

        vid_folders = sorted(vid_folders)
        Logger()('Preprocessing...')
        for vid_folder in tqdm(vid_folders):
            self.extract_video(vid_folder)
            self.process_annotations(vid_folder)
        Logger()('Preprocessing done')

    def extract_video(self, vid_folder):
        vidpath = sorted(list(vid_folder.joinpath('camera','center').iterdir()))[0]
        
        process_vid_folder = self.dir_processed_img.joinpath(vidpath.stem)
        if not process_vid_folder.exists():
            process_vid_folder.mkdir(parents=True, exist_ok=True)
            call(['ffmpeg',
                  '-i', vidpath.as_posix(),
                  '-threads', '4',
                  '-s','640*360',
                  '-qscale:v', '10',
                  process_vid_folder.as_posix()+'/%06d.jpg'
                 ])
        return None


    def process_annotations(self, vid_folder):

        vidpath = sorted(list(vid_folder.joinpath('camera','center').iterdir()))[0]
        annot_folder = self.dir_processed_annot.joinpath(vidpath.stem)
        # They are all in the training set.
        # for these videos, we don't have gps signals, so we don't use them.
        if vidpath.stem in ['2017-04-11-09-43-46_new_0.75', 
                            '2017-06-08-14-45-09_new_0.75', 
                            '2017-06-08-16-26-31_new_0.75', 
                            '2017-06-08-17-07-21_new_0.75',
                            '2017-06-13-09-52-51_new_0.75',
                            '2017-06-13-11-27-56_new_0.75',
                            '2017-06-13-13-18-53_new_0.75',
                            '2017-06-14-11-47-21_new_0.75',
                            '2017-10-04-09-38-27_new_0.75']:
            return

        if annot_folder.exists():
            return

        try: 
            frame_number, timestamps, X, Y, pos_ts, lturns, rturns, turn_ts = self.extract_signals_from_csvs(vid_folder)
        except FileNotFoundError:
            Logger()(f"CSV not found for {vidpath.stem}")
            return
        except utm.error.OutOfRangeError:
            Logger()(f"Lat-lng range error for {vidpath.stem}")
            return
        
        video_init_timestamp = timestamps[0]
        video_duration = timestamps[-1] - video_init_timestamp

        frame_number, timestamps, X, Y, pos_ts, lturns, rturns, turn_ts = self.process_signals(frame_number, timestamps, 
                                                                                               X, Y, pos_ts, 
                                                                                               lturns, rturns, turn_ts)

        labels_dict = self.process_labels(vid_folder, video_init_timestamp, video_duration, timestamps)
        
        N = len(timestamps)

        annot_folder.mkdir(parents=True, exist_ok=True)

        for idx, ts in enumerate(timestamps):
            item = {}
            xy = np.array((X[idx], Y[idx])).T
            begin = max(0, idx-self.length)
            end = min(idx + self.length + 1, N)
            prev_xy = np.array((X[begin:idx] - xy[0], Y[begin:idx] - xy[1])).T
            next_xy = np.array((X[idx+1: end] - xy[0], Y[idx+1: end] - xy[1])).T
            if len(prev_xy) > 0:
                direction = prev_xy[-1]

                if direction[0] == 0:
                    angle = 0.
                else:
                    angle = np.arctan(direction[1] / direction[0])
                    
                angle += np.pi/2
                if direction[0] < 0:
                    angle += np.pi

                c, s = np.cos(angle), np.sin(angle)
                rotation = np.array([
                    [c, -s],
                    [s, c]
                ])

                r_all_xy = np.matmul(np.concatenate([prev_xy, [[0,0]], next_xy], 0), rotation)
                
                r_prev_xy = r_all_xy[:len(prev_xy)]
                r_next_xy = r_all_xy[len(prev_xy)+1:]
            else:
                r_prev_xy = np.array([[]])
                r_next_xy = next_xy
                r_all_xy = np.concatenate([[[0,0]], next_xy], 0)

            labels = {k: v[idx] for k,v in labels_dict.items()}
            blinkers = '' + (lturns[idx] == 1.)*'lturn' + (rturns[idx] == 1.)*'rturn'

            item = {
                'timestamp': ts,
                'xy':xy.tolist(),
                'prev_xy':prev_xy.tolist(),
                'next_xy':next_xy.tolist(),
                'r_prev_xy':r_prev_xy.tolist(),
                'r_next_xy':r_next_xy.tolist(),
                'labels':labels,
                'blinkers':blinkers
            }
            annot_path = annot_folder.joinpath(f'{frame_number[idx]:06d}.json')
            with open(annot_path, 'w') as F:
                F.write(json.dumps(item))
        return  

    def extract_signals_from_csvs(self, vid_folder):
        signals_folder = vid_folder.joinpath('general', 'csv')
        png_ts_folder = Path(vid_folder.as_posix().replace('07_08', '07_25'))
        png_ts_folder = png_ts_folder.joinpath('camera', 'center')
        png_reader = csv.DictReader(open(png_ts_folder.joinpath('png_scu_merged.csv')))
        timestamps, frame_number = [], []
        for l in png_reader:
            timestamps.append(float(l['#unix_timestamp']))
            frame_number.append(int(l['frame_number']))

        timestamps = np.array(timestamps)
        frame_number = np.array(frame_number)

        pos_reader = csv.DictReader(open(signals_folder.joinpath('rtk_pos.csv')))
        coords, pos_ts = [], []
        for l in pos_reader:
            lat, lng = float(l['lat']), float(l['lng'])
            if lat < -90:
                lat = lat % (-90)
            coords.append((lat, lng))
            pos_ts.append(float(l['# unix_timestamp']))

        coords = np.array(coords)
        X, Y, _, _ = utm.from_latlon(coords[:,0], coords[:,1])

        
        X = -np.array(X) #Minus is required, don't know why
        Y = np.array(Y)
        pos_ts = np.array(pos_ts)

        lturns, rturns, turn_ts = [], [], []
        turn_reader = csv.DictReader(open(signals_folder.joinpath('turn_signal.csv')))
        for l in turn_reader:
            turn_ts.append(float(l['# unix_timestamp']))
            lturns.append(float(l['lturn']))
            rturns.append(float(l['rturn']))

        lturns = np.array(lturns)
        rturns = np.array(rturns)
        turn_ts = np.array(turn_ts)
        return frame_number, timestamps, X, Y, pos_ts, lturns, rturns, turn_ts

    def crop_signal(self, signal, times, start, end):
        return signal[(times>=start) & (times<=end)], times[(times>=start) & (times<=end)]

    def process_signals(self, frame_number, timestamps, X, Y, pos_ts, lturns, rturns, turn_ts):
        start = max(timestamps[0], pos_ts[0], turn_ts[0])
        end = min(timestamps[-1], pos_ts[-1], turn_ts[-1])
        X, _ = self.crop_signal(X, pos_ts, start, end) # We need the original timestamps for X and Y, this is why we don't modify it in this line
        Y, pos_ts = self.crop_signal(Y, pos_ts, start, end)
        lturns, _ = self.crop_signal(lturns, turn_ts, start, end)
        rturns, turn_ts = self.crop_signal(rturns, turn_ts, start, end)
        frame_number, timestamps = self.crop_signal(frame_number, timestamps, start, end)

        frame_number, timestamps = self.subsample(frame_number, timestamps)
        X, _ = self.subsample(X, pos_ts)
        Y, pos_ts  = self.subsample(Y, pos_ts)
        lturns, _ = self.subsample(lturns, turn_ts)
        rturns, turn_ts = self.subsample(rturns, turn_ts)

        # Signals have been cropped and subsampled at self.fps, but it still difficult to get signal values for times in `timestamps`,
        # as `pos_ts` and `turn_ts` have different values. In this step, for each time in `timestamps`, we get the signal value of the 
        # closest time

        ix_turn_ts = np.zeros(len(timestamps), 
                              dtype=int)
        ix_pos_ts = np.zeros(len(timestamps), 
                             dtype=int)
        for i, t in enumerate(timestamps):
            ix_turn_ts[i] = np.abs(t-turn_ts).argmin()
            ix_pos_ts[i] = np.abs(t-pos_ts).argmin()
            
        X = X[ix_pos_ts]
        Y = Y[ix_pos_ts]
        lturns = lturns[ix_turn_ts]
        rturns = rturns[ix_turn_ts]
        return frame_number, timestamps, X, Y, pos_ts, lturns, rturns, turn_ts

    def process_labels(self, vid_folder, video_init_timestamp, video_duration, timestamps):
        def convert_time(event, reference_frame_index):
            '''Pandas(Index=3583, layer=5, event_type=68,
                      session_id='201704111540', start=6537340, end=6541750)
            event[3] - session_id
            event[4] - start
            event[5] - end
            '''
            # 30 frames per second
            start_frame = int(event["start"] * .03)
            end_frame = min(int(math.ceil(event["end"] * .03)), reference_frame_index.shape[0] - 1)
            assert start_frame < end_frame
            # print start_frame, end_frame, reference_frame_index.shape
            return reference_frame_index.iloc[start_frame][0], reference_frame_index.iloc[end_frame][0]

        # The usage of video_init_timestamp and video_duration is DEPRECATED 
        session_id = vid_folder.stem.replace('-','').split('_')[0][:12]
        session_events = self.events_pd[self.events_pd['session_id'] == session_id]

        png_ts_folder = Path(vid_folder.as_posix().replace('07_08', '07_25'))
        png_ts_folder = png_ts_folder.joinpath('camera', 'center')

        reference_frame_index = pd.read_csv(png_ts_folder.joinpath('png_timestamp.csv'), usecols=[0, 1, 2])
        starts, ends = [], []
        for event_id in range(len(session_events)):
            start, end = convert_time(session_events.iloc[event_id], reference_frame_index)
            starts.append(start)
            ends.append(end)
        session_events.loc[session_events.index, 'start'] = starts
        session_events.loc[session_events.index, 'end'] = ends

        # label_offset = video_init_timestamp + video_duration - session_events['end'].max() / 1000
        # session_events.loc[session_events.index, 'start'] = session_events.loc[session_events.index, 'start'] / 1000 + label_offset
        # session_events.loc[session_events.index, 'end'] = session_events.loc[session_events.index, 'end'] / 1000 + label_offset

        labels_dict = {}
        for layer in self.ix_to_layer:
            labels_dict[layer] = np.zeros(len(timestamps), 'int') - 1 
            
        for k, event in session_events.transpose().to_dict().items():
            layer_id = event['layer']
            event_id = event['event_type']
            idx_time_event = (event['start'] < timestamps ) * (timestamps < event['end'])
            labels_dict[layer_id][idx_time_event] =  event_id

        for layer in self.ix_to_layer:
            labels_dict[layer] = labels_dict[layer].tolist()

        return labels_dict


if __name__ == "__main__":
    dir_data = Path("/datasets_local/HDD")
    split = "train"
    im_size = "small"
    fps = 3
    horizon = 2
    batch_size = 13
    debug = True
    dataset = HDD(
            dir_data=dir_data,
            split=split,
            im_size=im_size,
            fps=fps,
            horizon=horizon,
            batch_size=batch_size,
            debug=debug,
            shuffle=False,
            pin_memory=False,
            nb_threads=0
        )

