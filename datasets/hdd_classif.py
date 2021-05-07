from collections import Counter
import json
from pathlib import Path
from PIL import Image 
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


from bootstrap.lib.logger import Logger
from bootstrap.datasets import transforms as bootstrap_tf

try:
    from .hdd import HDD 
except:
    from hdd import HDD

class HDDClassif(HDD):
    def __init__(self,
                 dir_data, 
                 split,
                 win_size,
                 im_size,
                 layer, # "goal" or "cause"
                 frame_position,
                 traintest_mode,
                 fps=10,
                 horizon=2, # in seconds
                 extract_mode=False,
                 batch_size=2,
                 debug=False,
                 shuffle=False,
                 pin_memory=False, 
                 nb_threads=0):

        self.win_size = win_size
        self.frame_position = frame_position

        super(HDDClassif, self).__init__(dir_data, 
                                         split,
                                         im_size,
                                         fps,
                                         horizon, # in seconds
                                         batch_size,
                                         debug,
                                         shuffle,
                                         pin_memory, 
                                         nb_threads)

        self.layer = layer

        if self.layer == "cause":
            self.layer_id = '1'
            self.classid_to_ix = [-1, 16, 17, 18, 19, 20, 22]
        elif self.layer == "goal":
            self.layer_id = '0'
            self.classid_to_ix = [-1, 0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12]
        else:
            raise ValueError(self.layer)
        # The classid 0 is the background class

        self.ix_to_classid = dict((ix, classid) for classid, ix in enumerate(self.classid_to_ix))

        self.class_freq = self.get_class_freq()

        self.collate_fn = bootstrap_tf.Compose([
            bootstrap_tf.ListDictsToDictLists(),
            bootstrap_tf.StackTensors()
        ])

        self.dir_navig_features = self.dir_processed_annot


        self.im_transform = transforms.Compose([transforms.Resize((self.im_h, self.im_w)),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean = [0.43216, 0.394666, 0.37645], 
                                                                     std = [0.22803, 0.22145, 0.216989])])

        self.traintest_mode = traintest_mode
        if self.traintest_mode:
            self.make_batch_loader = self._make_batch_loader_traintest
        else:
            self.make_batch_loader = self._make_batch_loader


    def classid_to_classname(self, classid):
        ix = self.classid_to_ix[classid]
        if ix == -1:
            return '__background__'
        else:
            return self.ix_to_event[ix]


    def _make_batch_loader(self, batch_size=None, shuffle=None, num_samples=200000):
        nb_threads = self.nb_threads
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
            num_workers=nb_threads,
            collate_fn=self.collate_fn,
            sampler=sampler)
        return batch_loader

    def _make_batch_loader_traintest(self, batch_size=None, shuffle=None):
        nb_threads = self.nb_threads
        batch_size = self.batch_size if batch_size is None else batch_size
        num_samples = batch_size*70000
        shuffle = self.shuffle if shuffle is None else shuffle

        if shuffle:
            sampler = data.RandomSampler(self, replacement=True, num_samples=num_samples)
            shuffle = None
        else:
            sampler = None

        batch_loader = data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            num_workers=nb_threads,
            collate_fn=self.collate_fn,
            sampler=sampler)
        return batch_loader


    def build_index(self):
        Logger()('Building index for %s split...' % self.split)
        split_file = self.dir_data.joinpath(self.split+'.txt')
        index = []
        session_template = "{0}-{1}-{2}-{3}-{4}" 
        self.vid_to_index = []
        self.vidname_to_vidid = {}
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
            if annot_path.exists():
                frame_annots = sorted(annot_path.iterdir())
                frame_annots = [None]*self.frame_position + frame_annots + [None]*(self.win_size-self.frame_position-1) # Zero-padding of the full video, such that each frame can get a context
                L = [frame_annots[i:i+self.win_size] for i in range(0, len(frame_annots)-self.win_size+1)]
                self.vid_to_index.append((len(index), len(index)+len(L)))
                self.vidname_to_vidid[annot_path.name] = len(index)
                index += L
                # if self.debug:
                #     index += frame_annots[5000:7000]
                #     break
                # else:
                #     index += frame_annots
            if self.debug and idx==1:
                break
        Logger()('Done')
        return index

    def get_class_freq(self):
        class_freq_path = self.dir_processed_annot.joinpath('%s_class_freq.json' % self.layer)
        if class_freq_path.exists():
            Logger()('Loading class frequency')
            class_freq = json.load(open(class_freq_path))
            Logger()('Loaded class frequency')
        else:
            Logger()('Computing class frequency')
            if self.split != "train":
                raise NotImplementedError('Extract class weigths on train set first')
            class_freq = self.compute_class_freq()
            with open(class_freq_path, 'w') as F:
                F.write(json.dumps(class_freq))
        return class_freq


    def compute_class_freq(self):
        class_freq = Counter()
        S = 0
        for paths in self.index:
            annot_path = paths[-1]
            if annot_path is None:
                continue
            annot = json.load(open(annot_path))
            event = annot['labels'][self.layer_id]
            classid = self.ix_to_classid.get(event, 0)
            class_freq[classid] += 1
            S += 1
        
        for classid in class_freq:
            class_freq[classid] = class_freq[classid] / S

        return class_freq

    def get_navig(self, annot):
        item = {}
        if len(annot['prev_xy']) == self.length:
            prev_xy = torch.Tensor(annot['prev_xy'])
            r_prev_xy = torch.Tensor(annot['r_prev_xy'])
        else:
            # should be padded before
            n = len(annot['prev_xy'])
            prev_xy = torch.Tensor(self.length,2).zero_()
            r_prev_xy = torch.Tensor(self.length,2).zero_()
            if n>0:
                prev_xy[self.length - n:] = torch.Tensor(annot['prev_xy'])
                r_prev_xy[self.length - n:] = torch.Tensor(annot['r_prev_xy'])
        item['prev_xy'] = prev_xy
        item['r_prev_xy'] = r_prev_xy

        if len(annot['next_xy']) == self.length:
            next_xy = torch.Tensor(annot['next_xy'])
            r_next_xy = torch.Tensor(annot['r_next_xy'])
        else:
            # should be padded after
            n = len(annot['next_xy'])
            next_xy = torch.Tensor(self.length,2).zero_()
            r_next_xy = torch.Tensor(self.length,2).zero_()
            if n>0:
                next_xy[:n] = torch.Tensor(annot['next_xy'])
                r_next_xy[:n] = torch.Tensor(annot['r_next_xy'])
        item['next_xy'] = next_xy
        item['r_next_xy'] = r_next_xy
        item['blinkers'] = torch.LongTensor([self.blinkers_to_ix[annot['blinkers']]])
        return item 

    def get_navig_path(self, annot_path):
        # Sometimes, due to sampling considerations, the navig annotation doesn't exist.
        # We simply take the navig annotation for the closest existing sample
        annot_navig_path = self.dir_navig_features.joinpath(annot_path.parent.name, 
                                                            annot_path.name)
        if not annot_navig_path.exists():
            annot_num = int(annot_path.stem)
            annot_navig_path = self.dir_navig_features.joinpath(annot_path.parent.name,
                                                                f"{annot_num-1:06d}.json")
        if not annot_navig_path.exists():
            annot_navig_path = self.dir_navig_features.joinpath(annot_path.parent.name,
                                                                f"{annot_num+1:06d}.json")
        if not annot_navig_path.exists():
            annot_navig_path = self.dir_navig_features.joinpath(annot_path.parent.name,
                                                                f"{annot_num-2:06d}.json")
        return annot_navig_path


    def __getitem__(self, idx):
        paths = self.index[idx]

        y_true = torch.LongTensor(self.win_size).zero_() -1 

        frames = None
        navig = None
        item = {}

        for frame_id, annot_path in enumerate(paths):
            if annot_path is None:
                continue
            frame_number = int(annot_path.stem) + 1
            frames_folder = self.dir_processed_img.joinpath(annot_path.parent.name)
            frame_path = frames_folder.joinpath(f"{frame_number:06d}.jpg")
            im = Image.open(frame_path)
            im = self.im_transform(im)
            if frames is None:
                frames = torch.Tensor(self.win_size, 3, self.im_h, self.im_w).zero_()
            frames[frame_id] = im
            annot = json.load(open(annot_path))
            event = annot['labels'][self.layer_id]
            y_true[frame_id] = self.ix_to_classid.get(event, 0)
            if navig is None:
                navig = {'prev_xy':torch.Tensor(self.win_size, self.length, 2).zero_() - 1,
                        'next_xy':torch.Tensor(self.win_size, self.length, 2).zero_() - 1,
                        'r_prev_xy':torch.Tensor(self.win_size, self.length, 2).zero_() - 1,
                        'r_next_xy':torch.Tensor(self.win_size, self.length, 2).zero_() - 1,
                        'xy_polynom':torch.Tensor(self.win_size, 5, 2).zero_() - 1,
                        'blinkers':torch.LongTensor(self.win_size).zero_() - 1}

            annot_navig_path = self.get_navig_path(annot_path)
            annot_navig = json.load(open(annot_navig_path))
            _navig = self.get_navig(annot_navig)
            for k in _navig:
                navig[k][frame_id] = _navig[k]
        item.update(navig)
        item['frames'] = frames
        item['idx'] = idx
        item['paths'] = paths
        item['frame_path'] = paths[self.frame_position]
        item['y_true_all'] = y_true
        item['y_true'] = y_true[self.frame_position]
        for k in navig:
            item[k+'_all'] = item[k]
            item[k] = item[k+'_all'][self.frame_position]
        item['frame_position'] = torch.LongTensor([self.frame_position])
        return item


if __name__ == "__main__":
    split = "val"
    fps = 3


    dir_data = Path("/datasets_local/HDD")
    nb_threads = 0
    horizon = 2

    win_size = 21
    layer = "goal"
    batch_size = 12
    use_navig = False

    im_size = "small"

    dataset = HDDClassif(dir_data, 
                 split,
                 win_size,
                 im_size,
                 layer, # "goal" or "cause"
                 use_navig=use_navig,
                 fps=fps,
                 horizon=horizon, # in seconds
                 batch_size=batch_size,
                 debug=False,
                 shuffle=False,
                 pin_memory=False, 
                 nb_threads=0)


    vidname_to_index = {}
    for idx, sequence in enumerate(dataset.index):
        vid_name = sequence[0].parent.name
        if vid_name not in vidname_to_index:
            vidname_to_index[vid_name] = []
        vidname_to_index[vid_name].append(idx)

    batch_sampler = SequentialBatchSampler(vidname_to_index, batch_size)

    N = 0
    for batch in batch_sampler:
        print(batch)
        N += 1




    # item = dataset[5]

    # loader = dataset.make_batch_loader(batch_size,
    #                                    shuffle=False)

    # for idx, batch in enumerate(loader):
    #     break