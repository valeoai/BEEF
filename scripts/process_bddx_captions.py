import os
import json
import pickle

from collections import defaultdict

import h5py
import numpy as np

import explainable_deep_driving.src.config

from explainable_deep_driving.src.utils import (bcolors, check_and_make_folder,
                                                dict2)
from explainable_deep_driving.src.utils_nlp import (build_caption_vector,
                                                    build_feat_matrix,
                                                    build_vocab,
                                                    cluster_annotations,
                                                    process_caption_data,
                                                    save_pickle)

data_path = "/datasets_master/BDD-X/"
processed_path = os.path.join(data_path, "processed")
cap_path = os.path.join(processed_path, "cap")
full_caption_path = os.path.join(processed_path, "captions_BDDX.json")

# Split captions into train, test and val
# this creates "captions_BDDX_[train,test,val].json"
# Need to be run only onced

# Load train, test and val splits
video_split = {}
for split in ["train", "test", "val"]:
    with open(os.path.join(processed_path, "%s.txt" % split), "r") as f:
        video_split[split] = set(f.read().splitlines())

separate_annotation = defaultdict(list)
separate_video = defaultdict(set)
data = {split: {"annotations": [], "videos": [], "info": []} for split in ["train", "test", "val"]}

# separate annotations in the different splits
captions = json.load(open(full_caption_path))
counter_of_video = defaultdict(set)
for i, annotation in enumerate(captions["annotations"]):
    vid_identifier = str(annotation["video_id"]) + "_" + annotation["vidName"]
    for split in ["train", "test", "val"]:
        if vid_identifier in video_split[split]:
            counter_of_video[split].add(vid_identifier)
            separate_annotation[split].append(annotation)
            separate_video[split].add(annotation["video_id"])

            for key in ["annotations", "info", "videos"]:
                data[split][key].append(captions[key][i])

# create the json
for split in ["train", "test", "val"]:
    print("%s contains %d captions" % (split, len(data[split]["annotations"])))
    with open(os.path.join(processed_path, "captions_BDDX_%s.json" % split), "w") as f:
        json.dump(data[split], f)

#-----------------------
# Parameters
#-----------------------
param = dict2(**{
    "max_length":       20,    # the maximum length of sentences
    "vid_max_length":   10,    # the maximum length of video sequences (in seconds ?)
    "size_of_dict":     10000, # the size of dictionary
    "chunksize":        10,    # for h5 format file writing
    "savepath":         cap_path,
    "FINETUNING":       True,
#     "FINETUNING":       False,
#     "SAVEPICKLE":       True })
     "SAVEPICKLE":       False })


caption_length = {}
for split in ['train', 'test', 'val']:
    caption_file = os.path.join(processed_path, "captions_BDDX_%s.json" % split)

    # Step1: Preprocess caption data + refine captions
    # load caption, concat action and justification, regex cleaning, delete captions longer than max_length=20
    annotations, caption_length[split] = process_caption_data(caption_file=caption_file, max_length=param.max_length)

    unique_id = set()
    for vid_id, vid_name in zip(annotations["video_id"], annotations["vidName"]):
        unique_id.add("%s_%s" % (vid_id, vid_name))
    print(len(unique_id))

    if param.SAVEPICKLE:
        save_pickle(annotations, os.path.join(cap_path, split, "%s.annotations.pkl" % split))
    print(bcolors.BLUE   + '[main] Length of {} : {}'.format(split, len(annotations)) + bcolors.ENDC)

    # Step2: Build dictionary
    if param.FINETUNING:
        with open(os.path.join(cap_path, "train", "word_to_idx.pkl"), 'rb') as f:
            word_to_idx = pickle.load(f)
    else:
        if split == 'train':
            word_to_idx, idx_to_word = build_vocab(annotations=annotations, size_of_dict=param.size_of_dict)
            if param.SAVEPICKLE:
                save_pickle(word_to_idx, os.path.join(cap_path, split, "word_to_idx.pkl"))
                save_pickle(idx_to_word, os.path.join(cap_path, split, "idx_to_word.pkl"))
        else:
            with open(os.path.join(cap_path, "train", "word_to_idx.pkl"), 'rb') as f:
                word_to_idx = pickle.load(f)

#     # Step3: word to index
    captions = {}
    for sentence in ["caption", "action", "justification"]:
        captions[sentence] = build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=param.max_length, sentence=sentence)
        if param.SAVEPICKLE:
            save_pickle(captions, os.path.join(cap_path, split, "%s.pkl" % sentence))


    ####
    # Step4: save captions and save metadata
    ####

#     # save padded captions
#     for sentence in ["caption", "action", "justification"]:
    for sentence in ["action", "justification"]:
        np.save(os.path.join(cap_path, split, "%ss.npy" % sentence), captions[sentence])

#     # save action, justification and caption textual data
#     for t in ["action", "justification", "caption", "vidName"]:
    for t in ["action", "justification"]:
        with open(os.path.join(cap_path, split, "%s.txt" % t), "w") as f:
            f.write("\n".join(annotations[t].to_list()))

    # save metadata
    np.save(os.path.join(cap_path, split, "sTime.npy"), annotations["sTime"].to_numpy(dtype=np.int32))
    np.save(os.path.join(cap_path, split, "eTime.npy"), annotations["eTime"].to_numpy(dtype=np.int32))
    np.save(os.path.join(cap_path, split, "video_id.npy"), annotations["video_id"].to_numpy(dtype=np.int32))
    np.save(os.path.join(cap_path, split, "caption_id.npy"), annotations["id"].to_numpy(dtype=np.int32))

