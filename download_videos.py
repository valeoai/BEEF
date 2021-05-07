#!/usr/bin/env python
from sys import platform
from tqdm import tqdm
import os
import wget

from explainable_deep_driving.src.utils import csv_dict_reader, bcolors


if __name__ == "__main__":
    annotation_path = "/datasets_master/BDD-X/BDD-X-Annotations_v1.csv"
    vid_path = "/datasets_master/BDD-X/videos/"

    with open(annotation_path) as f_obj:
        examples = csv_dict_reader(f_obj)

        '''
        Keys:
            1. Input.Video
            2. Answer.1start
            3. Answer.1end
            4. Answer.1action
            5. Answer.1justification
        '''
        all_vid_names = set()
        for i, item in enumerate(examples):

            vidName  = item['Input.Video'].split("/")[-1][:-4]

            if len(vidName) == 0:
                continue

            #--------------------------------------------------
            # Read video clips
            #--------------------------------------------------
            str2read = '%s%s.mov'%(vid_path, vidName) # original resolution: 720x1280
            print(str2read)
            all_vid_names.add(str2read)
            if not os.path.exists(str2read):
                print(bcolors.BOLD + "Download video clips: {}".format(vidName) + bcolors.ENDC)
                wget.download(item['Input.Video'], out=str2read)
            else:
                #Already downloaded
                continue

        print(i, len(all_vid_names))


