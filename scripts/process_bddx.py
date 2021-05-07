import os
import csv
import glob
import json
import time

from sys import platform
from random import shuffle
from collections import defaultdict

import cv2
import h5py
import numpy as np
import scipy.misc
import skvideo.io
import skvideo.datasets

from tqdm import tqdm

from scipy import interpolate
from scipy.ndimage import rotate

import bddx_helper

bddx_path = "/datasets_local/BDD-X"
annotation_path = os.path.join(bddx_path, "BDD-X-Annotations_v1.csv")
video_path = os.path.join(bddx_path, "videos")
output_path = os.path.join(bddx_path, "processed")

# JSON format
data = {}
data['annotations'] = []
data['info']        = []
data['videos']      = []

# Parameters
maxItems = 15 # Each csv file contains action / justification pairs of 15 at maximum
CHUNKSIZE = 20 # To store in h5 files

# Read information about video clips
with open(annotation_path, "r") as f:
    annotations = csv.DictReader(f, delimiter=',')
    '''
    Keys:
        1. Input.Video, 2. Answer.1start, 3. Answer.1end, 4. Answer.1action, 5. Answer.1justification
    '''
    captionid   = 0
    videoid     = 0
    vidNames    = []
    vidNames_notUSED = []
    vid_not_used = defaultdict(list)
    for annotation in tqdm(annotations):

        vidName  = annotation['Input.Video'].split("/")[-1][:-4]
        vid_unique = str(videoid) + "_" + str(vidName)

        # removes bad videos
        if len(vidName) == 0:
            vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
            vid_not_used["no_name"].append(vid_unique)
            continue
        if len(annotation["Answer.1start"]) == 0:
            vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
            vid_not_used["no_start"].append(vid_unique)
            continue
        if len(annotation["Answer.1justification"]) == 0:
            vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
            vid_not_used["no_justif"].append(vid_unique)
            continue

        videoid += 1

        #--------------------------------------------------
        # 1. Control signals
        #--------------------------------------------------
        str2find  = os.path.join(bddx_path, "info", "%s.json" % vidName)
        json2read = glob.glob(str2find)

        if json2read:
            json2read = json2read[0]
        else:
            vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
            vid_not_used["info_not_found"].append(vid_unique)
            continue

        # keys: timestamp, longitude, course, latitude, speed, accuracy
        timestamp, longitude, course, latitude, speed, accuracy, gps_x, gps_y = [], [], [], [], [], [], [], []
        with open(json2read) as json_data:
            trajectories = json.load(json_data)['locations']
            for trajectory in trajectories:
                timestamp.append(trajectory['timestamp']) # timestamp
                longitude.append(trajectory['longitude']) # gps longitude
                course.append(trajectory['course']) # angle of the car (degree)
                latitude.append(trajectory['latitude']) # gps latitude
                speed.append(trajectory['speed']) # speed intensity (in m/s ?)
                accuracy.append(trajectory['accuracy']) # ???

                # gps to flatten earth coordinates (meters)
                _x, _y, _ = bddx_helper.lla2flat( (trajectory['latitude'], trajectory['longitude'], 1000.0),
                                             (latitude[0], longitude[0]), 0.0, -100.0)
                gps_x.append(_x)
                gps_y.append(_y)

        # Use interpolation to prevent variable periods
        if np.array(timestamp).shape[0] < 2:
            vidNames_notUSED.append(str(videoid) + "_" + str(vidName))
            vid_not_used["interpolation_impossible"].append(vid_unique)
            continue

        # extract equally-spaced points (at the meter precision ?)
        # cumulative_dist_along_path: cumulative distance (in meter?) along the path. Size = (#seconds - 1)
        # dist_steps: integer numbers from 0 to then number of meters that were done in the video (the last point is not an integer). Size = (#meters + 1)
        # points: (x,y) coordinates taken along the path, every meters. Size = (#meters + 1)
        points, dist_steps, cumulative_dist_along_path = bddx_helper.get_equally_spaced_points( gps_x, gps_y )

        # Generate target direction
        # Get angle between the current vehicle orientation and the final vehicle position (at the end of the session) at every meter
        goalDirection_equal  = bddx_helper.get_goalDirection( dist_steps, points )
        goalDirection_interp = interpolate.interp1d(dist_steps, goalDirection_equal)
        # Get angle between the current vehicle orientation and the final vehicle position (at the end of the session) at every second
        goalDirection        = goalDirection_interp(cumulative_dist_along_path)

        # Generate curvatures / accelerator
        # Get curvarture at every meter
        curvature_raw = bddx_helper.compute_curvature(points[0], points[1])
        curvature_interp = interpolate.interp1d(dist_steps, curvature_raw)
        # Get curvature at every second
        curvature = curvature_interp(cumulative_dist_along_path)

        # Get acceleration as the derivative of the speed
        accelerator = np.gradient(speed)

        #--------------------------------------------------
        # 2. Captions
        #--------------------------------------------------
        nEx = 0
        for segment in range(maxItems - 1):
            sTime           = annotation["Answer.{}start".format(segment + 1)]
            eTime           = annotation["Answer.{}end".format(segment + 1)]
            action          = annotation["Answer.{}action".format(segment + 1)]
            justification   = annotation["Answer.{}justification".format(segment + 1)]

            if not sTime or not eTime or not action or not justification:
                continue

            nEx         += 1
            captionid   += 1

            # Info
            feed_dict = { 'contributor':    'Berkeley DeepDrive',
                          'date_created':   time.strftime("%d/%m/%Y"),
                          'description':    'This is 0.1 version of the BDD-X dataset',
                          'url':            'https://deepdrive.berkeley.edu',
                          'year':           2017}
            data['info'].append(feed_dict)

            # annotations
            feed_dict = { 'action':         action,
                          'justification':  justification,
                          'sTime':          sTime,
                          'eTime':          eTime,
                          'id':             captionid,
                          'vidName':        vidName,
                          'video_id':       videoid,
                          }
            data['annotations'].append(feed_dict)

            # Video
            feed_dict = { 'url':            annotation['Input.Video'],
                          'video_name':     vidName,
                          'height':         720,
                          'width':          1280,
                          'video_id':       videoid,
                           }

            data['videos'].append(feed_dict)


        #--------------------------------------------------
        # 3. Read video clips
        #--------------------------------------------------
        str2read = os.path.join(bddx_path, "videos", "%s.mov" % vidName) # original image: 720x1280
        frames   = []
        cnt      = 0
        scalefactor = 1

        if os.path.isfile(os.path.join(output_path, "cam", "%s_%s.h5" % (videoid, vidName))):
            #print('File already generated (decoding): {}'.format(str(videoid) + "_" + str(vidName)))
            pass

        elif os.path.exists(str2read):
            metadata = skvideo.io.ffprobe(str2read)

            if ("side_data_list" in metadata["video"].keys()) == False:
                rotation = 0
            else:
                rotation = float(metadata["video"]["side_data_list"]["side_data"]["@rotation"])

            cap = cv2.VideoCapture(str2read)
            nFrames, img_width, img_height, fps = bddx_helper.get_vid_info(cap)

            print('ID: {}, #Frames: {}, nGPSrecords: {}, Image: {}x{}, FPS: {}'.format(vidName, nFrames, len(trajectories), img_width, img_height, fps))

            for i in tqdm(range(nFrames)):
                gotImage, frame = cap.read()
                cnt += 1
                if gotImage:
                    if cnt % 3 == 0: # reduce to 10Hz
                        frame = frame.swapaxes(1,0)

                        if rotation > 0:
                            frame = cv2.flip(frame,0)
                        elif rotation < 0:
                            frame = cv2.flip(frame,1)
                        else:
                            frame = frame.swapaxes(1,0)

                        frame = cv2.resize(frame, None, fx=0.125*scalefactor, fy=0.125*scalefactor)

                        assert frame.shape == (90*scalefactor, 160*scalefactor, 3)

                        if cnt %100 == 0:
                            #cv2.imshow('image', frame)
                            #cv2.waitKey(10)
                            cv2.imwrite('sample.png',frame)

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 640x360x3

                        frames.append(frame)

            cap.release()
        else:
            print('ERROR: Unable to open video {}'.format(str2read))
            break

        frames = np.array(frames).astype(int)



        #--------------------------------------------------
        # 4. Saving
        #--------------------------------------------------
        vidNames.append(str(videoid) + "_" + str(vidName))

         # Video is stored at 10Hz rate
        if not os.path.isfile(os.path.join(output_path, "cam", "%s_%s.h5" % (videoid, vidName))):
            cam = h5py.File(os.path.join(output_path, "cam", "%s_%s.h5" % (videoid, vidName)), "w")
            dset = cam.create_dataset("/X",         data=frames,   chunks=(CHUNKSIZE,90*scalefactor,160*scalefactor,3), dtype='uint8')
        else:
             #print('File already generated (cam): {}'.format(str(videoid) + "_" + str(vidName)))
            pass

         # Log are sotred at 1Hz rate
        if not os.path.isfile(os.path.join(output_path, "log", "%s_%s.h5" % (videoid, vidName))):
            log = h5py.File(os.path.join(output_path, "log", "%s_%s.h5" % (videoid, vidName)), "w")
            dset = log.create_dataset("/timestamp", data=timestamp)
            dset = log.create_dataset("/longitude", data=longitude)
            dset = log.create_dataset("/course",    data=course)
            dset = log.create_dataset("/latitude",  data=latitude)
            dset = log.create_dataset("/speed",     data=speed)
            dset = log.create_dataset("/accuracy",  data=accuracy)
            #dset = log.create_dataset("/fps",       data=fps)
            dset = log.create_dataset("/curvature",  data=curvature, 	 dtype='float')
            dset = log.create_dataset("/accelerator",data=accelerator, 	 dtype='float')
            dset = log.create_dataset("/goaldir",    data=goalDirection, dtype='float')
        else:
            pass
            #print('File already generated (log): {}'.format(str(videoid) + "_" + str(vidName)))

with open(os.path.join(output_path, 'captions_BDDX.json'), 'w') as outfile:
    json.dump(data, outfile)

np.save(os.path.join(output_path, 'vidNames_notUSED.npy'), vidNames_notUSED)
