# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:48:57 2018

keyframes extract tool

this key frame extract algorithm is based on interframe difference.

The principle is very simple
First, we load the video and compute the interframe difference between each frames

Then, we can choose one of these three methods to extract keyframes, which are 
all based on the difference method:
    
1. use the difference order
    The first few frames with the largest average interframe difference 
    are considered to be key frames.
2. use the difference threshold
    The frames which the average interframe difference are large than the 
    threshold are considered to be key frames.
3. use local maximum
    The frames which the average interframe difference are local maximum are 
    considered to be key frames.
    It should be noted that smoothing the average difference value before 
    calculating the local maximum can effectively remove noise to avoid 
    repeated extraction of frames of similar scenes.

After a few experiment, the third method has a better key frame extraction effect.

The original code comes from the link below, I optimized the code to reduce 
unnecessary memory consumption.
https://blog.csdn.net/qq_21997625/article/details/81285096

@author: zyb_as

Modifid on Fri Jan 25 2019
add:
    key frame extrated from videos folder, videos_process()
    images rename from images folder, rename_process()
    image name filter from image folder, image_rename_process()
"""

import os
import time
import argparse
import glob
import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema

 
def smooth(x, window_len=13, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
        
    example:
    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """
    # print(len(x), window_len)
    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
 
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    #print(len(s))
 
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]
 

class Frame:
    """class to hold information about each frame
    
    """
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff
 
    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id
 
    def __gt__(self, other):
        return other.__lt__(self)
 
    def __eq__(self, other):
        return self.id == other.id and self.id == other.id
 
    def __ne__(self, other):
        return not self.__eq__(other)
 
 
def rel_change(a, b):
   x = (b - a) / max(a, b)
   print(x)
   return x


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='key frames extract from video.')
    parser.add_argument('--input', dest='input',
                        help='Input video folder',
                        default=None, type=str)
    parser.add_argument('--output', dest='output',
                        help='Output result folder',
                        default="/opt/tmp/movie/dst", type=str)

    args = parser.parse_args()

    if not args.input:
        parser.error('Input folder not given')
    return args


def videos_process():
    args = parse_args()
    # Setting fixed threshold criteria
    USE_THRESH = False
    # fixed threshold value
    THRESH = 0.6
    # Setting fixed threshold criteria
    USE_TOP_ORDER = False
    # Setting local maxima criteria
    USE_LOCAL_MAXIMA = True
    # Number of top sorted frames
    NUM_TOP_FRAMES = 50
    len_window = int(50)

    videos_path = glob.glob(args.input + "/*")
    print(videos_path)
    result_path = args.output
    count=0
    video_nums=len(videos_path)

    def _frames_process():
        # load video and compute diff between frames
        dst=result_path +"/"+videoname+"/"
        print(dst)
        if not os.path.exists(dst):
            os.makedirs(dst.rstrip("/"))
        cap = cv2.VideoCapture(str(video_path))
        curr_frame = None
        prev_frame = None
        frame_diffs = []
        frames = []
        success, frame = cap.read()
        i = 0
        while (success):
            luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
            curr_frame = luv
            if curr_frame is not None and prev_frame is not None:
                # logic here
                diff = cv2.absdiff(curr_frame, prev_frame)
                diff_sum = np.sum(diff)
                diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
                frame_diffs.append(diff_sum_mean)
                frame = Frame(i, diff_sum_mean)
                frames.append(frame)
            prev_frame = curr_frame
            i = i + 1
            success, frame = cap.read()
        cap.release()

        # compute keyframe
        keyframe_id_set = set()
        if USE_TOP_ORDER:
            # sort the list in descending order
            frames.sort(key=operator.attrgetter("diff"), reverse=True)
            for keyframe in frames[:NUM_TOP_FRAMES]:
                keyframe_id_set.add(keyframe.id)
        if USE_THRESH:
            print("Using Threshold")
            for i in range(1, len(frames)):
                if (rel_change(np.float(frames[i - 1].diff), np.float(frames[i].diff)) >= THRESH):
                    keyframe_id_set.add(frames[i].id)
        if USE_LOCAL_MAXIMA:
            # print("Using Local Maxima")
            diff_array = np.array(frame_diffs)
            sm_diff_array = smooth(diff_array, len_window)
            frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
            for i in frame_indexes:
                keyframe_id_set.add(frames[i - 1].id)

            plt.switch_backend('agg')
            plt.figure(figsize=(40, 20))
            plt.locator_params(numticks=100)
            plt.stem(sm_diff_array)
            plt.savefig(dst + 'plot.png')

        # save all keyframes as image
        cap = cv2.VideoCapture(str(video_path))
        curr_frame = None
        keyframes = []
        success, frame = cap.read()
        idx = 0
        while (success):
            if idx in keyframe_id_set:
                name = videoname+"_" + str(idx) + ".jpg"
                cv2.imwrite(dst+ name, frame)
                keyframe_id_set.remove(idx)
            idx = idx + 1
            success, frame = cap.read()
        cap.release()

    for video_path in videos_path:
        count+=1
        print("Processing video %d/%d"%(count,video_nums))
        videoname = str(video_path.split('/')[-1]).split(".")[-2]
        t_start=time.time()
        _frames_process()
        print("time = %f s"%(time.time()-t_start))


def rename_process():
    args = parse_args()
    input=args.input
    output = args.output
    videos_name = os.listdir(args.input)
    video_nums = len(videos_name)
    print("Videos num : %d"%video_nums)
    count = 0

    def _rename():
        try:
            path=input + "/" + video_name
            if os.path.isdir(path):
                images_name = os.listdir(path)
                for image_name in images_name:
                    rename=image_name.replace("keyframe",video_name)
                    src=path+"/"+image_name
                    dst=path+"/"+rename
                    os.rename(src,dst)
        except Exception as e:
            print(e)

    for video_name in videos_name:
        count+=1
        print("Processing image %d/%d"%(count,video_nums))
        _rename()


def image_rename_process():
    '''
    处理：
    com]风骚律师第一季第10集[中英双字]_9897.jpg
    到：
    风骚律师第一季第10集_9897.jpg
    '''
    args = parse_args()
    input=args.input
    output = args.output
    images_name = os.listdir(args.input)
    image_nums = len(images_name)
    print("Videos num : %d"%image_nums)
    count = 0

    def _rename():
        try:
            temp=image_name.split("]")
            rename=temp[1].split("[")[0]+temp[2]
            src = input + "/" + image_name
            dst = input + "/" + rename
            os.rename(src, dst)
        except Exception as e:
            print(e)

    for image_name in images_name:
        count+=1
        if count%1000==0:
            print("Processing image %d/%d"%(count,image_nums))
        _rename()


if __name__ == "__main__":
    print("start processsing...")
    start = time.time()
    videos_process()
    # rename_process()
    # image_rename_process()
    end = time.time()
    print("total time = %f s"%(end - start))
    print("Done!")





