import cv2
import glob
import os
import scipy.io as scio
import numpy as np
import random
import time
from datetime import timedelta


from tqdm import tqdm

import argparse
import torch
import imageio
import moviepy.editor as mp


from framework import Stylization

## -------------------
##  Parameters

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


# Path of the checkpoint (please download and replace the empty file)
# i.e. your pretrained model (weights) by the authors of the paper
checkpoint_path = "./Model/style_net-TIP-final.pth"

# Device settings, use cuda if available
cuda = torch.cuda.is_available()
# device = 'cpu' if args.cpu else 'cuda'

# The proposed Sequence-Level Global Feature Sharing
use_Global = True

# Where to save the results
results_base = '../results'
if not os.path.exists(results_base):
    os.mkdir(results_base)

result_videos_path = os.path.join(results_base, 'video')
if not os.path.exists(result_videos_path):
    os.mkdir(result_videos_path)


## -------------------
##  Tools

def read_img(img_path):
    return cv2.imread(img_path)


class ReshapeTool():
    def __init__(self):
        self.record_H = 0
        self.record_W = 0

    def process(self, img):
        H, W, C = img.shape

        if self.record_H == 0 and self.record_W == 0:
            new_H = H + 128
            if new_H % 64 != 0:
                new_H += 64 - new_H % 64

            new_W = W + 128
            if new_W % 64 != 0:
                new_W += 64 - new_W % 64

            self.record_H = new_H
            self.record_W = new_W

        new_img = cv2.copyMakeBorder(img, 64, self.record_H-64-H,
                                          64, self.record_W-64-W, cv2.BORDER_REFLECT)
        return new_img




## -------------------
##  Preparation

def process_video(style_img, input_video, interval = 8, write_frames_to_disk = False):

    start_time_process = time.monotonic()

    # Read style image
    if not os.path.exists(style_img):
        exit('Style image %s does not exist (typo on your path?)'%(style_img))
    style = cv2.imread(style_img)
    style_fname = os.path.split(style_img)[1]
    print('Opened style image "{}"'.format(style_fname))

    # Read input video
    video_fname = os.path.split(input_video)[1]
    if not os.path.exists(input_video):
        exit('Input video %s does not exists (typo on your path?)' % (input_video))
    video = imageio.get_reader(input_video)
    fps = video.get_meta_data()['fps']
    print('Opened input video "{}" for style transfer (fps = {})'.format(video_fname, fps))

    # TODO! could be nicer to use just one library for image and audio data
    # https://towardsdatascience.com/extracting-audio-from-video-using-python-58856a940fd
    my_clip = mp.VideoFileClip(input_video)
    audio_track = my_clip.audio
    audio_sampling_freq = audio_track.fps
    audio_chs = audio_track.nchannels
    print('Opened the audio of the input video f_sampling = {} Hz, number of channels = {}'.format(audio_sampling_freq, audio_chs))

    # TODO! modify here if you do not like the output filename convention
    name = 'ReReVST-' + style_fname + '-' + video_fname
    if not use_Global:
        name = name + '-no-global'

    # Build model
    start_time = time.monotonic()
    framework = Stylization(checkpoint_path, cuda, use_Global)
    framework.prepare_style(style)
    end_time = time.monotonic()
    print('Stylization Model built in {}'.format(timedelta(seconds=end_time - start_time)))

    # Build tools
    reshape = ReshapeTool()

    ## -------------------
    ##  Inference

    # TODO! isn't there a way to just get the number of frames without this?
    for i, frame in enumerate(video):
        frame_num = i+1
    print('Number of frames in the input video = {} (length {:.2f} seconds)'.format(frame_num, frame_num/fps))

    # Prepare for proposed Sequence-Level Global Feature Sharing
    if use_Global:

        start_time = time.monotonic()
        print('Preparations for Sequence-Level Global Feature Sharing')
        framework.clean()
        sample_sum = (frame_num-1)//interval # increase interval if you run out of CUDA mem, e.g. to 16
        # TODO! it is actually the number of frames used (sample_sum), so you could do autocheck for this
        #  based on your hardware, and the processed video, and automatically increase the interval

        # get frame indices to be used
        indices = list()
        print('Using a total of {} frames to do global feature sharing (trying to use too many might result memory running out)'.format(sample_sum))
        for s in range(sample_sum):
            i = s * interval
            indices.append(i)
            # print(' add frame %d , %d frames in total'%(i, sample_sum))

        # actually adding the frames once we know the indices
        no_of_frames_added = 0
        for i, frame in enumerate(video):
            if i in indices or i == frame_num-1: # add the last frame always (from original code)
                no_of_frames_added += 1
                framework.add(frame)

        if no_of_frames_added != sample_sum+1:
            print(' -- for some reason reason you did not add all the frames picked to be added?')

        print('Computing global features')
        framework.compute()

        end_time = time.monotonic()
        print('Preparations finished in {}!'.format(timedelta(seconds=end_time - start_time)))


    # Main stylization
    video_path_out = os.path.join(result_videos_path, name)
    writer = imageio.get_writer(video_path_out, fps=fps)

    # go through the video frames
    start_time = time.monotonic()
    print('Applying style transfer to the video')
    for i, frame in tqdm(enumerate(video)):

        # Crop the image
        H,W,C = frame.shape
        new_input_frame = reshape.process(frame)

        # Stylization
        styled_input_frame = framework.transfer(new_input_frame)

        # Crop the image back
        styled_input_frame = styled_input_frame[64:64+H,64:64+W,:]

        # cast as unsigned 8-bit integer (not necessarily needed)
        styled_input_frame = styled_input_frame.astype('uint8')

        # add to the output video
        # https://imageio.readthedocs.io/en/stable/examples.html
        writer.append_data(styled_input_frame)

    writer.close()
    end_time = time.monotonic()
    print('Video style transferred in {}'.format(timedelta(seconds=end_time - start_time)))

    audio_path_out = video_path_out.replace('mp4', 'mp3')
    print('writing audio (quick and dirty solution) to disk as separate .mp3 (TODO! combine with the video and get rid of this extra step)')
    print(' path for audio = {}'.format(audio_path_out))
    my_clip.audio.write_audiofile(audio_path_out)

    if write_frames_to_disk:

        print('TODO! if your subsequent workflow would prefer PNG frames instead of a video')

        result_frames_path = os.path.join(results_base, 'frames')
        if not os.path.exists(result_frames_path):
            os.mkdir(result_frames_path)

        # Mkdir corresponding folders
        if not os.path.exists('{}/{}'.format(result_frames_path, name)):
            os.mkdir('{}/{}'.format(result_frames_path, name))

    end_time_process = time.monotonic()
    print('Prcessing as a whole took {}'.format(timedelta(seconds=end_time_process - start_time_process)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training of segmentation model for sCROMIS2ICH dataset')
    parser.add_argument('-style_img', '--style_img', type=str, default='../inputs/styles/620ef3a5cd28cd75b742341cc8433eb8.jpg',
                        help='Style img (e.g. jpeg or png)')
    parser.add_argument('-input_video', '--input_video', type=str, default='../inputs/video/scatman.mp4',
                        help='Video input that you want to style (e.g. MP4)')
    parser.add_argument('-write_frames_to_disk', '--write_frames_to_disk', type=bool, default=False,
                        help='Writes the frames to disk as well')
    parser.add_argument('-interval', '--interval', type=int, default=8,
                        help="Affects the number of frames needed for 'Sequence-Level Global Feature Sharing', "
                             "i.e. can make your RAM run out if too small (make maybe automatic at point, default was = 8),"
                             "but if you large videos with big resolutions, you might need to ")
    args = parser.parse_args()

    process_video(style_img = os.path.join(DIR_PATH, args.style_img),
                  input_video = os.path.join(DIR_PATH, args.input_video),
                  interval = args.interval,
                  write_frames_to_disk = args.write_frames_to_disk)

    # TODO! you could do a "style_dir" here, so that you process the same video with multiple styles in a loop
    #  i.e. you have a 50 style imgs on a folder and you do not know which works, and you could have the processing
    #  run like overnight, and see the results in the morning

    # TODO! similarly you could have multiple input video(s), and you batch process them with multiple style imgs