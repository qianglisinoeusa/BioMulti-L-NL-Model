import os
import sys
import cv2
import numpy as np

def convert_video_to_numpy(video_path, frames, width, height, channel):
    video_array = np.zeros((frames,width,height,channel))
    cap = cv2.VideoCapture(video_path)
    contFrame = 0
    while (contFrame<frames):
        ret, frame = cap.read()
        if ret == True:
            frame = np.reshape(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (width,height,channel))
            video_array[contFrame,...] = frame
            contFrame += 1
            print(contFrame)
    return np.array(video_array/255)

video_path = '/home/qiang/QiangLi/Computer_Vision/video_datasets/vistasoft-master/ucf_sports_actions/ucf_action/Riding_Horse/001/'
saveVideoPath  = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/WECSF/video/'
frames=60
width=720
height=404
channel=3

data_test_video = os.path.join(video_path + '4456-16_700040' + '.avi')
video_input = convert_video_to_numpy(data_test_video, frames, width, height, channel)
print(video_input.shape[0])
