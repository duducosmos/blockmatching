#!/usr/bin/env python
# -*- Codigin: UTF-8 -*-
"""unit test form layers decorator."""
import unittest
import os

import cv2
from blockmatching import *
from numpy import  hstack, vstack


CDIR = os.path.dirname(os.path.abspath(__file__))

VIDEOTEST = CDIR + '/videos/car.mp4'
OUTPUT = CDIR + '/videos/output.mp4'
output = SaveVideo(OUTPUT)

@dlayers(0.01, 19, 19, 7)
def background(videofile):
    cap = cv2.VideoCapture(videofile)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            cap.release()
            break
        yield frame


for bg, fg, mask, meand, layers in background(VIDEOTEST):

    sizex = 1
    sizey = 1

    if mask is not None:
        try:
            img = cv2.add(fg, mask)
        except:
            img = fg

        lyr = layers[0]

        '''
        msk = lyr == 0
        tmp = cv2.subtract(bg, lyr)
        tmp[msk] = 0
        lyr[tmp == 0] = 0
        '''

        img = cv2.resize(img, (0,0), fx=sizex, fy=sizey)
        bg = cv2.resize(bg, (0,0), fx=sizex, fy=sizey)

        lyr = cv2.resize(lyr, (0,0), fx=sizex, fy=sizey)

        bg = hstack((bg, img))
        bg = hstack((bg, lyr))

        cv2.imshow('Foreground', bg)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        output.save_frame(bg, False)

output.save_frame(bg, True)
cv2.destroyAllWindows()
