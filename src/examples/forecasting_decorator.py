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

@forecasting(60, 1, 0.01, 17, 17, 3)
def background(videofile):
    cap = cv2.VideoCapture(videofile)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (0,0), fx=1 / 2, fy=1 / 2)
        else:
            cap.release()
            break
        yield frame


for frame, forecasting, foreground   in background(VIDEOTEST):

    sizex = 1 / 2
    sizey = 1 / 2

    if forecasting is not None:
        img = cv2.resize(frame, (0,0), fx=sizex, fy=sizey)
        fct = cv2.resize(forecasting, (0,0), fx=sizex, fy=sizey)
        fgd = cv2.resize(foreground, (0,0), fx=sizex, fy=sizey)


        frct = hstack((img, fct))
        frct = hstack((frct, fgd))

        cv2.imshow('Forecasting', frct)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        output.save_frame(frct, False)

output.save_frame(frct, True)
cv2.destroyAllWindows()
