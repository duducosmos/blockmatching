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
OUTPUT = CDIR + '/videos/output.avi'

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

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT, fourcc, 20.0, (640,480))

for bg, fg, mask, meand, layers in background(VIDEOTEST):

    sizex = 1 / 4
    sizey = 1 / 4

    if mask is not None:
        try:
            img = cv2.add(fg, mask)
        except:
            img = fg

        img = cv2.resize(img, (0,0), fx=sizex, fy=sizey)
        bg = cv2.resize(bg, (0,0), fx=sizex, fy=sizey)
        lyr = layers[0]
        lyr = cv2.resize(lyr, (0,0), fx=sizex, fy=sizey)

        bg = hstack((bg, img))
        bg = hstack((bg, lyr))

        cv2.imshow('Foreground', bg)
        out.write(bg)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

out.release()
cv2.destroyAllWindows()
